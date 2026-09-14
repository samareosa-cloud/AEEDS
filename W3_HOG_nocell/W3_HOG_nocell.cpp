#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace cv;
using std::vector;

constexpr float PI = 3.14159265358979323846f;


// HOG 추출 결과
struct HogResult {
    vector<float> descriptor;      // 모든 block의 histogram을 연결한 HOG descriptor
    vector<float> finalHistogram;  // 전체 block을 방향별로 합친 최종 9-bin histogram
};


// Cell을 나누지 않는 HOG 추출
// block 크기: 16×16
// block 이동 간격: 8 pixel
// block당 histogram: 9-bin
static HogResult extractHogNoCell(
    const Mat& input,
    int block,
    int interval,
    int nbins,
    float eps,
    const char* csvPath)
{
    // 이미지 크기
    const int width = input.cols;
    const int height = input.rows;

    // x, y 방향 gradient 계산을 위한 3×3 Prewitt 마스크
    const int maskX[9] = {
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1
    };

    const int maskY[9] = {
        -1, -1, -1,
         0,  0,  0,
         1,  1,  1
    };

    // 반복문과 좌표 계산에 사용하는 변수
    int x, y;           // 입력 이미지의 현재 픽셀 좌표
    int xx, yy;         // 3×3 마스크가 적용되는 주변 픽셀 좌표
    int kx, ky;         // 3×3 마스크 내부의 x, y 인덱스
    int bin;            // 현재 픽셀의 gradient 방향이 속하는 histogram bin
    int i;              // histogram과 배열을 순회하기 위한 반복 변수
    int nx, ny;         // 가로 방향과 세로 방향의 block 개수
    int bx, by;         // 현재 block의 왼쪽 위 시작 좌표
    int blockIndex;     // 현재 처리 중인 block 번호

    // Gradient와 정규화 계산에 사용하는 변수
    float gx, gy;       // x 방향과 y 방향의 gradient 계산 결과
    float pixel;        // 0~1 범위로 변환한 현재 픽셀값
    float theta;        // 현재 픽셀의 gradient 방향(0° 이상 180° 미만)
    float sumSquares;   // L2 정규화를 위한 histogram 값의 제곱합
    float inverseNorm;  // L2 norm의 역수
    float normalized;   // 정규화된 histogram bin 값

    const uchar* row;   // 입력 이미지에서 현재 처리하는 행의 시작 주소
    FILE* fp;           // block별 histogram을 저장할 CSV 파일 포인터

    // 각 픽셀의 gradient 크기와 방향 bin 저장
    vector<float> magnitude(width * height, 0.0f);
    vector<unsigned char> binImage(width * height, 0);

    // 현재 block의 9-bin histogram
    vector<float> hist(nbins, 0.0f);

    // HOG 추출 결과
    HogResult result;


    // 1) 각 픽셀의 gradient magnitude와 orientation 계산
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {

            gx = 0.0f;
            gy = 0.0f;

            // 현재 픽셀을 중심으로 3×3 convolution
            for (yy = y - 1; yy <= y + 1; ++yy) {

                // 이미지의 위·아래 경계를 벗어나면 제외
                if (yy < 0 || yy >= height)
                    continue;

                // yy번째 행의 시작 주소
                row = input.ptr<uchar>(yy);

                for (xx = x - 1; xx <= x + 1; ++xx) {

                    // 이미지의 좌·우 경계를 벗어나면 제외
                    if (xx < 0 || xx >= width)
                        continue;

                    // 현재 픽셀에 대응하는 마스크 인덱스
                    ky = yy - (y - 1);
                    kx = xx - (x - 1);

                    // 픽셀값을 0~1 범위로 변환
                    pixel = row[xx] / 255.0f;

                    // x, y 방향 gradient 누적
                    gx += maskX[ky * 3 + kx] * pixel;
                    gy += maskY[ky * 3 + kx] * pixel;
                }
            }

            // Gradient 방향을 degree 단위로 계산
            theta = std::atan2(gy, gx) * 180.0f / PI;

            // 방향 범위를 [0, 180)으로 변환
            if (theta < 0.0f)
                theta += 180.0f;

            // 180도는 0도와 같은 방향으로 처리
            if (theta >= 180.0f)
                theta -= 180.0f;

            // Gradient 방향을 9개 bin으로 분류
            bin = static_cast<int>(theta / (180.0f / nbins));

            // 계산 오차로 bin 범위를 벗어나는 경우 방지
            if (bin >= nbins) bin = nbins - 1;

            // Gradient magnitude 저장
            magnitude[y * width + x] = std::sqrt(gx * gx + gy * gy);

            // Gradient orientation bin 저장
            binImage[y * width + x] = static_cast<unsigned char>(bin);
        }
    }


    // 2) 가로·세로 방향 block 개수 계산
    nx = (width - block) / interval + 1;
    ny = (height - block) / interval + 1;

    // Descriptor 크기만큼 메모리 미리 확보
    result.descriptor.reserve(nx * ny * nbins);

    // 최종 9-bin histogram을 0으로 초기화
    result.finalHistogram.assign(nbins, 0.0f);


    // 3) Block별 histogram을 저장할 CSV 파일 열기
    fp = NULL;
    fopen_s(&fp, csvPath, "w");

    if (fp != NULL) {
        std::fprintf(fp, "Block");

        // CSV 파일의 방향 구간 제목 작성
        for (i = 0; i < nbins; ++i) {
            std::fprintf(
                fp,
                ",%d-%d",
                i * 180 / nbins,
                (i + 1) * 180 / nbins
            );
        }

        std::fprintf(fp, "\n");
    }


    // 4) 16×16 block을 8픽셀씩 이동
    blockIndex = 0;

    for (by = 0; by <= height - block; by += interval) {
        for (bx = 0; bx <= width - block; bx += interval) {

            // 새로운 block을 계산하기 전 histogram 초기화
            for (i = 0; i < nbins; ++i) {
                hist[i] = 0.0f;
            }


            // 4-1) Block 내부의 gradient magnitude 누적
            for (y = by; y < by + block; ++y) {
                for (x = bx; x < bx + block; ++x) {

                    bin = binImage[y * width + x];

                    hist[bin] += magnitude[y * width + x];
                }
            }


            // 4-2) 현재 block histogram의 L2 크기 계산
            sumSquares = eps;

            for (i = 0; i < nbins; ++i) {
                sumSquares += hist[i] * hist[i];
            }

            inverseNorm = 1.0f / std::sqrt(sumSquares);


            // CSV에 block 번호 저장
            if (fp != NULL) {
                std::fprintf(fp, "%d", blockIndex + 1);
            }


            // 4-3) Block histogram을 정규화하여 descriptor에 연결
            for (i = 0; i < nbins; ++i) {

                normalized = hist[i] * inverseNorm;

                // 전체 HOG descriptor에 추가
                result.descriptor.push_back(normalized);

                // 같은 방향 bin끼리 누적
                result.finalHistogram[i] += normalized;

                // 정규화된 histogram을 CSV에 저장
                if (fp != NULL) {
                    std::fprintf(fp, ",%f", normalized);
                }
            }

            if (fp != NULL) {
                std::fprintf(fp, "\n");
            }

            ++blockIndex;
        }
    }


    // CSV 파일 닫기
    if (fp != NULL) {
        std::fclose(fp);
    }


    // 5) 전체 block을 합친 최종 9-bin histogram 정규화
    sumSquares = eps;

    for (i = 0; i < nbins; ++i) {
        sumSquares += result.finalHistogram[i] * result.finalHistogram[i];
    }

    inverseNorm = 1.0f / std::sqrt(sumSquares);

    for (i = 0; i < nbins; ++i) {
        result.finalHistogram[i] *= inverseNorm;
    }


    // HOG 정보 출력
    std::printf(
        "Blocks = %d x %d = %d\n",
        nx,
        ny,
        nx * ny
    );

    std::printf(
        "Descriptor dimension = %zu\n",
        result.descriptor.size()
    );

    return result;
}


// 동일한 histogram index에 있는 값의 절댓값 차이 평균
static double meanDifference(
    const vector<float>& a,
    const vector<float>& b)
{
    size_t i;                // 두 descriptor의 각 index를 순회하는 변수
    long double difference;  // 동일한 index에 있는 두 bin 값의 차이
    long double sum;         // 모든 bin의 절댓값 차이를 누적한 합

    // Descriptor의 크기가 다르거나 비어 있으면 비교 불가
    if (a.size() != b.size() || a.empty()) {
        return -1.0;
    }

    sum = 0.0L;

    // 동일한 index에 있는 bin 값의 절댓값 차이 누적
    for (i = 0; i < a.size(); ++i) {
        difference = static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
        sum += std::fabs(difference);
    }

    // 절댓값 차이의 합을 전체 bin 개수로 나눔
    return static_cast<double>(sum / a.size());
}


int main()
{
    // 이미지 경로
    const char* refPath = "C:/Users/sinbc/source/repos/W3_HOG/W3_HOG/LectureNote_03.bmp";

    const char* c1Path = "C:/Users/sinbc/source/repos/W3_HOG/W3_HOG/compare1.bmp";

    const char* c2Path = "C:/Users/sinbc/source/repos/W3_HOG/W3_HOG/compare2.bmp";

    // HOG 파라미터
    const int BLOCK = 16;
    const int INTERVAL = 8;
    const int NBINS = 9;
    const float EPS = 1e-6f;

    // 입력 이미지
    Mat ref;
    Mat c1;
    Mat c2;

    // 각 이미지의 HOG 추출 결과
    HogResult hRef;
    HogResult hC1;
    HogResult hC2;

    // 평균 절대 차이
    double mean1;
    double mean2;


    // 이미지를 grayscale로 불러오기
    ref = imread(refPath, IMREAD_GRAYSCALE);
    c1 = imread(c1Path, IMREAD_GRAYSCALE);
    c2 = imread(c2Path, IMREAD_GRAYSCALE);


    // 이미지가 정상적으로 불러와졌는지 확인
    if (ref.empty() || c1.empty() || c2.empty()) {
        std::printf("Image load failed!\n");
        return -1;
    }


    // 세 이미지의 크기가 같은지 확인
    if (ref.size() != c1.size()
        || ref.size() != c2.size()) {

        std::printf("Image size mismatch!\n");
        return -1;
    }


    // 기준 이미지의 HOG 추출
    hRef = extractHogNoCell(
        ref,
        BLOCK,
        INTERVAL,
        NBINS,
        EPS,
        "LectureNote_03_no_cell.csv"
    );

    // Compare1 이미지의 HOG 추출
    hC1 = extractHogNoCell(
        c1,
        BLOCK,
        INTERVAL,
        NBINS,
        EPS,
        "compare1_no_cell.csv"
    );

    // Compare2 이미지의 HOG 추출
    hC2 = extractHogNoCell(
        c2,
        BLOCK,
        INTERVAL,
        NBINS,
        EPS,
        "compare2_no_cell.csv"
    );


    // 기준 이미지와 각 비교 이미지의 평균 절대 차이 계산
    mean1 = meanDifference(
        hRef.descriptor,
        hC1.descriptor
    );

    mean2 = meanDifference(
        hRef.descriptor,
        hC2.descriptor
    );


    // 비교 결과 출력
    std::printf("\n[No Cell: Mean Difference]\n");

    std::printf("Lecture03 vs Compare1 = %.9f\n",mean1);

    std::printf("Lecture03 vs Compare2 = %.9f\n", mean2);


    // Mean Difference가 작은 이미지를 더 유사하다고 판단
    if (mean1 < mean2) std::printf("Compare1 is more similar.\n");
    else if (mean2 < mean1) std::printf("Compare2 is more similar.\n");
    else std::printf("Same difference.\n");
  
    return 0;
}
