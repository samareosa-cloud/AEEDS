#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace cv;
using std::vector;

constexpr float PI = 3.14159265358979323846f;


// HOG 추출 결과
struct HogResult {
    vector<float> descriptor;      // 전체 HOG 벡터
    vector<float> finalHistogram;  // 최종 9-bin histogram
};


// Cell을 사용하는 HOG 추출
// 16×16 block → 8×8 cell 4개 → block당 36차원
static HogResult extractHogWithCell(
    const Mat& input,
    int block,
    int cell,
    int interval,
    int nbins,
    float eps,
    const char* csvPath)
{
    const int width = input.cols;    // 이미지 너비
    const int height = input.rows;   // 이미지 높이

    // 3×3 Prewitt 마스크
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

    int x, y;              // 이미지 픽셀 좌표
    int xx, yy;            // 주변 픽셀 좌표
    int kx, ky;            // 마스크 인덱스
    int bin;               // 방향 bin
    int i, c;              // 반복 변수
    int nx, ny;            // 가로·세로 block 개수
    int bx, by;            // block 시작 좌표
    int cx, cy;            // block 내부 cell 좌표
    int cellsPerSide;      // 한 변의 cell 개수
    int cellsPerBlock;     // block당 cell 개수
    int blockDimension;    // block당 HOG 차원
    int blockIndex;        // block 번호
    int cellIndex;         // block 내부 cell 번호

    float gx, gy;          // x, y 방향 gradient
    float pixel;           // 정규화된 픽셀값
    float theta;           // Gradient 방향
    float sumSquares;      // Histogram 제곱합
    float inverseNorm;     // L2 norm의 역수
    float normalized;      // 정규화된 bin 값

    const uchar* row;      // 이미지의 현재 행
    FILE* fp;              // CSV 파일 포인터

    // Block과 cell 구조 계산
    nx = (width - block) / interval + 1;
    ny = (height - block) / interval + 1;
    cellsPerSide = block / cell;
    cellsPerBlock = cellsPerSide * cellsPerSide;
    blockDimension = cellsPerBlock * nbins;

    // 픽셀별 gradient 정보
    vector<float> magnitude(width * height, 0.0f);
    vector<unsigned char> binImage(width * height, 0);

    // 현재 block의 36차원 histogram
    vector<float> blockHist(blockDimension, 0.0f);

    // HOG 추출 결과
    HogResult result;


    // 1) Gradient magnitude와 orientation 계산
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {

            gx = 0.0f;
            gy = 0.0f;

            // 3×3 convolution
            for (yy = y - 1; yy <= y + 1; ++yy) {

                if (yy < 0 || yy >= height) {
                    continue;
                }

                row = input.ptr<uchar>(yy);

                for (xx = x - 1; xx <= x + 1; ++xx) {

                    if (xx < 0 || xx >= width) {
                        continue;
                    }

                    ky = yy - (y - 1);
                    kx = xx - (x - 1);
                    pixel = row[xx] / 255.0f;

                    gx += maskX[ky * 3 + kx] * pixel;
                    gy += maskY[ky * 3 + kx] * pixel;
                }
            }

            // Gradient 방향 계산
            theta = std::atan2(gy, gx) * 180.0f / PI;

            // 방향을 [0, 180) 범위로 변환
            if (theta < 0.0f) {
                theta += 180.0f;
            }

            if (theta >= 180.0f) {
                theta -= 180.0f;
            }

            // 방향을 9개 bin으로 분류
            bin = static_cast<int>(
                theta / (180.0f / nbins)
                );

            if (bin >= nbins) {
                bin = nbins - 1;
            }

            // Gradient 크기와 방향 저장
            magnitude[y * width + x]
                = std::sqrt(gx * gx + gy * gy);

            binImage[y * width + x]
                = static_cast<unsigned char>(bin);
        }
    }


    // 2) 결과 벡터 초기화
    result.descriptor.reserve(
        nx * ny * blockDimension
    );

    result.finalHistogram.assign(
        nbins,
        0.0f
    );


    // 3) CSV 파일 열기
    fp = NULL;
    fopen_s(&fp, csvPath, "w");

    if (fp != NULL) {
        std::fprintf(fp, "Block");

        // Cell별 9-bin 열 제목 작성
        for (c = 0; c < cellsPerBlock; ++c) {
            for (i = 0; i < nbins; ++i) {
                std::fprintf(
                    fp,
                    ",Cell%d_%d-%d",
                    c + 1,
                    i * 180 / nbins,
                    (i + 1) * 180 / nbins
                );
            }
        }

        std::fprintf(fp, "\n");
    }


    // 4) 16×16 block을 8픽셀씩 이동
    blockIndex = 0;

    for (by = 0; by <= height - block; by += interval) {
        for (bx = 0; bx <= width - block; bx += interval) {

            // 새로운 block의 histogram 초기화
            for (i = 0; i < blockDimension; ++i) {
                blockHist[i] = 0.0f;
            }

            cellIndex = 0;


            // 4-1) Block을 8×8 cell로 분할
            for (cy = 0; cy < block; cy += cell) {
                for (cx = 0; cx < block; cx += cell) {

                    // 현재 cell 내부의 픽셀 순회
                    for (y = by + cy;
                        y < by + cy + cell;
                        ++y) {

                        for (x = bx + cx;
                            x < bx + cx + cell;
                            ++x) {

                            bin = binImage[y * width + x];

                            // 해당 cell의 방향 bin에 크기 누적
                            blockHist[cellIndex * nbins + bin]
                                += magnitude[y * width + x];
                        }
                    }

                    ++cellIndex;
                }
            }


            // 4-2) 36차원 block histogram L2 정규화
            sumSquares = eps;

            for (i = 0; i < blockDimension; ++i) {
                sumSquares += blockHist[i] * blockHist[i];
            }

            inverseNorm = 1.0f / std::sqrt(sumSquares);


            // CSV에 block 번호 저장
            if (fp != NULL) {
                std::fprintf(fp, "%d", blockIndex + 1);
            }


            // 4-3) 정규화된 값을 descriptor에 연결
            for (i = 0; i < blockDimension; ++i) {

                normalized = blockHist[i] * inverseNorm;

                result.descriptor.push_back(normalized);

                // 같은 방향 bin끼리 누적
                result.finalHistogram[i % nbins]
                    += normalized;

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


    // 5) 최종 9-bin histogram 정규화
    sumSquares = eps;

    for (i = 0; i < nbins; ++i) {
        sumSquares += result.finalHistogram[i]
            * result.finalHistogram[i];
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
        "Cells per block = %d\n",
        cellsPerBlock
    );

    std::printf(
        "Descriptor dimension = %zu\n",
        result.descriptor.size()
    );

    return result;
}


// 동일한 index의 bin 값 차이 평균
static double meanDifference(
    const vector<float>& a,
    const vector<float>& b)
{
    size_t i;                // Descriptor index
    long double difference;  // 두 bin 값의 차이
    long double sum;         // 절댓값 차이의 합

    if (a.size() != b.size() || a.empty()) {
        return -1.0;
    }

    sum = 0.0L;

    for (i = 0; i < a.size(); ++i) {

        difference
            = static_cast<long double>(a[i])
            - static_cast<long double>(b[i]);

        sum += std::fabs(difference);
    }

    return static_cast<double>(sum / a.size());
}


int main()
{
    // 이미지 경로
    const char* refPath
        = "C:/Users/sinbc/source/repos/W3_HOG/W3_HOG/LectureNote_03.bmp";

    const char* c1Path
        = "C:/Users/sinbc/source/repos/W3_HOG/W3_HOG/compare1.bmp";

    const char* c2Path
        = "C:/Users/sinbc/source/repos/W3_HOG/W3_HOG/compare2.bmp";

    // HOG 파라미터
    const int BLOCK = 16;       // Block 크기
    const int CELL = 8;         // Cell 크기
    const int INTERVAL = 8;     // Block 이동 간격
    const int NBINS = 9;        // 방향 bin 개수
    const float EPS = 1e-6f;    // 0 나눗셈 방지값

    Mat ref;    // 기준 이미지
    Mat c1;     // 비교 이미지 1
    Mat c2;     // 비교 이미지 2

    HogResult hRef;  // 기준 이미지 HOG
    HogResult hC1;   // Compare1 HOG
    HogResult hC2;   // Compare2 HOG

    double mean1;    // Ref와 Compare1의 차이
    double mean2;    // Ref와 Compare2의 차이


    // 이미지를 grayscale로 불러오기
    ref = imread(refPath, IMREAD_GRAYSCALE);
    c1 = imread(c1Path, IMREAD_GRAYSCALE);
    c2 = imread(c2Path, IMREAD_GRAYSCALE);


    // 이미지 로드 확인
    if (ref.empty() || c1.empty() || c2.empty()) {
        std::printf("Image load failed!\n");
        return -1;
    }


    // 이미지 크기 확인
    if (ref.size() != c1.size()
        || ref.size() != c2.size()) {

        std::printf("Image size mismatch!\n");
        return -1;
    }


    // 기준 이미지 HOG 추출
    hRef = extractHogWithCell(
        ref,
        BLOCK,
        CELL,
        INTERVAL,
        NBINS,
        EPS,
        "LectureNote_03_with_cell.csv"
    );

    // Compare1 HOG 추출
    hC1 = extractHogWithCell(
        c1,
        BLOCK,
        CELL,
        INTERVAL,
        NBINS,
        EPS,
        "compare1_with_cell.csv"
    );

    // Compare2 HOG 추출
    hC2 = extractHogWithCell(
        c2,
        BLOCK,
        CELL,
        INTERVAL,
        NBINS,
        EPS,
        "compare2_with_cell.csv"
    );


    // 평균 절대 차이 계산
    mean1 = meanDifference(
        hRef.descriptor,
        hC1.descriptor
    );

    mean2 = meanDifference(
        hRef.descriptor,
        hC2.descriptor
    );


    // 비교 결과 출력
    std::printf("\n[With Cell: Mean Difference]\n");

    std::printf(
        "Lecture03 vs Compare1 = %.9f\n",
        mean1
    );

    std::printf(
        "Lecture03 vs Compare2 = %.9f\n",
        mean2
    );


    // 값이 작은 이미지를 더 유사하다고 판단
    if (mean1 < mean2) {
        std::printf("Compare1 is more similar.\n");
    }
    else if (mean2 < mean1) {
        std::printf("Compare2 is more similar.\n");
    }
    else {
        std::printf("Same difference.\n");
    }

    return 0;
}