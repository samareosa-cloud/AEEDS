#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace cv;
using std::vector;

const float PI = 3.14159265358979323846f;


// HOG 계산 결과
struct HogResult
{
    vector<float> descriptor;          // 전체 945차원 HOG
    vector<vector<float>> blockHist;   // 105개 Block의 9-bin
    vector<float> finalHistogram;      // 보고서 그래프용 최종 9-bin

    int blocksX;                       // 가로 Block 개수
    int blocksY;                       // 세로 Block 개수
};


// Cell을 나누지 않는 HOG 계산
static HogResult extractHOG(
    const Mat& input,
    int blockSize,
    int interval,
    int numberOfBins,
    float epsilon)
{
    int x, y, xx, yy;
    int bx, by;
    int i;
    int bin;
    int blockIndex;

    int height;
    int width;
    int blocksX;
    int blocksY;

    const int maskSize = 3;

    float convX;
    float convY;
    float pixel;
    float magnitudeValue;
    float direction;
    float binWidth;
    float sumSquares;
    float norm;

    // Prewitt mask
    int maskX[9] = {
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1
    };

    int maskY[9] = {
        -1, -1, -1,
         0,  0,  0,
         1,  1,  1
    };

    height = input.rows;
    width = input.cols;

    // 방향 범위 180도를 9개로 분할
    binWidth = 180.0f / numberOfBins;

    // 픽셀별 Magnitude 저장
    vector<float> magnitude(height * width, 0.0f);

    // 픽셀별 방향 bin 저장
    vector<int> binImage(height * width, 0);

    HogResult result;
 

    // 1. 모든 픽셀의 Gradient 계산
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {

            convX = 0.0f;
            convY = 0.0f;

            // 현재 픽셀 주변의 3×3 영역
            for (yy = y - maskSize / 2;
                yy <= y + maskSize / 2;
                yy++) {

                for (xx = x - maskSize / 2;
                    xx <= x + maskSize / 2;
                    xx++) {

                    // 이미지 범위 내부인지 확인
                    if (yy >= 0 && yy < height
                        && xx >= 0 && xx < width) {

                        // 픽셀값을 0~1로 변환
                        pixel = input.at<uchar>(yy, xx) / 255.0f;

                        // x 방향 Gradient
                        convX += maskX[ (yy - (y - 1)) * maskSize + (xx - (x - 1)) ] * pixel;

                        // y 방향 Gradient
                        convY += maskY[ (yy - (y - 1)) * maskSize + (xx - (x - 1)) ] * pixel;
                    }
                }
            }


            // Gradient Magnitude
            magnitudeValue = std::sqrt(convX * convX + convY * convY);

            // Gradient 방향: radian → degree
            direction = std::atan2(convY, convX) * 180.0f / PI;

            // 방향을 0° 이상 180° 미만으로 변환
            if (direction < 0.0f) {
                direction += 180.0f;
            }

            if (direction >= 180.0f) {
                direction -= 180.0f;
            }

            // 방향을 9개 bin 중 하나로 분류
            bin = static_cast<int>(direction / binWidth);

            // 계산 오차로 인한 범위 초과 방지
            if (bin >= numberOfBins) bin = numberOfBins - 1;

            // 현재 픽셀의 Magnitude 저장
            magnitude[y * width + x] = magnitudeValue;

            // 현재 픽셀의 방향 bin 저장
            binImage[y * width + x] = bin;
        }
    }


    // 2. Block 개수 계산
    blocksX = (width - blockSize) / interval + 1;

    blocksY = (height - blockSize) / interval + 1;

    result.blocksX = blocksX;
    result.blocksY = blocksY;

    // 전체 HOG 크기만큼 메모리 확보
    result.descriptor.reserve(blocksX * blocksY * numberOfBins);

    // 보고서 그래프용 최종 9-bin 초기화
    result.finalHistogram.assign(numberOfBins, 0.0f);


    // 3. 16×16 Block을 8픽셀 간격으로 이동
    blockIndex = 0;

    for (by = 0; by <= height - blockSize; by += interval) {

        for (bx = 0; bx <= width - blockSize; bx += interval) {

            // 현재 Block의 9-bin histogram
            vector<float> histogram(numberOfBins, 0.0f);


            // 4. 현재 16×16 Block 내부 픽셀 순회
            for (y = by; y < by + blockSize; y++) {

                for (x = bx; x < bx + blockSize; x++) {

                    // 현재 픽셀의 방향 bin
                    bin = binImage[y * width + x];

                    // 해당 방향 bin에 Magnitude 누적
                    histogram[bin] += magnitude[y * width + x];
                }
            }


            // 5. Block 단위 L2 normalization
            sumSquares = 0.0f;

            for (i = 0; i < numberOfBins; i++) {

                sumSquares += histogram[i] * histogram[i];
            }

            norm = std::sqrt(sumSquares + epsilon);


            // 6. 정규화 및 전체 Descriptor 연결
            for (i = 0; i < numberOfBins; i++) {

                histogram[i] /= norm;

                // Block의 9개 값을 전체 HOG 뒤에 연결
                result.descriptor.push_back(histogram[i]);

                // 보고서용 전체 9-bin에 누적
                result.finalHistogram[i] += histogram[i];
            }


            // Block별 정규화 Histogram 저장
            result.blockHist.push_back(histogram);

            blockIndex++;
        }
    }


    // 7. 보고서용 최종 9-bin 정규화
    sumSquares = 0.0f;

    for (i = 0; i < numberOfBins; i++) {

        sumSquares += result.finalHistogram[i] * result.finalHistogram[i];
    }

    norm = std::sqrt(sumSquares + epsilon);

    for (i = 0; i < numberOfBins; i++) {

        result.finalHistogram[i] /= norm;
    }

    return result;
}


// 각 이미지의 Block별 9-bin Histogram 저장
static void saveBlockHistograms(
    const HogResult& result,
    int numberOfBins,
    const char* fileName)
{
    FILE* file;
    int block;
    int bin;

    file = NULL;
    fopen_s(&file, fileName, "w");

    if (file == NULL) {
        std::printf(
            "File open failed: %s\n",
            fileName
        );
        return;
    }

    // CSV 제목
    std::fprintf(file, "Block");

    for (bin = 0; bin < numberOfBins; bin++) {

        std::fprintf(
            file,
            ",%d-%d",
            bin * 180 / numberOfBins,
            (bin + 1) * 180 / numberOfBins
        );
    }

    std::fprintf(file, "\n");


    // Block별 9-bin 값 저장
    for (block = 0; block < static_cast<int>(result.blockHist.size()); block++) {

        std::fprintf(
            file,
            "%d",
            block + 1
        );

        for (bin = 0; bin < numberOfBins; bin++) {

            std::fprintf(
                file,
                ",%.9f",
                result.blockHist[block][bin]
            );
        }

        std::fprintf(file, "\n");
    }

    std::fclose(file);
}


// 세 이미지의 최종 9-bin 저장
static void saveFinalHistograms(
    const HogResult& reference,
    const HogResult& compare1,
    const HogResult& compare2,
    int numberOfBins,
    const char* fileName)
{
    FILE* file;
    int bin;

    file = NULL;
    fopen_s(&file, fileName, "w");

    if (file == NULL) {
        std::printf(
            "File open failed: %s\n",
            fileName
        );
        return;
    }

    std::fprintf(
        file,
        "Degree,LectureNote_03,Compare1,Compare2\n"
    );

    for (bin = 0;
        bin < numberOfBins;
        bin++) {

        std::fprintf(
            file,
            "%d-%d,%.9f,%.9f,%.9f\n",
            bin * 180 / numberOfBins,
            (bin + 1) * 180 / numberOfBins,
            reference.finalHistogram[bin],
            compare1.finalHistogram[bin],
            compare2.finalHistogram[bin]
        );
    }

    std::fclose(file);
}


// 945개 Descriptor의 동일 index 차이 저장
static void saveDescriptorComparison(
    const vector<float>& reference,
    const vector<float>& compare1,
    const vector<float>& compare2,
    const char* fileName)
{
    FILE* file;
    size_t i;

    file = NULL;
    fopen_s(&file, fileName, "w");

    if (file == NULL) {
        std::printf(
            "File open failed: %s\n",
            fileName
        );
        return;
    }

    std::fprintf(
        file,
        "Index,LectureNote_03,Compare1,Compare2,"
        "Difference_1,Difference_2\n"
    );

    for (i = 0; i < reference.size(); i++) {

        std::fprintf(
            file,
            "%zu,%.9f,%.9f,%.9f,%.9f,%.9f\n",
            i,
            reference[i],
            compare1[i],
            compare2[i],
            std::fabs(reference[i] - compare1[i]),
            std::fabs(reference[i] - compare2[i])
        );
    }

    std::fclose(file);
}


// 동일한 Histogram index 값의 차이에 대한 평균
static double meanDifference(
    const vector<float>& first,
    const vector<float>& second)
{
    size_t i;
    long double sum;

    // Descriptor 크기 확인
    if (first.size() != second.size() || first.empty()) {

        return -1.0;
    }

    sum = 0.0L;

    // 동일 index의 절댓값 차이 누적
    for (i = 0; i < first.size(); i++) {

        sum += std::fabs(
            static_cast<long double>(first[i])
            - static_cast<long double>(second[i])
        );
    }

    // 전체 Descriptor 개수로 나눔
    return static_cast<double>(
        sum / first.size()
        );
}


int main()
{
    // 이미지 경로
    const char* referencePath =
        "C:/Users/sinbc/source/repos/W3_HOG/W3_HOG/LectureNote_03.bmp";

    const char* compare1Path =
        "C:/Users/sinbc/source/repos/W3_HOG/W3_HOG/compare1.bmp";

    const char* compare2Path =
        "C:/Users/sinbc/source/repos/W3_HOG/W3_HOG/compare2.bmp";


    // 과제 조건
    const int BLOCK_SIZE = 16;
    const int INTERVAL = 8;
    const int NUMBER_OF_BINS = 9;
    const float EPSILON = 1e-6f;


    Mat referenceImage;
    Mat compare1Image;
    Mat compare2Image;

    HogResult referenceHOG;
    HogResult compare1HOG;
    HogResult compare2HOG;

    double difference1;
    double difference2;


    // 이미지 읽기
    referenceImage = imread(
        referencePath,
        IMREAD_GRAYSCALE
    );

    compare1Image = imread(
        compare1Path,
        IMREAD_GRAYSCALE
    );

    compare2Image = imread(
        compare2Path,
        IMREAD_GRAYSCALE
    );


    // 이미지 로드 확인
    if (referenceImage.empty()
        || compare1Image.empty()
        || compare2Image.empty()) {

        std::printf("Image load failed!\n");
        return -1;
    }


    // 이미지 크기 확인
    if (referenceImage.size() != compare1Image.size()
        || referenceImage.size() != compare2Image.size()) {

        std::printf("Image size mismatch!\n");
        return -1;
    }


    // 각 이미지의 HOG 추출
    referenceHOG = extractHOG(
        referenceImage,
        BLOCK_SIZE,
        INTERVAL,
        NUMBER_OF_BINS,
        EPSILON
    );

    compare1HOG = extractHOG(
        compare1Image,
        BLOCK_SIZE,
        INTERVAL,
        NUMBER_OF_BINS,
        EPSILON
    );

    compare2HOG = extractHOG(
        compare2Image,
        BLOCK_SIZE,
        INTERVAL,
        NUMBER_OF_BINS,
        EPSILON
    );


    // Block별 Histogram 저장
    saveBlockHistograms(
        referenceHOG,
        NUMBER_OF_BINS,
        "LectureNote_03_blocks.csv"
    );

    saveBlockHistograms(
        compare1HOG,
        NUMBER_OF_BINS,
        "compare1_blocks.csv"
    );

    saveBlockHistograms(
        compare2HOG,
        NUMBER_OF_BINS,
        "compare2_blocks.csv"
    );


    // 보고서 그래프용 최종 9-bin 저장
    saveFinalHistograms(
        referenceHOG,
        compare1HOG,
        compare2HOG,
        NUMBER_OF_BINS,
        "final_histogram.csv"
    );


    // 945차원 값과 차이 저장
    saveDescriptorComparison(
        referenceHOG.descriptor,
        compare1HOG.descriptor,
        compare2HOG.descriptor,
        "descriptor_comparison.csv"
    );


    // 동일 index 값의 평균 절대 차이 계산
    difference1 = meanDifference(
        referenceHOG.descriptor,
        compare1HOG.descriptor
    );

    difference2 = meanDifference(
        referenceHOG.descriptor,
        compare2HOG.descriptor
    );


    // 결과 출력
    std::printf(
        "Image size = %d x %d\n",
        referenceImage.cols,
        referenceImage.rows
    );

    std::printf(
        "Blocks = %d x %d = %d\n",
        referenceHOG.blocksX,
        referenceHOG.blocksY,
        referenceHOG.blocksX
        * referenceHOG.blocksY
    );

    std::printf(
        "Dimension per block = %d\n",
        NUMBER_OF_BINS
    );

    std::printf(
        "HOG descriptor dimension = %zu\n",
        referenceHOG.descriptor.size()
    );

    std::printf(
        "\nLectureNote_03 vs Compare1 = %.9f\n",
        difference1
    );

    std::printf(
        "LectureNote_03 vs Compare2 = %.9f\n",
        difference2
    );


    // 평균 차이가 작은 이미지가 더 유사
    if (difference1 < difference2) {
        std::printf("\nCompare1 is more similar.\n");
    }
    else if (difference2 < difference1) {
        std::printf("\nCompare2 is more similar.\n");
    }
    else {
        std::printf("\nBoth images have the same difference.\n");
    }

    return 0;
}
