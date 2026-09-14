#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace cv;
using std::vector;

constexpr float PI = 3.14159265358979323846f;

struct HogResult {
    vector<float> descriptor;
    vector<float> finalHistogram;
};

static HogResult extractHogNoCell(const Mat& input, int block, int interval,
    int nbins, float eps, const char* csvPath)
{
    const int width = input.cols;
    const int height = input.rows;
    const int maskX[9] = { -1,0,1,-1,0,1,-1,0,1 };
    const int maskY[9] = { -1,-1,-1,0,0,0,1,1,1 };
    vector<float> magnitude(width * height, 0.0f);
    vector<unsigned char> binImage(width * height, 0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float gx = 0.0f, gy = 0.0f;
            for (int yy = y - 1; yy <= y + 1; ++yy) {
                if (yy < 0 || yy >= height) continue;
                const uchar* row = input.ptr<uchar>(yy);
                for (int xx = x - 1; xx <= x + 1; ++xx) {
                    if (xx < 0 || xx >= width) continue;
                    const int ky = yy - (y - 1);
                    const int kx = xx - (x - 1);
                    const float pixel = row[xx] / 255.0f;
                    gx += maskX[ky * 3 + kx] * pixel;
                    gy += maskY[ky * 3 + kx] * pixel;
                }
            }
            float theta = std::atan2(gy, gx) * 180.0f / PI;
            if (theta < 0.0f) theta += 180.0f;
            if (theta >= 180.0f) theta -= 180.0f;
            int bin = static_cast<int>(theta / (180.0f / nbins));
            if (bin >= nbins) bin = nbins - 1;
            magnitude[y * width + x] = std::sqrt(gx * gx + gy * gy);
            binImage[y * width + x] = static_cast<unsigned char>(bin);
        }
    }

    const int nx = (width - block) / interval + 1;
    const int ny = (height - block) / interval + 1;
    HogResult result;
    result.descriptor.reserve(nx * ny * nbins);
    result.finalHistogram.assign(nbins, 0.0f);

    FILE* fp = nullptr;
    fopen_s(&fp, csvPath, "w");
    if (fp) {
        std::fprintf(fp, "Block");
        for (int i = 0; i < nbins; ++i)
            std::fprintf(fp, ",%d-%d", i * 180 / nbins, (i + 1) * 180 / nbins);
        std::fprintf(fp, "\n");
    }

    int blockIndex = 0;
    for (int by = 0; by <= height - block; by += interval) {
        for (int bx = 0; bx <= width - block; bx += interval) {
            vector<float> hist(nbins, 0.0f);
            for (int y = by; y < by + block; ++y)
                for (int x = bx; x < bx + block; ++x)
                    hist[binImage[y * width + x]] += magnitude[y * width + x];

            float sumSquares = eps;
            for (float value : hist) sumSquares += value * value;
            const float inverseNorm = 1.0f / std::sqrt(sumSquares);

            if (fp) std::fprintf(fp, "%d", blockIndex + 1);
            for (int i = 0; i < nbins; ++i) {
                const float normalized = hist[i] * inverseNorm;
                result.descriptor.push_back(normalized);
                result.finalHistogram[i] += normalized;
                if (fp) std::fprintf(fp, ",%f", normalized);
            }
            if (fp) std::fprintf(fp, "\n");
            ++blockIndex;
        }
    }
    if (fp) std::fclose(fp);

    float sumSquares = eps;
    for (float value : result.finalHistogram) sumSquares += value * value;
    const float inverseNorm = 1.0f / std::sqrt(sumSquares);
    for (float& value : result.finalHistogram) value *= inverseNorm;

    std::printf("Blocks = %d x %d = %d\n", nx, ny, nx * ny);
    std::printf("Descriptor dimension = %zu\n", result.descriptor.size());
    return result;
}

static double meanDifference(const vector<float>& a, const vector<float>& b)
{
    if (a.size() != b.size() || a.empty()) return -1.0;
    long double sum = 0.0L;
    for (size_t i = 0; i < a.size(); ++i)
        sum += std::fabs(static_cast<long double>(a[i]) - b[i]);
    return static_cast<double>(sum / a.size());
}

int main()
{
    const char* refPath = "C:/Users/sinbc/source/repos/W3_HOG/W3_HOG/LectureNote_03.bmp";
    const char* c1Path = "C:/Users/sinbc/source/repos/W3_HOG/W3_HOG/compare1.bmp";
    const char* c2Path = "C:/Users/sinbc/source/repos/W3_HOG/W3_HOG/compare2.bmp";
    Mat ref = imread(refPath, IMREAD_GRAYSCALE);
    Mat c1 = imread(c1Path, IMREAD_GRAYSCALE);
    Mat c2 = imread(c2Path, IMREAD_GRAYSCALE);
    if (ref.empty() || c1.empty() || c2.empty()) {
        std::printf("Image load failed!\n");
        return -1;
    }
    if (ref.size() != c1.size() || ref.size() != c2.size()) {
        std::printf("Image size mismatch!\n");
        return -1;
    }

    constexpr int BLOCK = 16, INTERVAL = 8, NBINS = 9;
    constexpr float EPS = 1e-6f;
    HogResult hRef = extractHogNoCell(ref, BLOCK, INTERVAL, NBINS, EPS, "LectureNote_03_no_cell.csv");
    HogResult hC1 = extractHogNoCell(c1, BLOCK, INTERVAL, NBINS, EPS, "compare1_no_cell.csv");
    HogResult hC2 = extractHogNoCell(c2, BLOCK, INTERVAL, NBINS, EPS, "compare2_no_cell.csv");

    const double mean1 = meanDifference(hRef.descriptor, hC1.descriptor);
    const double mean2 = meanDifference(hRef.descriptor, hC2.descriptor);
    std::printf("\n[No Cell: Mean Difference]\n");
    std::printf("Lecture03 vs Compare1 = %.9f\n", mean1);
    std::printf("Lecture03 vs Compare2 = %.9f\n", mean2);
    if (mean1 < mean2) std::printf("Compare1 is more similar.\n");
    else if (mean2 < mean1) std::printf("Compare2 is more similar.\n");
    else std::printf("Same difference.\n");
    return 0;
}
