// W5-2. cornerDetector.cpp — W4 Harris 를 프레임 단위 함수로
#include "cornerDetector.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using std::vector;

int HarrisCornerDetector(const Mat& frame, Mat& cornerMap)
{
    const float k = 0.04f;
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);       // 비디오 실습에서는 OpenCV 변환 사용 가능
    int h = gray.rows, w = gray.cols;

    // 3x3 Gaussian 근사 {1,2,1; 2,4,2; 1,2,1} / 16
    vector<float> I(h * w, 0);
    int gk[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            float s = 0;
            int n = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++, n++) s += gk[n] * gray.at<uchar>(y + dy, x + dx);
            I[y * w + x] = s / 16.0f;
        }

    // Ix, Iy (Prewitt)
    vector<float> Ix(h * w, 0), Iy(h * w, 0);
    for (int y = 2; y < h - 2; y++)
        for (int x = 2; x < w - 2; x++) {
            Ix[y * w + x] = (I[(y - 1) * w + x + 1] + I[y * w + x + 1] + I[(y + 1) * w + x + 1])
                          - (I[(y - 1) * w + x - 1] + I[y * w + x - 1] + I[(y + 1) * w + x - 1]);
            Iy[y * w + x] = (I[(y + 1) * w + x - 1] + I[(y + 1) * w + x] + I[(y + 1) * w + x + 1])
                          - (I[(y - 1) * w + x - 1] + I[(y - 1) * w + x] + I[(y - 1) * w + x + 1]);
        }

    // ★ R = det(M) - k·tr²(M)
    vector<float> R(h * w, 0);
    float maxR = 0;
    for (int y = 3; y < h - 3; y++)
        for (int x = 3; x < w - 3; x++) {
            float Sxx = 0, Syy = 0, Sxy = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    float ix = Ix[(y + dy) * w + x + dx], iy = Iy[(y + dy) * w + x + dx];
                    Sxx += ix * ix;  Syy += iy * iy;  Sxy += ix * iy;
                }
            R[y * w + x] = (Sxx * Syy - Sxy * Sxy) - k * (Sxx + Syy) * (Sxx + Syy);
            if (R[y * w + x] > maxR) maxR = R[y * w + x];
        }

    // ★ threshold + 3x3 NMS → 원 표시
    cornerMap = frame.clone();
    int count = 0;
    for (int y = 3; y < h - 3; y++)
        for (int x = 3; x < w - 3; x++) {
            float v = R[y * w + x];
            if (v <= maxR * 0.01f) continue;
            bool isMax = true;
            for (int dy = -1; dy <= 1 && isMax; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    if (R[(y + dy) * w + x + dx] > v) { isMax = false; break; }
            if (isMax) {
                circle(cornerMap, Point(x, y), 3, Scalar(0, 0, 255), 2, 8, 0);
                count++;
            }
        }
    return count;
}
