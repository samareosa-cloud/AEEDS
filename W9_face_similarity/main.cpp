// macos version
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;

#define PI 3.14159265f
#define WIN 36
#define BLOCK 12
#define STRIDE 6
#define NBIN 9
#define DIM (((WIN - BLOCK) / STRIDE + 1) * ((WIN - BLOCK) / STRIDE + 1) * NBIN)   // 225

float cutoff = 0.87f;
int nmsR = 16;
int boxSize = 32;

void computeHOG(const Mat& img, float* desc)
{
    int maskX[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    int maskY[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
    float mag[WIN * WIN];
    int bin[WIN * WIN];

    for (int y = 0; y < WIN; y++) {
        for (int x = 0; x < WIN; x++) {
            float gx = 0, gy = 0;
            for (int yy = y - 1; yy <= y + 1; yy++) {
                for (int xx = x - 1; xx <= x + 1; xx++) {
                    if (yy < 0 || yy >= WIN || xx < 0 || xx >= WIN) continue;
                    float p = img.at<uchar>(yy, xx) / 255.0f;
                    int k = (yy - y + 1) * 3 + (xx - x + 1);
                    gx += maskX[k] * p;
                    gy += maskY[k] * p;
                }
            }
            float theta = atan2(gy, gx) * 180 / PI;
            if (theta < 0) theta += 180;
            if (theta >= 180) theta -= 180;
            int b = (int)(theta / 20);
            if (b >= NBIN) b = NBIN - 1;

            mag[y * WIN + x] = sqrt(gx * gx + gy * gy);
            bin[y * WIN + x] = b;
        }
    }

    int k = 0;
    for (int by = 0; by <= WIN - BLOCK; by += STRIDE) {
        for (int bx = 0; bx <= WIN - BLOCK; bx += STRIDE) {
            float* h = &desc[k * NBIN];
            for (int i = 0; i < NBIN; i++) h[i] = 0;

            for (int y = by; y < by + BLOCK; y++)
                for (int x = bx; x < bx + BLOCK; x++)
                    h[bin[y * WIN + x]] += mag[y * WIN + x];

            // L2 normalization
            float sum = 1e-6f;
            for (int i = 0; i < NBIN; i++) sum += h[i] * h[i];
            sum = sqrt(sum);
            for (int i = 0; i < NBIN; i++) h[i] /= sum;
            k++;
        }
    }
}

float computeSimilarity(const float* ref, const float* tar)
{
    float nomi = 0, refMag = 0, tarMag = 0;
    for (int k = 0; k < DIM; k++) {
        nomi += ref[k] * tar[k];
        refMag += ref[k] * ref[k];
        tarMag += tar[k] * tar[k];
    }
    return nomi / (sqrt(refMag) * sqrt(tarMag) + 1e-6f);
}

void refHOG(const Mat& ref, float* refHOG_descriptor)
{
    computeHOG(ref, refHOG_descriptor);
}

void drawBox(Mat& img, int x0, int y0, int size)
{
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < size; i++) {
            img.at<Vec3b>(y0 + t, x0 + i) = Vec3b(255, 255, 255);
            img.at<Vec3b>(y0 + size - 1 - t, x0 + i) = Vec3b(255, 255, 255);
            img.at<Vec3b>(y0 + i, x0 + t) = Vec3b(255, 255, 255);
            img.at<Vec3b>(y0 + i, x0 + size - 1 - t) = Vec3b(255, 255, 255);
        }
    }
}

void searchFaces(const Mat& tar, float* refHOG_descriptor, Mat& simMap, Mat& result)
{
    int w = tar.cols, h = tar.rows;
    int nx = w - WIN + 1, ny = h - WIN + 1;
    vector<float> score(nx * ny);
    float tarHOG[DIM];
    Mat B(WIN, WIN, CV_8UC1);
    float maxScore = -1;

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            // 1. 현재 위치에서 36x36 블록 B
            for (int yy = 0; yy < WIN; yy++)
                for (int xx = 0; xx < WIN; xx++)
                    B.at<uchar>(yy, xx) = tar.at<uchar>(y + yy, x + xx);

            // 2. B 의 HOG
            computeHOG(B, tarHOG);

            // 3. ref HOG 와 cosine similarity
            float s = computeSimilarity(refHOG_descriptor, tarHOG);

            score[y * nx + x] = s;
            if (s > maxScore) maxScore = s;
        }
    }

    // similarity map (0~255) : 평균 이하는 0, 최고점은 255
    float meanScore = 0;
    for (int i = 0; i < nx * ny; i++) meanScore += score[i];
    meanScore /= nx * ny;

    simMap = Mat::zeros(h, w, CV_8UC1);
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            float v = (score[y * nx + x] - meanScore) / (maxScore - meanScore);
            if (v < 0) v = 0;
            simMap.at<uchar>(y + WIN / 2, x + WIN / 2) = (uchar)(v * 255);
        }
    }

    result = Mat(h, w, CV_8UC3);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uchar g = tar.at<uchar>(y, x);
            result.at<Vec3b>(y, x) = Vec3b(g, g, g);
        }

    // cut-off + NMS (주변에서 가장 큰 점만)
    int count = 0;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            float o = score[y * nx + x];
            if (o < cutoff) continue;

            bool isMax = true;
            for (int yy = max(0, y - nmsR); yy <= min(ny - 1, y + nmsR) && isMax; yy++)
                for (int xx = max(0, x - nmsR); xx <= min(nx - 1, x + nmsR); xx++)
                    if (score[yy * nx + xx] > o) { isMax = false; break; }
            if (!isMax) continue;

            int cx = x + WIN / 2, cy = y + WIN / 2;
            drawBox(result, cx - boxSize / 2, cy - boxSize / 2, boxSize);
            printf("face %d : (%d, %d)  %.3f\n", ++count, cx, cy, o);
        }
    }
    printf("total faces : %d\n", count);
}

int main()
{
    Mat ref = imread("face_ref.bmp", 0);
    Mat tar = imread("face_tar.bmp", 0);

    float refHOG_descriptor[300];
    Mat simMap, result;

    refHOG(ref, refHOG_descriptor);
    searchFaces(tar, refHOG_descriptor, simMap, result);

    imwrite("similarity_map.bmp", simMap);
    imwrite("detection_result.bmp", result);
    imshow("similarity map", simMap);
    imshow("detection result", result);
    waitKey(0);
    return 0;
}
