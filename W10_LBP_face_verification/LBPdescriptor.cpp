// macos version
#include "LBP.h"
#include <math.h>

using namespace cv;
using namespace std;

void LBPimage(const Mat& inputGray, Mat& lbp)
{
    lbp = Mat::zeros(inputGray.size(), CV_8UC1);

    for (int y = 1; y < inputGray.rows - 1; y++) {
        for (int x = 1; x < inputGray.cols - 1; x++) {
            int val = 0;
            if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y - 1, x))     val += 1;
            if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y - 1, x + 1)) val += 2;
            if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y, x + 1))     val += 4;
            if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y + 1, x + 1)) val += 8;
            if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y + 1, x))     val += 16;
            if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y + 1, x - 1)) val += 32;
            if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y, x - 1))     val += 64;
            if (inputGray.at<uchar>(y, x) < inputGray.at<uchar>(y - 1, x - 1)) val += 128;
            lbp.at<uchar>(y, x) = val;
        }
    }
}

// 얼굴 잘라서 128x128 -> LBP -> 32x32 block 49개 histogram (block마다 L2 정규화) 이어붙이기
void LBPdescriptor(const Mat& frame, vector<float>& hist, int x, int y, int w, int h)
{
    Mat gray, face, lbp;
    cvtColor(frame(Rect(x, y, w, h)), gray, COLOR_BGR2GRAY);
    resize(gray, face, Size(FACE_SIZE, FACE_SIZE), 0, 0, INTER_AREA);
    LBPimage(face, lbp);

    hist.assign(DIM, 0.0f);
    int k = 0;
    for (int by = 0; by <= FACE_SIZE - BLOCK; by += STRIDE) {
        for (int bx = 0; bx <= FACE_SIZE - BLOCK; bx += STRIDE) {
            float* h = &hist[k * NBIN];

            for (int yy = by; yy < by + BLOCK; yy++)
                for (int xx = bx; xx < bx + BLOCK; xx++)
                    h[lbp.at<uchar>(yy, xx)]++;

            // h_k = N_k / sqrt(sum N_q^2 + delta)
            float sum = 1e-6f;
            for (int i = 0; i < NBIN; i++) sum += h[i] * h[i];
            sum = sqrt(sum);
            for (int i = 0; i < NBIN; i++) h[i] /= sum;
            k++;
        }
    }
}