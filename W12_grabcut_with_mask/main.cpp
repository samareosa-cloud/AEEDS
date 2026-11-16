#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;

void GraphCut_mask(Mat input, Mat mask_img)
{
    int x, y;
    int height = input.rows;
    int width  = input.cols;

    Rect rectangle(0, 0, width - 1, height - 1);

    // 1) grabCut용 라벨 mask : 처음엔 전부 "아마 배경"
    Mat mask = Mat::ones(height, width, CV_8UC1) * GC_PR_BGD;

    // 2) 칠한 색으로 확실한 라벨 지정 (BGR 순서 : [0]=B, [1]=G, [2]=R)
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            Vec3b c = mask_img.at<Vec3b>(y, x);
            if (c[2] > 150 && c[1] < 100 && c[0] < 100)          // 빨강 -> 확실한 물체
                mask.at<uchar>(y, x) = GC_FGD;
            else if (c[0] > 150 && c[2] < 100 && c[1] < 100)     // 파랑 -> 확실한 배경
                mask.at<uchar>(y, x) = GC_BGD;
        }
    }

    // 3) 결과 정리
    Mat final_result = Mat::zeros(height, width, CV_8UC3);
    Mat mask_result  = Mat::zeros(height, width, CV_8UC1);
    Mat bg, fg;

    final_result = input.clone();               // 원본을 복사해 두고 배경만 흰색으로 덮음

    grabCut(input, mask, rectangle, bg, fg, 8, GC_INIT_WITH_MASK);   // 8번 반복

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            if (mask.at<uchar>(y, x) == GC_FGD || mask.at<uchar>(y, x) == GC_PR_FGD) {
                mask_result.at<uchar>(y, x) = 255;            // 물체 -> 흰색(255)
            }
            else {
                final_result.at<Vec3b>(y, x)[0] = 255;        // 배경 -> 흰색으로 지움
                final_result.at<Vec3b>(y, x)[1] = 255;
                final_result.at<Vec3b>(y, x)[2] = 255;
            }
        }
    }

    imshow("Seeds", mask_img);
    imshow("mask_result", mask_result);
    imshow("final_result", final_result);
    moveWindow("Seeds", 0, 0);                      // Mac에서 창이 겹치지 않게
    moveWindow("mask_result", width + 10, 0);
    moveWindow("final_result", 2 * (width + 10), 0);
    waitKey(0);
}

int main()
{
    Mat input    = imread("input.jpg");
    Mat mask_img = imread("mask.bmp");
    if (input.empty() || mask_img.empty()) {
        std::cerr << "input.jpg 또는 mask.bmp를 읽지 못함" << std::endl;
        return -1;
    }
    GraphCut_mask(input, mask_img);
    return 0;
}
