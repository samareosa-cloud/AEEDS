#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

static void MakeCornerMap(
    const Mat& input,
    Mat& cornerMap,
    double thresholdRatio)
{
    int x, y, xx, yy;
    int width, height, index, neighborIndex, maskIndex;
    double convX, convY;
    double gx, gy, pixel;
    double sumIx2, sumIy2, sumIxIy;
    double determinant, trace, response;
    double maxR, threshold;

    int maskX[9] = {
        -1,0,1,
        -1,0,1,
        -1,0,1
    };

    int maskY[9] = {
        -1,-1,-1,
         0, 0, 0,
         1, 1, 1
    };

    width = input.cols;
    height = input.rows;

    vector<double> Ix(width * height, 0.0);
    vector<double> Iy(width * height, 0.0);
    vector<double> R(width * height, 0.0);

    // Step 2: Ix, Iy 계산
    for (y = 1;y < height - 1;y++) {
        for (x = 1;x < width - 1;x++) {
            convX = 0.0;
            convY = 0.0;
            maskIndex = 0;

            for (yy = -1;yy <= 1;yy++) {
                for (xx = -1;xx <= 1;xx++) {
                    pixel = input.at<uchar>(y + yy, x + xx);

                    convX += pixel * maskX[maskIndex];
                    convY += pixel * maskY[maskIndex];

                    maskIndex++;
                }
            }

            index = y * width + x;

            Ix[index] = convX;
            Iy[index] = convY;
        }
    }

    maxR = 0.0;

    // Step 3: Harris R 계산
    for (y = 2;y < height - 2;y++) {
        for (x = 2;x < width - 2;x++) {
            sumIx2 = 0.0;
            sumIy2 = 0.0;
            sumIxIy = 0.0;

            // 현재 픽셀 주변 3x3 영역
            for (yy = -1;yy <= 1;yy++) {
                for (xx = -1;xx <= 1;xx++) {
                    neighborIndex = (y + yy) * width + (x + xx);

                    gx = Ix[neighborIndex];
                    gy = Iy[neighborIndex];

                    sumIx2 += gx * gx;
                    sumIy2 += gy * gy;
                    sumIxIy += gx * gy;
                }
            }

            determinant =
                sumIx2 * sumIy2 - sumIxIy * sumIxIy;

            trace = sumIx2 + sumIy2;

            response =
                determinant - 0.04 * trace * trace;

            index = y * width + x;
            R[index] = response;

            if (response > maxR) {
                maxR = response;
            }
        }
    }

    // cornerMap은 0 또는 1
    cornerMap = Mat::zeros(height, width, CV_8UC1);

    // 최대 R의 5%를 임계값으로 사용
    threshold = maxR * thresholdRatio;

    for (y = 2;y < height - 2;y++) {
        for (x = 2;x < width - 2;x++) {
            index = y * width + x;

            if (R[index] > threshold) {
                cornerMap.at<uchar>(y, x) = 1;
            }
            else {
                cornerMap.at<uchar>(y, x) = 0;
            }
        }
    }

    cout << "max R = " << maxR << endl;
    cout << "threshold = " << threshold << endl;
}

static void DrawCornerCircle(
    Mat& result,
    const Mat& cornerMap)
{
    int x, y;
    int width, height;
    int radius;
    Point center;
    Scalar color;

    width = cornerMap.cols;
    height = cornerMap.rows;

    radius = 2;

    // OpenCV는 BGR 순서: 빨간색
    color = Scalar(0, 0, 255);

    for (y = 0;y < height;y++) {
        for (x = 0;x < width;x++) {
            if (cornerMap.at<uchar>(y, x) == 1) {
                center.x = x;
                center.y = y;

                circle(
                    result,
                    center,
                    radius,
                    color,
                    1,
                    8,
                    0
                );
            }
        }
    }
}

int main()
{
    Mat refGray;
    Mat tarGray;
    Mat refColor;
    Mat tarColor;

    Mat refCornerMap;
    Mat tarCornerMap;

    double thresholdRatio;

    refGray = imread(
        "C:/Users/sinbc/source/repos/W4_Corner/ref.bmp",
        IMREAD_GRAYSCALE
    );

    tarGray = imread(
        "C:/Users/sinbc/source/repos/W4_Corner/tar.bmp",
        IMREAD_GRAYSCALE
    );

    refColor = imread(
        "C:/Users/sinbc/source/repos/W4_Corner/ref.bmp",
        IMREAD_COLOR
    );

    tarColor = imread(
        "C:/Users/sinbc/source/repos/W4_Corner/tar.bmp",
        IMREAD_COLOR
    );

    if (refGray.empty() ||
        tarGray.empty() ||
        refColor.empty() ||
        tarColor.empty()) {
        cout << "Image open error!" << endl;
        return -1;
    }

    // maxR의 5%
    thresholdRatio = 0.05;

    cout << "[REF IMAGE]" << endl;

    MakeCornerMap(
        refGray,
        refCornerMap,
        thresholdRatio
    );

    cout << "[TARGET IMAGE]" << endl;

    MakeCornerMap(
        tarGray,
        tarCornerMap,
        thresholdRatio
    );

    DrawCornerCircle(
        refColor,
        refCornerMap
    );

    DrawCornerCircle(
        tarColor,
        tarCornerMap
    );

    imshow("Reference Corners", refColor);
    imshow("Target Corners", tarColor);

    imwrite("ref_corners.png", refColor);
    imwrite("tar_corners.png", tarColor);

    waitKey(0);
    return 0;
}