#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

static void MakeCornerMap(const Mat& input, Mat& cornerMap, double thresholdRatio)
{
    int x, y, xx, yy;
    int width, height, index, neighborIndex, maskIndex;
    double convX, convY;
    double gx, gy, pixel;
    double sumIx2, sumIy2, sumIxIy;
    double determinant, trace, response;
    double maxR, threshold;

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

    width = input.cols;
    height = input.rows;

    vector<double> Ix(width * height, 0.0);
    vector<double> Iy(width * height, 0.0);
    vector<double> R(width * height, 0.0);

    for (y = 1; y < height - 1; y++) {
        for (x = 1; x < width - 1; x++) {
            convX = 0.0;
            convY = 0.0;
            maskIndex = 0;

            for (yy = -1; yy <= 1; yy++) {
                for (xx = -1; xx <= 1; xx++) {
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

    for (y = 2; y < height - 2; y++) {
        for (x = 2; x < width - 2; x++) {
            sumIx2 = 0.0;
            sumIy2 = 0.0;
            sumIxIy = 0.0;

            for (yy = -1; yy <= 1; yy++) {
                for (xx = -1; xx <= 1; xx++) {
                    neighborIndex = (y + yy) * width + (x + xx);

                    gx = Ix[neighborIndex];
                    gy = Iy[neighborIndex];

                    sumIx2 += gx * gx;
                    sumIy2 += gy * gy;
                    sumIxIy += gx * gy;
                }
            }

            determinant = sumIx2 * sumIy2 - sumIxIy * sumIxIy;
            trace = sumIx2 + sumIy2;
            response = determinant - 0.04 * trace * trace;

            index = y * width + x;
            R[index] = response;

            if (response > maxR)
                maxR = response;
        }
    }

    cornerMap = Mat::zeros(height, width, CV_8UC1);
    threshold = maxR * thresholdRatio;

    for (y = 2; y < height - 2; y++) {
        for (x = 2; x < width - 2; x++) {
            index = y * width + x;

            if (R[index] > threshold)
                cornerMap.at<uchar>(y, x) = 1;
        }
    }
}

static void DrawCornerCircle(Mat& result, const Mat& cornerMap)
{
    int x, y;
    int width = cornerMap.cols;
    int height = cornerMap.rows;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            if (cornerMap.at<uchar>(y, x) == 1) {
                circle(
                    result,
                    Point(x, y),
                    2,
                    Scalar(0, 0, 255),
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
    Mat frame;
    Mat gray;
    Mat cornerMap;
    Mat result;

    VideoCapture capture(
        "C:/Users/sinbc/source/repos/W5_videocapture/video.mp4"
    );

    if (!capture.isOpened()) {
        cout << "Video open error!" << endl;
        return -1;
    }

    while (true) {
        capture >> frame;

        if (frame.empty())
            break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        MakeCornerMap(
            gray,
            cornerMap,
            0.05
        );

        result = frame.clone();

        DrawCornerCircle(
            result,
            cornerMap
        );

        imshow(
            "Corner Detection",
            result
        );

        if (waitKey(30) == 27)
            break;
    }

    capture.release();
    destroyAllWindows();

    return 0;
}