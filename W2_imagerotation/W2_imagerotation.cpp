#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

void imageRotation(Mat input, float degree)
{
    int x, y;
    int height, width;
    int sx, sy;

    float pos_x, pos_y;
    float rad;

    height = input.rows;
    width = input.cols;

    rad = degree * 3.141592f / 180.0f;

    Mat result = Mat::zeros(height, width, CV_8UC1);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {

            // Backward Mapping
            // 이미지 중심을 (0,0)으로 이동한 뒤 회전
            pos_x =
                cos(rad) * (x - width / 2.0f) +
                sin(rad) * (y - height / 2.0f);

            pos_y =
                -sin(rad) * (x - width / 2.0f) +
                cos(rad) * (y - height / 2.0f);

            // 다시 원래 영상 좌표계로 이동
            pos_x += width / 2.0f;
            pos_y += height / 2.0f;

            sx = (int)(pos_x);
            sy = (int)(pos_y);

            // 단순 평균을 이용한 보간
            if (sx >= 0 && sx < width - 1 &&
                sy >= 0 && sy < height - 1)
            {
                result.at<uchar>(y, x) =
                    0.25 * (
                        input.at<uchar>(sy, sx) +
                        input.at<uchar>(sy, sx + 1) +
                        input.at<uchar>(sy + 1, sx) +
                        input.at<uchar>(sy + 1, sx + 1)
                        );
            }
        }
    }

    imshow("Original Image", input);
    imshow("Rotation Result", result);

    imwrite("C:/Users/sinbc/rotationResult.bmp", result);

    waitKey(0);
}

int main()
{
    Mat input = imread(
        "C:/Users/sinbc/test.jpg",
        CV_LOAD_IMAGE_GRAYSCALE
    );

    if (input.empty()) {
        cout << "Image load failed!" << endl;
        return -1;
    }

    float degree;

    cout << "Input your degree : ";
    cin >> degree;

    imageRotation(input, degree);

    return 0;
}
