#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main()
{
    Mat input = imread("C:/Users/sinbc/test.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    float deg;
    cout << "Input your degree : ";
    cin >> deg;

    int x, y;
    int height, width;

    float rad = deg * 3.141592f / 180.0f;

    float pos_x, pos_y;
    int sx, sy;

    float R[2][2] =
    {
        { cos(rad),  sin(rad) },
        {-sin(rad),  cos(rad) }
    };

    height = input.rows;
    width = input.cols;

    Mat result = Mat::zeros(height, width, CV_8UC1);

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            pos_x =
                R[0][0] * (x - width / 2.0f) +
                R[0][1] * (y - height / 2.0f);

            pos_y =
                R[1][0] * (x - width / 2.0f) +
                R[1][1] * (y - height / 2.0f);

            pos_x += width / 2.0f;
            pos_y += height / 2.0f;

            sx = (int)pos_x;
            sy = (int)pos_y;

            if (sx >= 0 && sx < width - 1 &&
                sy >= 0 && sy < height - 1)
            {
                float dx = pos_x - sx;
                float dy = pos_y - sy;

                float p1 = input.at<uchar>(sy, sx);
                float p2 = input.at<uchar>(sy, sx + 1);
                float p3 = input.at<uchar>(sy + 1, sx);
                float p4 = input.at<uchar>(sy + 1, sx + 1);

                float value =
                    (1 - dx) * (1 - dy) * p1 +
                    dx * (1 - dy) * p2 +
                    (1 - dx) * dy * p3 +
                    dx * dy * p4;

                result.at<uchar>(y, x) = (uchar)value;
            }
            else
            {
                result.at<uchar>(y, x) = 0;
            }
        }
    }

    imshow("Original", input);
    imshow("Rotation", result);

    waitKey(0);

    return 0;
}
