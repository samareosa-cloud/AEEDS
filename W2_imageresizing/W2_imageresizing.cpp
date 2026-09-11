#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // 이미지 읽기
    Mat img = imread("C:/Users/sinbc/test.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    float scale = 2.2;

    int newWidth = (int)(img.cols * scale);
    int newHeight = (int)(img.rows * scale);

    Mat result(newHeight, newWidth, CV_8UC1);

    for (int y = 0; y < newHeight; y++)
    {
        for (int x = 0; x < newWidth; x++)
        {
            float srcX = x * (1.0f / scale);
            float srcY = y * (1.0f / scale);

            int x1 = (int)srcX;
            int y1 = (int)srcY;

            int x2 = x1 + 1;
            int y2 = y1 + 1;

            // out of boundary
            if (x2 >= img.cols)
                x2 = img.cols - 1;

            if (y2 >= img.rows)
                y2 = img.rows - 1;

            float dx = srcX - x1;
            float dy = srcY - y1;

            float p1 = img.at<uchar>(y1, x1);
            float p2 = img.at<uchar>(y1, x2);
            float p3 = img.at<uchar>(y2, x1);
            float p4 = img.at<uchar>(y2, x2);

            float value = (p1 + p2 + p3 + p4) / 4.0f; //4개평균낸거임 다른 수식 있음

            result.at<uchar>(y, x) = (uchar)value;
        }
    }

    imshow("Original", img);
    imshow("Result", result);

    waitKey(0);

    return 0;
}