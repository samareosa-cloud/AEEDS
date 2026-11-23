#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Mat input = imread("/Users/samareosa/Desktop/OpenCV_Project/AEEDS/W13/input.jpg");
    if (input.empty()) { cout << "이미지 못 읽음" << endl; return -1; }
    resize(input, input, Size(800, 600));

    putText(input, "Hello Inpainting", Point(60, 340),
            FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 255), 3);

    Mat textBox = Mat::zeros(input.size(), CV_8UC1);

    int min_x = 60,  max_x = 460;
    int min_y = 300, max_y = 350;
    int margin = 5;              

    for (int y = min_y - margin; y < max_y + margin; y++) {
        for (int x = min_x - margin; x < max_x + margin; x++) {
            if (x < 0 || y < 0 || x >= textBox.cols || y >= textBox.rows) continue;  // 범위 체크
            textBox.at<uchar>(y, x) = 255;
        }
    }
    
    // inpaint
    Mat result;
    inpaint(input, textBox, result, 3, INPAINT_TELEA);

    imshow("input", input);
    imshow("mask", textBox);
    imshow("result", result);
    waitKey(0);
    return 0;
}