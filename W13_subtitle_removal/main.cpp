#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat input = imread(argc > 1 ? argv[1] : "//Users/samareosa/Desktop/OpenCV_Project/AEEDS/W13_cornerdetector/input.jpg");
    if (input.empty()) { cout << "이미지 못 읽음" << endl; return -1; }

    Mat gray;
    cvtColor(input, gray, COLOR_BGR2GRAY);

    Mat gray32, harris;
    gray.convertTo(gray32, CV_32F);
    cornerHarris(gray32, harris, 2, 3, 0.04);
    double maxVal;
    minMaxLoc(harris, nullptr, &maxVal);

    Mat white = (gray > 200);
    dilate(white, white, Mat(), Point(-1, -1), 2);
    Mat cornerAll = input.clone();   // 사진 전체의 코너
    Mat cornerSub = input.clone();   // 자막으로 판단된 코너

    // 전체 코너 표시 (빨간 점)
    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            if (harris.at<float>(y, x) > 0.01 * maxVal)
                circle(cornerAll, Point(x, y), 2, Scalar(0, 0, 255), -1);
        }
    }

    int min_x = input.cols, max_x = -1;
    int min_y = input.rows, max_y = -1;
    int startY = input.rows * 2 / 3;

    for (int y = input.rows - 1; y >= startY; y--) {
        for (int x = 0; x < input.cols; x++) {
            bool isCorner = harris.at<float>(y, x) > 0.01 * maxVal;
            bool isWhite  = white.at<uchar>(y, x) > 0;
            if (isCorner && isWhite) {
                min_x = min(min_x, x);  max_x = max(max_x, x);
                min_y = min(min_y, y);  max_y = max(max_y, y);
                circle(cornerSub, Point(x, y), 2, Scalar(0, 255, 0), -1);   // 자막 코너 (초록 점)
            }
        }
    }
    if (max_x < 0) { cout << "자막 코너 없음" << endl; return -1; }
    cout << "subtitle box: x " << min_x << "~" << max_x
         << ", y " << min_y << "~" << max_y << endl;


    line(cornerSub, Point(0, startY), Point(input.cols, startY), Scalar(255, 0, 0), 2);         // 파란 선
    rectangle(cornerSub, Point(min_x, min_y), Point(max_x, max_y), Scalar(0, 255, 255), 2);    // 노란 박스

    Mat textBox = Mat::zeros(input.size(), CV_8UC1);
    int margin = 5;
    for (int y = min_y - margin; y < max_y + margin; y++) {
        for (int x = min_x - margin; x < max_x + margin; x++) {
            if (x < 0 || y < 0 || x >= textBox.cols || y >= textBox.rows) continue;
            textBox.at<uchar>(y, x) = 255;
        }
    }
    
    Mat result;
    inpaint(input, textBox, result, 3, INPAINT_TELEA);

    imwrite("corner_all.png", cornerAll);
    imwrite("corner_subtitle.png", cornerSub);
    imwrite("mask.png", textBox);
    imwrite("result.png", result);

    imshow("input", input);
    imshow("1. all corners", cornerAll);
    imshow("2. subtitle corners + box", cornerSub);
    imshow("3. mask", textBox);
    imshow("4. result", result);
    waitKey(0);
    return 0;
}