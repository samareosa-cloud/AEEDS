#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void imageResize(Mat input, float scale)
{
    int x, y;
    int height, width;
    int re_height, re_width;
    int sx, sy;
    float pos_x, pos_y;

    height = input.rows;
    width = input.cols;

    re_height = scale * (float)height;
    re_width = scale * (float)width;

    // 컬러 이미지이므로 3채널
    Mat result(re_height, re_width, CV_8UC3);

    for (y = 0; y < re_height; y++) {
        for (x = 0; x < re_width; x++) {

            // Backward Mapping
            pos_x = (1.0 / scale) * x;
            pos_y = (1.0 / scale) * y;

            sx = (int)pos_x;
            sy = (int)pos_y;

            // 이미지 범위 확인
            if (sx >= 0 && sx < width - 1 &&
                sy >= 0 && sy < height - 1)
            {
                // 주변 4개 컬러 픽셀
                Vec3b p1 = input.at<Vec3b>(sy, sx);
                Vec3b p2 = input.at<Vec3b>(sy, sx + 1);
                Vec3b p3 = input.at<Vec3b>(sy + 1, sx);
                Vec3b p4 = input.at<Vec3b>(sy + 1, sx + 1);

                // B, G, R 각각 평균
                for (int c = 0; c < 3; c++) {
                    result.at<Vec3b>(y, x)[c] =
                        (uchar)(0.25 * (
                            p1[c] +
                            p2[c] +
                            p3[c] +
                            p4[c]
                            ));
                }
            }
        }
    }

    // 결과 화면 출력
    imshow("Original Image", input);
    imshow("Resize Result", result);

    // 결과 저장
    imwrite("C:/Users/sinbc/resizeResult.bmp", result);

    waitKey(0);
}

int main()
{
    // 컬러 이미지로 읽기
    Mat img = imread("C:/Users/sinbc/test.jpg", IMREAD_COLOR);

    if (img.empty()) {
        cout << "Image load failed!" << endl;
        return -1;
    }

    // 0.7배로 resizing
    imageResize(img, 0.7f);

    return 0;
}
