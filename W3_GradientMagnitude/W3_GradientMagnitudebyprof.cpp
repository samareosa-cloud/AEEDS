#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace cv;
using namespace std;


// 교수님 코드
void EdgeDetection(Mat input)
{
    int x, y, xx, yy;
    int height, width;
    int b_size = 3;
    int win_size = 3;
    int len = 9;
    const int size = 9;
    float conv_x, conv_y;
    float min, max;
    float dir;

    int mask_x[size] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    int mask_y[size] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

    height = input.rows;
    width = input.cols;

    min = 1000000;
    max = -1;

    float* val = (float*)calloc(height * width, sizeof(float));
    int* bin = (int*)calloc(len, sizeof(int));
    float* pdf = (float*)calloc(len, sizeof(float));

    Mat MagImage(height, width, CV_8UC1);

    // Gradient computation
    for (y = 0; y < height; y++) {

        for (x = 0; x < width; x++) {

            conv_x = 0;
            conv_y = 0;

            for (yy = y - b_size / 2;
                yy <= y + b_size / 2;
                yy++) {

                for (xx = x - b_size / 2;
                    xx <= x + b_size / 2;
                    xx++) {

                    if (yy >= 0 && yy < height
                        && xx >= 0 && xx < width) {

                        conv_x +=
                            mask_x[
                                (yy - (y - 1)) * b_size
                                    + (xx - (x - 1))
                            ]
                            * input.at<uchar>(yy, xx)
                            / 255.0;

                        conv_y +=
                            mask_y[
                                (yy - (y - 1)) * b_size
                                    + (xx - (x - 1))
                            ]
                            * input.at<uchar>(yy, xx)
                            / 255.0;
                    }
                }
            }

            // magnitude
            val[y * width + x]
                = sqrt(conv_x * conv_x + conv_y * conv_y);

            if (min > val[y * width + x])
                min = val[y * width + x];

            if (max < val[y * width + x])
                max = val[y * width + x];
        }
    }

    // visualization
    for (y = 0; y < height; y++) {

        for (x = 0; x < width; x++) {

            MagImage.at<uchar>(y, x)
                = 255
                - 255 * (val[y * width + x] - min)
                / (max - min);
        }
    }

    imwrite("MagImage.bmp", MagImage);
}


// 실행 부분
int main()
{
    Mat input;
    Mat magnitudeImage;

    // 흑백영상으로 불러오기
    input = imread("C:/Users/sinbc/source/repos/W3_HOG/W3_HOG/LectureNote_03.bmp", IMREAD_GRAYSCALE);

    // 이미지 로드 확인
    if (input.empty()) {
        cout << "Image load failed!" << endl;
        cout << "LectureNote_03.bmp 파일 위치를 확인하세요." << endl;
        return -1;
    }

    // 원본 흑백영상 출력
    imshow("Input Image", input);

    // 교수님 함수 실행
    EdgeDetection(input);

    // 함수가 저장한 결과영상 다시 불러오기
    magnitudeImage = imread("MagImage.bmp", IMREAD_GRAYSCALE);

    // 결과영상 로드 확인
    if (magnitudeImage.empty()) {
        cout << "MagImage.bmp load failed!" << endl;
        return -1;
    }

    // Gradient magnitude 영상 출력
    imshow("Gradient Magnitude", magnitudeImage);

    cout << "Gradient magnitude calculation completed." << endl;
    cout << "MagImage.bmp file saved." << endl;

    // 키보드 입력까지 창 유지
    waitKey(0);

    return 0;
}