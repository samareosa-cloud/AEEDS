#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace cv;
using namespace std;


// 입력영상의 gradient magnitude를 계산하는 함수
void EdgeDetection(Mat input)
{
    int x, y, xx, yy; // x, y -> gradient를 계산한 중심 픽셀, xx, yy -> 중심 픽셀 주변의 3x3 픽셀
    int height, width; // 입력영상의 세로와 가로 크기
    int b_size = 3; // Gradeint 계산에 사용하는 prewitt mask 크기: 3x3
    int win_size = 3; // 윈도우 크기
    int len = 9; // 방향 histogram의 bin개수
    const int size = 9; // 3x3 mask의 전체 원소 개수
    float conv_x, conv_y; //x방향, y방향 gradient
    float min, max; // 전체 magnitude 중 최솟값과 최댓값
    float dir; // gradient 방향 저장용 변수

    int mask_x[size] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    int mask_y[size] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

    height = input.rows; // 입력영상의 행 개수, 세로길이
    width = input.cols; // 입력영상의 열 개수, 가로길이

    min = 1000000;
    max = -1;

    float* val = (float*)calloc(height * width, sizeof(float)); // 모든 픽셀의 magnitude를 저장할 배열 생성
    int* bin = (int*)calloc(len, sizeof(int)); // 크기 9인 정수형 배열 생성
    float* pdf = (float*)calloc(len, sizeof(float)); // 크기 9인 실수형 배열 생성

    Mat MagImage(height, width, CV_8UC1); // magnitude를 시각화할 흑백 결과영상 생성

    // Gradient computation
    for (y = 0; y < height; y++) { // 모든 행을 위->아래

        for (x = 0; x < width; x++) { // 현재 행의 모든 열을 왼->오

            conv_x = 0; // 새로운 중심 픽셀의 x방향 gradient를 0으로 초기화
            conv_y = 0;
            
            // 중심 픽셀 기준 yy는 y-1 ~ y+1 까지 이동
            for (yy = y - b_size / 2;
                yy <= y + b_size / 2;
                yy++) {

                //중심 픽셀 기준 xx는 x-1 ~ x+1 까지 이동
                for (xx = x - b_size / 2;
                    xx <= x + b_size / 2;
                    xx++) {

                    if (yy >= 0 && yy < height
                        && xx >= 0 && xx < width) {

                        // 현재 주변 픽셀 값에 mask_x 값 곱해서 누적
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

            // min , max 갱신
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

    // 함수 실행
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
