#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int x, y, xx, yy;
int width, height, index, mask_index;
double conv_x, conv_y;
double gx, gy;
double sum_Ix2, sum_Iy2, sum_IxIy;
double det_M, trace_M, response;
double max_R, min_R;
double corner_threshold, edge_threshold;
double* Ix;
double* Iy;
double* R;
Vec3b color;

int mask_x[9] = {
    -1,0,1,
    -1,0,1,
    -1,0,1
};

int mask_y[9] = {
    -1,-1,-1,
     0, 0, 0,
     1, 1, 1
};

void HarrisStep23(Mat input, Mat& result)
{
    width = input.cols;
    height = input.rows;

    Ix = new double[width * height]();
    Iy = new double[width * height]();
    R = new double[width * height]();

    // 기본값을 flat 영역인 회색으로 설정
    result = Mat(height, width, CV_8UC3, Scalar(128, 128, 128));

    // Step 2: 모든 픽셀의 Ix,Iy 계산
    for (y = 1;y < height - 1;y++) {
        for (x = 1;x < width - 1;x++) {
            conv_x = 0.0;
            conv_y = 0.0;
            mask_index = 0;

            for (yy = -1;yy <= 1;yy++) {
                for (xx = -1;xx <= 1;xx++) {
                    color = input.at<Vec3b>(y + yy, x + xx);

                    // 체커보드는 B=G=R이므로 B 채널 사용
                    conv_x += color[0] * mask_x[mask_index];
                    conv_y += color[0] * mask_y[mask_index];

                    mask_index++;
                }
            }

            index = y * width + x;
            Ix[index] = conv_x;
            Iy[index] = conv_y;
        }
    }

    max_R = 0.0;
    min_R = 0.0;

    // Step 3: Ix,Iy로 Harris R 계산
    for (y = 2;y < height - 2;y++) {
        for (x = 2;x < width - 2;x++) {
            sum_Ix2 = 0.0;
            sum_Iy2 = 0.0;
            sum_IxIy = 0.0;

            // 현재 픽셀 주변 3x3 영역
            for (yy = -1;yy <= 1;yy++) {
                for (xx = -1;xx <= 1;xx++) {
                    index = (y + yy) * width + (x + xx);

                    gx = Ix[index];
                    gy = Iy[index];

                    sum_Ix2 += gx * gx;
                    sum_Iy2 += gy * gy;
                    sum_IxIy += gx * gy;
                }
            }

            det_M = sum_Ix2 * sum_Iy2 - sum_IxIy * sum_IxIy;
            trace_M = sum_Ix2 + sum_Iy2;
            response = det_M - 0.04 * trace_M * trace_M;

            index = y * width + x;
            R[index] = response;

            if (response > max_R)
                max_R = response;

            if (response < min_R)
                min_R = response;
        }
    }

    // 양수 방향 최대값의 5% 이상이면 corner
    corner_threshold = max_R * 0.05;

    // 음수 방향 최솟값의 5% 이하이면 edge
    edge_threshold = min_R * 0.05;

    // R값에 따라 corner,flat,edge 구분
    for (y = 0;y < height;y++) {
        for (x = 0;x < width;x++) {
            index = y * width + x;
            response = R[index];

            if (response >= corner_threshold) {
                // Corner: 흰색
                result.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
            }
            else if (response <= edge_threshold) {
                // Edge: 검은색
                result.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
            }
            else {
                // Flat: 회색
                result.at<Vec3b>(y, x) = Vec3b(128, 128, 128);
            }
        }
    }

    delete[] Ix;
    delete[] Iy;
    delete[] R;
}

int main()
{
    Mat input = imread("C:/Users/sinbc/source/repos/W4_Corner/checkerboard.jpg", IMREAD_COLOR);

    if (input.empty()) {
        cout << "Image open error!" << endl;
        return -1;
    }

    Mat result;
    HarrisStep23(input, result);

    imshow("Input", input);
    imshow("Harris Classification", result);

    imwrite("Harris_classification.jpg", result);

    waitKey(0);
    return 0;
}