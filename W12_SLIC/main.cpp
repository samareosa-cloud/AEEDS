#include "main.h"

int main()
{
    Mat image = imread("./images/input.jpg");
    if (image.empty()) {
        std::cerr << "이미지를 읽지 못함: ./images/input.jpg 경로 확인" << std::endl;
        return -1;
    }

    SLICsegmentation(image);
    return 0;
}
