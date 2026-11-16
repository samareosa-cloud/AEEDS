#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;

int main()
{
    Mat image = imread("input.jpg");          
    if (image.empty()) {
        std::cerr << "이미지를 못 읽음: 경로 확인!" << std::endl;
        return -1;
    }

    Rect rectangle(10, 100, 380, 280);
    rectangle &= Rect(0, 0, image.cols, image.rows);   

    Mat result;   
    Mat bg, fg;  

    grabCut(image, result, rectangle, bg, fg, 5, GC_INIT_WITH_RECT);
    compare(result, GC_PR_FGD, result, CMP_EQ);

    Mat foreground(image.size(), CV_8UC3, Scalar(255, 255, 255)); // 흰 배경
    image.copyTo(foreground, result);         

    
    Mat show = image.clone();
    cv::rectangle(show, rectangle, Scalar(0, 0, 255), 2);

    imshow("Input", show);
    imshow("Foreground", foreground);
    waitKey(0);
    return 0;
}
