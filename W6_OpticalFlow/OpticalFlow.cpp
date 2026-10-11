// W6-1. Dense optical flow (Farneback, polynomial expansion) — LN6 Implementation (1/3 ~ 3/3)
// 두 번째 프레임부터 calcOpticalFlowFarneback → 5픽셀 격자마다 flow 를 초록 선으로 표시
// 실행: 이 파일 열고 Cmd+Shift+B  /  종료: 영상 창 클릭 후 아무 키
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>

#define Width  320
#define Height 240

using namespace cv;
using namespace std;

int main()                                  // 슬라이드: void main()
{
    VideoCapture capture(0);
    if (!capture.isOpened()) { printf("Couldn't open the web camera...\n"); return -1; }

    int x, y, key;
    Mat flow, frame, prevFrame, img;
    double total = 0;
    int count = 0;

    while (1) {
        capture >> frame;
        if (frame.empty()) continue;            // 맥: 카메라 켜지는 동안 빈 프레임
        resize(frame, img, Size(Width, Height));
        cvtColor(img, frame, COLOR_BGR2GRAY);   // 슬라이드: CV_BGR2GRAY

        // ★ 첫 프레임은 저장만, 두 번째 프레임부터 flow 계산
        if (prevFrame.empty() == false) {
            // 처리 시간 (슬라이드: QueryPerformance 사용)
            double t1 = getTickCount();

            // ★ calculate optical flow
            // (prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
            calcOpticalFlowFarneback(prevFrame, frame, flow, 0.4, 1, 12, 2, 8, 1.2, 0);

            double diff = (getTickCount() - t1) / getTickFrequency();
            total += diff;
            count++;
            printf("%f sec (avg %f sec)\n", diff, total / count);

            // ★ By y += 5, x += 5 you can specify the grid
            for (y = 0; y < Height; y += 5) {
                for (x = 0; x < Width; x += 5) {
                    const Point2f flowatxy = flow.at<Point2f>(y, x) * 2;    // (dx, dy), 2배 확대 표시
                    line(img, Point(x, y),
                         Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)),
                         Scalar(0, 255, 0));                                 // 초록 선 = 움직임
                    circle(img, Point(x, y), 1, Scalar(0, 0, 0), -1);        // 검은 점 = 격자, -1 = 채움
                }
            }
            frame.copyTo(prevFrame);                // 현재 → 이전
        }
        else frame.copyTo(prevFrame);

        imshow("Optical Flow", img);
        key = waitKey(20);
        if (key >= 0) break;                       // 슬라이드 코드엔 없음 (종료용으로 추가)
    }
    return 0;
}
