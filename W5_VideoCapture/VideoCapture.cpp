// W5-1. 웹캠 영상 받기 (VideoCapture) — LN5 Extension to Videos (1/3)
// 모든 비디오 실습(W5 Harris, ORB, W6 Optical flow)의 기본 뼈대
// 실행: 이 파일 열고 Cmd+Shift+B  /  종료: 영상 창 클릭 후 아무 키
#include <opencv2/opencv.hpp>
#include <cstdio>

using namespace cv;

int main()
{
    const int CAM_INDEX = 0;           // 아이폰(연속성 카메라)이 잡히면 1 로 바꾸기

    VideoCapture capture(CAM_INDEX);   // 객체 capture
    Mat frame;

    if (!capture.isOpened()) {
        printf("Couldn't open the web camera...\n");
        printf("맥: 시스템 설정 > 개인정보 보호 및 보안 > 카메라 에서 VS Code 허용 후 재시작\n");
        return -1;
    }

    int frameNum = 0;
    while (true) {
        capture >> frame;              // 프레임 한 장 획득
        if (frame.empty()) continue;   // 맥: 카메라 켜지는 동안 빈 프레임이 올 수 있음

        /* Our algorithm (여기에 처리 함수를 넣는다) */

        imshow("Video", frame);
        if (waitKey(30) >= 0) break;   // 30ms 대기, 아무 키나 누르면 종료

        frameNum++;
    }
    printf("captured %d frames (%d x %d)\n", frameNum, frame.cols, frame.rows);
    return 0;
}
