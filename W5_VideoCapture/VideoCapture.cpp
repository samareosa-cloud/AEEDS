// W5-1. 웹캠 영상 받기 (VideoCapture) — LN5 Extension to Videos (1/3)
// 모든 비디오 실습(W5 Harris, ORB, W6 Optical flow)의 기본 뼈대
// 실행: 이 파일 열고 Cmd+Shift+B  /  종료: 영상 창 클릭 후 아무 키
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include <cstdio>

using namespace cv;

// ── 맥용 카메라 열기 ────────────────────────────────────────────
// 0번 카메라가 아이폰(연속성 카메라)이나 가상 카메라라서 검은 화면만 나올 수 있다.
// 0~3번을 차례로 열어서 실제로 밝은 영상이 나오는 카메라를 사용한다.
// (윈도우/수업 코드에서는 그냥 VideoCapture capture(0); 이면 된다)
static bool openWorkingCamera(cv::VideoCapture& cap)
{
    for (int idx = 0; idx < 4; idx++) {
        cap.open(idx);
        if (!cap.isOpened()) continue;
        cv::Mat f, g;
        double best = 0;
        for (int i = 0; i < 90; i++) {                     // 카메라 예열: 최대 약 3초
            cap >> f;
            if (f.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(30)); continue; }
            cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
            double m = cv::mean(g)[0];
            if (m > best) best = m;
            if (m > 15) {
                std::printf("camera %d: OK (%d x %d, brightness %.0f)\n", idx, f.cols, f.rows, m);
                return true;
            }
        }
        std::printf("camera %d: black frames only (max brightness %.0f) -> skip\n", idx, best);
        cap.release();
    }
    std::printf("밝은 영상이 나오는 카메라를 못 찾았어요.\n"
                " - 시스템 설정 > 개인정보 보호 및 보안 > 카메라 에서 Visual Studio Code 허용 후 VS Code 재시작\n"
                " - FaceTime / Photo Booth 에서 카메라가 나오는지 확인, 카메라 쓰는 다른 앱(Zoom 등) 종료\n");
    return false;
}
// ────────────────────────────────────────────────────────────────

int main()
{
    VideoCapture capture;              // 객체 capture  (수업 코드: VideoCapture capture(0);)
    openWorkingCamera(capture);        // 맥: 영상이 나오는 카메라 자동 선택
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
