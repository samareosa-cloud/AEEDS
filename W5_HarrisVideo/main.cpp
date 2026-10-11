// W5-2. 웹캠 프레임마다 Harris corner + 처리 속도(평균) 출력 — LN5 Extension to Videos (2/3, 3/3)
// Harris 함수는 별도 파일 cornerDetector.cpp 에 구현 (슬라이드 요구사항)
// 실행: 이 폴더의 아무 파일이나 열고 Cmd+Shift+B (폴더의 .cpp 를 전부 같이 빌드)
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include <cstdio>
#include "cornerDetector.h"

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
    VideoCapture capture;            // 수업 코드: VideoCapture capture(0);
    openWorkingCamera(capture);      // 맥: 영상이 나오는 카메라 자동 선택
    Mat frame, cornerMap;            // cornerMap 크기는 첫 프레임 이후에 정해진다
    if (!capture.isOpened()) {
        printf("Couldn't open the web camera...\n");
        return -1;
    }

    double total = 0;
    int frames = 0;

    while (true) {
        capture >> frame;
        if (frame.empty()) continue;
        resize(frame, frame, Size(320, 240));

        // ★ 프레임마다 처리 시간 측정 → 평균 처리 속도
        // (윈도우 슬라이드: QueryPerformanceFrequency / QueryPerformanceCounter 와 같은 역할)
        double t1 = getTickCount();
        int n = HarrisCornerDetector(frame, cornerMap);
        double diff = (getTickCount() - t1) / getTickFrequency();     // 초

        total += diff;
        frames++;
        printf("%f sec (avg %f sec, %.1f fps), corners %d\n",
               diff, total / frames, frames / total, n);

        imshow("Corner Detection", cornerMap);
        if (waitKey(30) >= 0) break;
    }
    return 0;
}
