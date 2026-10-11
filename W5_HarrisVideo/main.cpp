// W5-2. 웹캠 프레임마다 Harris corner + 처리 속도(평균) 출력 — LN5 Extension to Videos (2/3, 3/3)
// Harris 함수는 별도 파일 cornerDetector.cpp 에 구현 (슬라이드 요구사항)
// 실행: 이 폴더의 파일을 연 상태에서 Cmd+Shift+P → "Tasks: Run Task" → "폴더 전체 .cpp 빌드 & 실행"
#include <opencv2/opencv.hpp>
#include <cstdio>
#include "cornerDetector.h"

using namespace cv;

int main()
{
    VideoCapture capture(0);         // 아이폰이 잡히면 1
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
