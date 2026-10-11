// W6-2. Lucas-Kanade optical flow 직접 구현 (웹캠, 개념 이해용) — LN6 Technical Details (2/8 ~ 4/8)
// OFE: Ix·u + Iy·v + It = 0,  3x3 patch 는 같은 (u, v) → 최소제곱
//   [u v]ᵀ = M⁻¹ · [-ΣIxIt  -ΣIyIt]ᵀ,   M = Harris 의 structure tensor
// W6-1 (Farneback) 과 나란히 돌려보며 비교: 손을 빠르게 움직이면 LK 는 잘 못 잡음 (small motion 가정)
// 실행: 이 파일 열고 Cmd+Shift+B  /  종료: 영상 창 클릭 후 아무 키
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cmath>

#define Width  320
#define Height 240

using namespace cv;

int main()
{
    VideoCapture capture(0);
    if (!capture.isOpened()) { printf("Couldn't open the web camera...\n"); return -1; }

    const int step = 5;          // 5픽셀 격자마다 계산
    const int r = 2;             // patch 반지름: 1 → 3x3 (슬라이드), 2 → 5x5 (웹캠에서 더 안정적)
    Mat frame, img, curr, prev;

    while (true) {
        capture >> frame;
        if (frame.empty()) continue;
        resize(frame, img, Size(Width, Height));
        cvtColor(img, curr, COLOR_BGR2GRAY);
        GaussianBlur(curr, curr, Size(5, 5), 1.0);     // 노이즈 줄이기 (미분 전에 LPF)

        if (!prev.empty()) {
            double t1 = getTickCount();
            int valid = 0;

            for (int y = r + 1; y < Height - r - 1; y += step) {
                for (int x = r + 1; x < Width - r - 1; x += step) {

                    // ★ patch 의 방정식들 → ΣIx², ΣIy², ΣIxIy, ΣIxIt, ΣIyIt
                    float Sxx = 0, Syy = 0, Sxy = 0, Sxt = 0, Syt = 0;
                    for (int dy = -r; dy <= r; dy++) {
                        for (int dx = -r; dx <= r; dx++) {
                            int yy = y + dy, xx = x + dx;
                            // 공간 미분: center difference (f(x+1) - f(x-1)) / 2
                            float Ix = (curr.at<uchar>(yy, xx + 1) - curr.at<uchar>(yy, xx - 1)) / 2.0f;
                            float Iy = (curr.at<uchar>(yy + 1, xx) - curr.at<uchar>(yy - 1, xx)) / 2.0f;
                            // 시간 미분: 현재 프레임 밝기 - 이전 프레임 밝기
                            float It = (float)curr.at<uchar>(yy, xx) - prev.at<uchar>(yy, xx);

                            Sxx += Ix * Ix;  Syy += Iy * Iy;  Sxy += Ix * Iy;
                            Sxt += Ix * It;  Syt += Iy * It;
                        }
                    }

                    // ★ 풀림 조건: M 가역 + 작은 고유값이 너무 작지 않을 것 (flat / edge 는 건너뜀)
                    float det = Sxx * Syy - Sxy * Sxy;
                    float tr = Sxx + Syy;
                    float lambdaMin = 0.5f * (tr - sqrt((Sxx - Syy) * (Sxx - Syy) + 4 * Sxy * Sxy));
                    if (fabs(det) < 1e-6f || lambdaMin < 200.0f) continue;

                    // ★ 2x2 역행렬 = 1/det [d -b; -c a]
                    float u = ( Syy * (-Sxt) - Sxy * (-Syt)) / det;
                    float v = (-Sxy * (-Sxt) + Sxx * (-Syt)) / det;

                    if (u * u + v * v < 0.25f) continue;     // 거의 안 움직인 점은 그리지 않음
                    line(img, Point(x, y), Point(cvRound(x + 3 * u), cvRound(y + 3 * v)), Scalar(0, 255, 0));
                    circle(img, Point(x, y), 1, Scalar(0, 0, 255), -1);
                    valid++;
                }
            }
            double ms = 1000.0 * (getTickCount() - t1) / getTickFrequency();
            printf("LK: %d moving points, %.1f ms\n", valid, ms);
        }
        curr.copyTo(prev);

        imshow("Lucas-Kanade (green: flow x3)", img);
        if (waitKey(20) >= 0) break;
    }
    return 0;
}
