// W5-4. ORB matching in videos — LN5 Practical Example
// "Compare the reference image with video sequences"
//   - 참조 영상(reference) 한 장을 정해두고
//   - 웹캠의 매 프레임과 ORB 로 매칭해서 같은 특징점끼리 선으로 연결
//
// 사용법
//   1) 실행하면 웹캠 화면이 뜬다 (카메라가 켜질 때까지 잠깐 검은 화면일 수 있음)
//   2) 책 표지 / 포스터 / 과자 상자처럼 무늬가 많은 물체를 비추고 SPACE → 참조 영상으로 저장
//   3) 물체를 움직이거나 돌려보면 왼쪽(참조) ↔ 오른쪽(웹캠) 사이에 파란 선이 따라옴
//   r : 참조 영상 다시 찍기,  q 또는 ESC : 종료
//   (같은 폴더에 ref.jpg 가 있으면 그 사진을 참조 영상으로 바로 사용)
// 실행: 이 폴더의 아무 파일이나 열고 Cmd+Shift+B (폴더의 .cpp 를 전부 같이 빌드)
#include <opencv2/opencv.hpp>
#include <cstdio>
#include "utils.h"

using namespace cv;
using namespace std;

float MIN_H_ERROR = 2.50f;   // RANSAC inlier 허용 오차(px). 키우면(예: 120) 거의 다 inlier → RANSAC 효과 비교해보기
float DRATIO = 0.80f;        // NNDR ratio

const Size SZ(400, 400);

// 카메라에서 "제대로 된" 프레임 한 장 읽기 (맥: 켜지는 동안 빈/검은 프레임이 옴)
static bool readFrame(VideoCapture& cap, Mat& frame)
{
    cap >> frame;
    if (frame.empty()) return false;
    resize(frame, frame, SZ);
    return true;
}

int main()
{
    VideoCapture capture(0);                 // 아이폰(연속성 카메라)이 잡히면 1
    if (!capture.isOpened()) { printf("Couldn't open the web camera...\n"); return -1; }

    Ptr<ORB> orb = ORB::create(1500, 1.2f, 4, 31, 0, 2, ORB::HARRIS_SCORE, 31);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    Mat frame, gray, ref, refRGB, descRef;
    vector<KeyPoint> kptsRef;

    // ★ 참조 영상 준비: ref.jpg 가 있으면 사용
    ref = imread("ref.jpg", IMREAD_GRAYSCALE);
    if (!ref.empty()) {
        resize(ref, ref, SZ);
        orb->detectAndCompute(ref, noArray(), kptsRef, descRef);
        cvtColor(ref, refRGB, COLOR_GRAY2BGR);
        printf("reference = ref.jpg (%d keypoints)\n", (int)kptsRef.size());
    }

    while (true) {
        if (!readFrame(capture, frame)) { waitKey(10); continue; }
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        int key = -1;

        if (ref.empty()) {
            // ★ 아직 참조 영상이 없으면: 웹캠 화면만 보여주고 SPACE 를 기다림
            Mat preview = frame.clone();
            bool dark = mean(gray)[0] < 15;              // 카메라 예열 중(검은 화면)
            putText(preview, dark ? "camera warming up..." : "SPACE: capture reference",
                    Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            imshow("ORB video (left: reference / right: webcam)", preview);
            key = waitKey(30);

            if (key == ' ' && !dark) {
                gray.copyTo(ref);
                orb->detectAndCompute(ref, noArray(), kptsRef, descRef);
                cvtColor(ref, refRGB, COLOR_GRAY2BGR);
                printf("reference captured: %d keypoints\n", (int)kptsRef.size());
                if (kptsRef.size() < 50) {
                    printf("  -> 특징점이 너무 적어요. 무늬가 많은 물체를 비추고 다시 SPACE\n");
                    ref.release();
                }
            }
        }
        else {
            // ★ 매 프레임: detect → knnMatch(k=2) → NNDR → RANSAC
            double t1 = getTickCount();
            vector<KeyPoint> kpts;
            Mat desc;
            orb->detectAndCompute(gray, noArray(), kpts, desc);

            vector<vector<DMatch>> dmatches;
            vector<Point2f> matches, inliers;
            if (!desc.empty() && !descRef.empty()) {
                matcher->knnMatch(descRef, desc, dmatches, 2);
                matches2points_nndr(kptsRef, kpts, dmatches, matches, DRATIO);
                compute_inliers_ransac(matches, inliers, MIN_H_ERROR, false);
            }
            double ms = 1000.0 * (getTickCount() - t1) / getTickFrequency();

            int nm = matches.size() / 2, ni = inliers.size() / 2;
            printf("keypoints %d, matches %d, inliers %d, ratio %.1f%%, %.1f ms\n",
                   (int)kpts.size(), nm, ni, nm ? 100.0 * ni / nm : 0.0, ms);

            // 왼쪽 = 참조 영상, 오른쪽 = 웹캠, inlier 쌍을 파란 선으로 연결
            Mat frameRGB, com;
            cvtColor(gray, frameRGB, COLOR_GRAY2BGR);
            draw_inliers(refRGB, frameRGB, com, inliers, 0);
            putText(com, format("matches %d  inliers %d   (r: retake, q: quit)", nm, ni),
                    Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            imshow("ORB video (left: reference / right: webcam)", com);
            key = waitKey(30);

            if (key == 'r') { ref.release(); printf("retake reference: SPACE\n"); }
        }

        if (key == 'q' || key == 27) break;      // q 또는 ESC
    }
    return 0;
}
