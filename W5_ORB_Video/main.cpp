// W5-4. ORB matching in videos — LN5 Practical Example
// 참조 영상(ref.jpg, 없으면 첫 프레임)과 웹캠의 매 프레임을 ORB 로 매칭
// 실행: 이 폴더의 아무 파일이나 열고 Cmd+Shift+B (폴더의 .cpp 를 전부 같이 빌드)
// 팁: 책 표지나 포스터를 첫 프레임에 비추고 시작 → 움직여도 선이 따라오는지 확인
#include <opencv2/opencv.hpp>
#include <cstdio>
#include "utils.h"

using namespace cv;
using namespace std;

float MIN_H_ERROR = 2.50f;   // 키우면(예: 120) 거의 다 inlier → RANSAC 효과 비교해보기
float DRATIO = 0.80f;

int main()
{
    VideoCapture capture(0);
    if (!capture.isOpened()) { printf("Couldn't open the web camera...\n"); return -1; }

    Ptr<ORB> orb = ORB::create(1500, 1.2f, 4, 31, 0, 2, ORB::HARRIS_SCORE, 31);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    Mat frame, gray, ref, refRGB;
    vector<KeyPoint> kptsRef;
    Mat descRef;

    // ★ 참조 영상 특징은 루프 밖에서 한 번만 계산
    ref = imread("ref.jpg", IMREAD_GRAYSCALE);
    if (ref.empty()) {                          // 참조 영상이 없으면 첫 프레임 사용
        for (int i = 0; i < 30 && frame.empty(); i++) capture >> frame;   // 카메라 켜질 때까지 대기
        if (frame.empty()) { printf("No frame from camera\n"); return -1; }
        cvtColor(frame, ref, COLOR_BGR2GRAY);
    }
    resize(ref, ref, Size(400, 400));
    orb->detectAndCompute(ref, noArray(), kptsRef, descRef);
    cvtColor(ref, refRGB, COLOR_GRAY2BGR);

    while (true) {
        capture >> frame;
        if (frame.empty()) continue;
        resize(frame, frame, Size(400, 400));
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // ★ 매 프레임: detect → knnMatch → NNDR → RANSAC
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
        printf("matches %d, inliers %d, ratio %.1f%%, %.1f ms\n", nm, ni, nm ? 100.0 * ni / nm : 0.0, ms);

        Mat frameRGB, com;
        cvtColor(gray, frameRGB, COLOR_GRAY2BGR);
        draw_inliers(refRGB, frameRGB, com, inliers, 0);
        imshow("ORB video (left: reference / right: webcam)", com);
        if (waitKey(30) >= 0) break;
    }
    return 0;
}
