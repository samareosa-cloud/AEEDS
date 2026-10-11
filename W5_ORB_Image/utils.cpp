// W5-3. utils.cpp — ORB 매칭 보조 함수
#include "utils.h"
#include <opencv2/opencv.hpp>

// findHomography / findFundamentalMat 선언 위치가 버전마다 다름
//  - OpenCV 5 (Homebrew): opencv2/geometry/3d.hpp  ← opencv.hpp 가 자동으로 안 불러옴
//  - OpenCV 3.4 / 4.x   : opencv2/calib3d.hpp
#if __has_include(<opencv2/geometry/3d.hpp>)
#include <opencv2/geometry/3d.hpp>
#elif __has_include(<opencv2/3d.hpp>)
#include <opencv2/3d.hpp>
#else
#include <opencv2/calib3d.hpp>
#endif

using namespace cv;
using std::vector;

// RANSAC 방법 번호. OpenCV 3.4 / 4.x / 5.x 모두 값이 8 로 같다.
// OpenCV 5 는 이 상수가 opencv2/geometry/3d.hpp 로 옮겨져서 opencv.hpp 만으로는
// 안 보일 수 있으므로 값을 직접 쓴다. (슬라이드/윈도우 코드: RANSAC, FM_RANSAC)
static const int METHOD_RANSAC = 8;      // = cv::RANSAC
static const int METHOD_FM_RANSAC = 8;   // = cv::FM_RANSAC

void draw_keypoints(Mat& img, const vector<KeyPoint>& kpts)
{
    for (size_t i = 0; i < kpts.size(); i++) {
        int x = (int)(kpts[i].pt.x + .5), y = (int)(kpts[i].pt.y + .5);
        float radius = kpts[i].size / 2.0f;
        circle(img, Point(x, y), (int)(radius * 2.5f), Scalar(0, 255, 0), 1);   // 초록: 크기(scale)
        circle(img, Point(x, y), 1, Scalar(0, 0, 255), -1);                     // 빨강: 위치
    }
}

void matches2points_nndr(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
                         const vector<vector<DMatch>>& matches, vector<Point2f>& pmatches, float nndr)
{
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].size() < 2) continue;
        // ★ NNDR: 1등 거리 < nndr × 2등 거리 일 때만 채택 → 점 쌍으로 저장
        DMatch dmatch = matches[i][0];
        float dist1 = matches[i][0].distance;    // 가장 가까운 후보
        float dist2 = matches[i][1].distance;    // 두 번째 후보
        if (dist1 < nndr * dist2) {
            pmatches.push_back(train[dmatch.queryIdx].pt);
            pmatches.push_back(query[dmatch.trainIdx].pt);
        }
    }
}

void compute_inliers_ransac(const vector<Point2f>& matches, vector<Point2f>& inliers,
                            float error, bool use_fund)
{
    vector<Point2f> points1, points2;
    int npoints = (int)matches.size() / 2;
    Mat status = Mat::zeros(npoints, 1, CV_8UC1);

    for (size_t i = 0; i < matches.size(); i += 2) {   // 쌍 분리
        points1.push_back(matches[i]);
        points2.push_back(matches[i + 1]);
    }

    if (npoints > 8) {
        // ★ RANSAC: homography 추정, 오차 ≤ error(px) 인 매칭만 inlier (status = 1)
        if (use_fund)
            findFundamentalMat(points1, points2, METHOD_FM_RANSAC, error, 0.99, status);
        else
            findHomography(points1, points2, METHOD_RANSAC, error, status);

        for (int i = 0; i < npoints; i++) {
            if (status.at<uchar>(i) == 1) {
                inliers.push_back(points1[i]);
                inliers.push_back(points2[i]);
            }
        }
    }
}

void draw_inliers(const Mat& img1, const Mat& img2, Mat& img_com, const vector<Point2f>& ptpairs, int color)
{
    img_com = Mat::zeros(std::max(img1.rows, img2.rows), img1.cols + img2.cols, CV_8UC3);
    img1.copyTo(img_com(Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(img_com(Rect(img1.cols, 0, img2.cols, img2.rows)));

    Scalar c = (color == 0) ? Scalar(255, 0, 0) : Scalar(255, 255, 0);   // 0: 파랑, 그 외: 하늘색
    for (size_t i = 0; i < ptpairs.size(); i += 2) {
        Point p1((int)(ptpairs[i].x + .5), (int)(ptpairs[i].y + .5));
        Point p2((int)(ptpairs[i + 1].x + img1.cols + .5), (int)(ptpairs[i + 1].y + .5));   // 오른쪽 영상: + img1.cols
        line(img_com, p1, p2, c, 2);
    }
}
