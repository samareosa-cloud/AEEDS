// SDM(RoboPai/sdm, chengzhengxin/sdm)을 OpenCV 4 / 5 에서 빌드하기 위한 호환 헤더
// 반드시 #include "ldmarkmodel.h" 보다 먼저 include 할 것
#pragma once
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <cmath>

// OpenCV 5: Haar CascadeClassifier가 opencv_contrib의 xobjdetect 모듈로 이동
// (이름은 cv::CascadeClassifier 그대로, opencv.hpp에는 포함 안 됨)
#if CV_VERSION_MAJOR >= 5
#include <opencv2/xobjdetect.hpp>
#endif

// OpenCV 2.4 시절 C 상수들 (OpenCV 4: 기본 include에서 빠짐, OpenCV 5: 완전히 삭제)
#ifndef CV_BGR2GRAY
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#endif
#ifndef CV_RGBA2GRAY
#define CV_RGBA2GRAY cv::COLOR_RGBA2GRAY
#endif
#ifndef CV_LOAD_IMAGE_GRAYSCALE
#define CV_LOAD_IMAGE_GRAYSCALE cv::IMREAD_GRAYSCALE
#endif
#ifndef CV_PCA_DATA_AS_ROW
#define CV_PCA_DATA_AS_ROW cv::PCA::DATA_AS_ROW
#endif

// OpenCV 5: decomposeProjectionMatrix가 calib3d -> geometry 모듈로 이동.
// 설치본에 따라 opencv.hpp에서 안 보이므로, SDM이 쓰는 형태(Mat 6개 + Vec3d)의
// 오버로드를 직접 제공한다. drawPose()는 오일러 각(Pitch/Yaw/Roll)만 쓰므로 그것만 계산.
// 계산은 OpenCV RQDecomp3x3 와 동일 (Givens 회전으로 Qx, Qy, Qz 분해)
namespace cv {
inline void decomposeProjectionMatrix(const Mat& projMatrix, Mat&, Mat&, Mat&,
                                      Mat&, Mat&, Mat&, Vec3d& eulerAngles)
{
    Matx34d P;
    projMatrix.convertTo(P, CV_64F);
    Matx33d M(P(0, 0), P(0, 1), P(0, 2), P(1, 0), P(1, 1), P(1, 2), P(2, 0), P(2, 1), P(2, 2));
    double z, c, s;

    // x축 회전 Qx
    s = std::abs(M(2, 1)) > DBL_EPSILON ? M(2, 1) : 0.;
    c = std::abs(M(2, 1)) > DBL_EPSILON ? M(2, 2) : 1.;
    z = 1. / std::sqrt(c * c + s * s); c *= z; s *= z;
    Matx33d Qx(1, 0, 0, 0, c, s, 0, -s, c);
    Matx33d R = M * Qx;
    R(2, 1) = 0;

    // y축 회전 Qy
    s = std::abs(R(2, 0)) > DBL_EPSILON ? -R(2, 0) : 0.;
    c = std::abs(R(2, 0)) > DBL_EPSILON ? R(2, 2) : 1.;
    z = 1. / std::sqrt(c * c + s * s); c *= z; s *= z;
    Matx33d Qy(c, 0, -s, 0, 1, 0, s, 0, c);
    M = R * Qy;
    M(2, 0) = 0;

    // z축 회전 Qz
    s = std::abs(M(1, 0)) > DBL_EPSILON ? M(1, 0) : 0.;
    c = std::abs(M(1, 0)) > DBL_EPSILON ? M(1, 1) : 1.;
    z = 1. / std::sqrt(c * c + s * s); c *= z; s *= z;
    Matx33d Qz(c, s, 0, -s, c, 0, 0, 0, 1);
    R = M * Qz;

    // 분해 모호성 처리 (R의 대각 성분이 양수가 되도록 180도 회전)
    if (R(0, 0) < 0) {
        if (R(1, 1) < 0) {
            Qz(0, 0) *= -1; Qz(0, 1) *= -1; Qz(1, 0) *= -1; Qz(1, 1) *= -1;
        } else {
            Qz = Qz.t();
            Qy(0, 0) *= -1; Qy(0, 2) *= -1; Qy(2, 0) *= -1; Qy(2, 2) *= -1;
        }
    } else if (R(1, 1) < 0) {
        Qz = Qz.t();
        Qy = Qy.t();
        Qx(1, 1) *= -1; Qx(1, 2) *= -1; Qx(2, 1) *= -1; Qx(2, 2) *= -1;
    }

    eulerAngles = Vec3d(
        std::acos(Qx(1, 1)) * (Qx(1, 2) >= 0 ? 1 : -1) * (180.0 / CV_PI),
        std::acos(Qy(0, 0)) * (Qy(2, 0) >= 0 ? 1 : -1) * (180.0 / CV_PI),
        std::acos(Qz(0, 0)) * (Qz(0, 1) >= 0 ? 1 : -1) * (180.0 / CV_PI));
}
} // namespace cv
