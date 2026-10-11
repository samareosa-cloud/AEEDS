// W5-3. utils.h — ORB 매칭 보조 함수 (원본: github.com/pablofdezalc/akaze 의 utils, 핵심만 정리)
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// keypoint 를 원으로 표시
void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts);

// NNDR(ratio test) 통과한 매칭 → 점 쌍 [p1, p2, p1, p2, ...]
void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
                         const std::vector<cv::KeyPoint>& query,
                         const std::vector<std::vector<cv::DMatch>>& matches,
                         std::vector<cv::Point2f>& pmatches, float nndr);

// RANSAC 으로 homography 추정 → inlier 점 쌍만 남김
void compute_inliers_ransac(const std::vector<cv::Point2f>& matches,
                            std::vector<cv::Point2f>& inliers,
                            float error, bool use_fund);

// 두 영상을 좌우로 붙이고 inlier 쌍을 선으로 연결
void draw_inliers(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
                  const std::vector<cv::Point2f>& ptpairs, int color);
