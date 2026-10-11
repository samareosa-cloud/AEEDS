// W5-2. cornerDetector.h — Harris 함수를 별도 파일로 분리 (LN5 Extension to Videos 2/3)
#pragma once
#include <opencv2/opencv.hpp>

// frame(BGR) 에서 Harris corner 를 찾아 cornerMap(= frame 복사본)에 원으로 그린다
// 반환값: 검출된 corner 수
int HarrisCornerDetector(const cv::Mat& frame, cv::Mat& cornerMap);
