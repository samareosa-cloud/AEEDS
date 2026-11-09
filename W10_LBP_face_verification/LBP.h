// macos version
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

#define FACE_SIZE 128                                   // 얼굴 resize 크기
#define BLOCK 32                                        // block 32x32
#define STRIDE 16                                       // 16픽셀씩 이동 (overlap)
#define NBLOCK ((FACE_SIZE - BLOCK) / STRIDE + 1)       // 한 줄에 7개 -> 7x7 = 49
#define NBIN 256
#define DIM (NBLOCK * NBLOCK * NBIN)                    // 49 x 256 = 12544

void LBPimage(const cv::Mat& gray, cv::Mat& lbp);
void LBPdescriptor(const cv::Mat& frame, std::vector<float>& hist, int x, int y, int w, int h);
float computeSimilarity(const std::vector<float>& ref, const std::vector<float>& tar);