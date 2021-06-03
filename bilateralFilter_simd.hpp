#pragma once
#include <opencv2/opencv.hpp>

void BilateralFilter(const cv::Mat& src, cv::Mat& dest, const int r, const float sigma_r, const float sigma_s);
void BilateralFilter_SIMD(const cv::Mat& src, cv::Mat& dest, const int r, const float sigma_r, const float sigma_s);
void BilateralFilter_SIMD_kernelloop(const cv::Mat& src, cv::Mat& dest, const int r, float sigma_r, float sigma_s);
void inline _mm256_stream_ps_color(void* dst, const __m256 rsrc, const __m256 gsrc, const __m256 bsrc);


void bilateralTest();