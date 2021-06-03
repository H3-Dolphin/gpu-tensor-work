#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

std::vector<double> bilateralFilterCuda_naive_texture(const cv::Mat& src, cv::Mat& dest, const int r, const float sigma_r, const float sigma_s, const int loop = 100);
std::vector<double> bilateralFilterCuda_naive_global(const cv::Mat& src, cv::Mat& dest, const int r, const float sigma_r, const float sigma_s, const int loop = 100);
