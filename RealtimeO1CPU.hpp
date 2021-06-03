#pragma once
#include <opencv2/opencv.hpp>

void RealtimeO1Scala(const cv::Mat& src, cv::Mat& dest, const int r, const double sigma_r, const double sigma_s);