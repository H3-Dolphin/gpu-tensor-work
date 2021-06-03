#pragma once
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>

void createGaussianKernel(double* weights, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double sigma, const int imstep, const bool isRectangle = true);
void createGaussianKernel(double* weights, int* space_ofs, int& maxk, const int radius, const double sigma, const int imstep, const bool isRectangle = true);

void createGaussianKernel(float* weights, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const float sigma, const int imstep, const bool isRectangle = true);
void createGaussianKernel(float* weights, int* space_ofs, int& maxk, const int radius, const float sigma, const int imstep, const bool isRectangle = true);

void createGaussianKernel(half* weights, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const float sigma, const int imstep, const bool isRectangle = true);
void createGaussianKernel(half* weights, int* space_ofs, int& maxk, const int radius, const float sigma, const int imstep, const bool isRectangle = true);


// experimental function
// for debug
void printRegister(const __m256i m, const int bdepth);
void printRegister(const __m256 m);
void printRegister(const __m256d m);
void printRegister(const __m128i m, const int bdepth);
void printRegister(const __m128 m);
void printRegister(const __m128d m);

// expand image to only horizontal direction
void myCopyMakeBorder_H_32F(cv::Mat& img, const int& left, const int& right);
void myCopyMakeBorder_H_8U(cv::Mat& img, const int& left, const int& right);
// expand image to only vertical direction
void myCopyMakeBorder_V_32F(cv::Mat& img, const int& top, const int& bottom);
void myCopyMakeBorder_V_8U(cv::Mat& img, const int& top, const int& bottom);

// calculate mean absolute error (MAE)
double calcMAE(const cv::Mat& src1_, const cv::Mat& src2_);
// calculate mean squared error (MSE)
double calcMSE(const cv::Mat& src1_, const cv::Mat& src2_);
// calculate peak signal-to-noise ratio (PSNR)
double calcPSNR(const cv::Mat& src1, const cv::Mat& src2);

enum
{
	TIME_AUTO = 0,
	TIME_NSEC,
	TIME_MSEC,
	TIME_SEC,
	TIME_MIN,
	TIME_HOUR,
	TIME_DAY
};

class CalcTime
{
	int64 pre{};
	std::string mes;

	int timeMode{};

	double cTime{};
	bool _isShow{};

	int autoMode{};
	int autoTimeMode() const;
	std::vector<std::string> lap_mes;
public:

	void start();
	void setMode(const int mode);
	void setMessage(const std::string& src);
	void restart();
	double getTime();
	void show();
	void show(const std::string message);
	void lap(const std::string message);
	void init(const std::string message, const int mode, const bool isShow);

	CalcTime(const std::string message, const int mode = TIME_AUTO, const bool isShow = true);
	CalcTime(const char* message, const int mode = TIME_AUTO, const bool isShow = true);
	CalcTime();

	~CalcTime();
	std::string getTimeString();
};

class Stat
{
public:
	std::vector<double> data;
	Stat();
	~Stat();
	int num_data() const;
	double getMin();
	double getMax();
	double getMean();
	double getStd();
	double getMedian();

	void push_back(const double val);

	void clear();
	void show();
};

class ConsoleImage
{
private:
	int count;
	std::string windowName;
	std::vector<std::string> strings;
	bool isLineNumber;
public:
	void setIsLineNumber(bool isLine = true);
	bool getIsLineNumber() const;
	cv::Mat show;

	void init(const cv::Size size, const std::string wname);
	ConsoleImage();
	ConsoleImage(const cv::Size size, const std::string wname = "console");
	~ConsoleImage();

	void printData();
	void clear();

	void operator()(const std::string src);
	void operator()(const char* format, ...);
	void operator()(const cv::Scalar color, const char* format, ...);

	void flush(const bool isClear = true);
};