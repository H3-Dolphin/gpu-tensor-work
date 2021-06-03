#include <opencv2/opencv.hpp>
#include "bilateralFilter_simd.hpp"
#include "GaussianCPU.hpp"
#include <cuda_runtime.h>
#include <opencv2/cudaimgproc.hpp>
#include "tensor_sample.h"
#include "BilateralCuda_naive.cuh"
#include "GaussianCuda.cuh"
#include "utils.hpp"
#include "RealtimeO1CPU.hpp"


#define CV_LIB_PREFIX comment(lib, "opencv_"

#define CV_LIB_VERSION CVAUX_STR(CV_MAJOR_VERSION)\
    CVAUX_STR(CV_MINOR_VERSION)\
    CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define CV_LIB_SUFFIX CV_LIB_VERSION "d.lib")
#else
#define CV_LIB_SUFFIX CV_LIB_VERSION ".lib")
#endif

#define CV_LIBRARY(lib_name) CV_LIB_PREFIX CVAUX_STR(lib_name) CV_LIB_SUFFIX

#pragma CV_LIBRARY(core)
#pragma CV_LIBRARY(highgui)
#pragma CV_LIBRARY(imgcodecs)
#pragma CV_LIBRARY(imgproc)
#pragma CV_LIBRARY(ximgproc)
#pragma CV_LIBRARY(cudaimgproc)
#pragma CV_LIBRARY(cudaarithm)
#pragma comment(lib, "cudart.lib")

using namespace std;
using namespace cv;
using namespace cv::cuda;


int main(char** argv, int argc)
{
	//bilateralTest();
	//showGPUInfo();


	//////////////////////////////////////////////////////////////
	// 非公開のリポジトリではここに提案手法が存在します         //
	// 提案手法の出力画像: img/output_GF/提案手法(GPU内でim2col)//
	//////////////////////////////////////////////////////////////

	if (0)
	{
		Mat src8u = imread("img/lenna.png", IMREAD_GRAYSCALE);
		Mat srcf_ori;
		src8u.convertTo(srcf_ori, CV_32FC1);
		int key = 0;
		Stat st_naive;
		Stat st_mat;
		String wname = "console";
		ConsoleImage ci(Size(640, 480), wname);
		namedWindow(wname);
		int r = 2;
		int ds = 8;
		int ds_ = 0;
		createTrackbar("r", wname, &r, 100);
		createTrackbar("downsize scale", wname, &ds, 8);
		Mat srcf;
		while (key != 'q')
		{
			if (key == 'r')
			{
				st_naive.clear();
				st_mat.clear();
			}

			if (ds != ds_)
			{
				resize(srcf_ori, srcf, Size(), 1.f / float(ds), 1.f / float(ds));
				ds_ = ds;
			}

			const float sigma = r / 3.f;

			// reference
			Mat reff;
			Mat ref;
			{
				GaussianBlur(srcf, reff, Size(2 * r + 1, 2 * r + 1), sigma, 0.f, BORDER_REPLICATE);
				reff.convertTo(ref, CV_8UC1, 1.f, 0.5f);
			}

			// naive implementation
			Mat destf_naive;
			Mat dest_naive;
			{
				cudaEvent_t start, stop;
				float t = 0;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start);
				{
					NaiveGaussianCuda(srcf, destf_naive, sigma, r);
					cudaDeviceSynchronize();
				}
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);

				cudaEventElapsedTime(&t, start, stop);
				st_naive.push_back(t);

				destf_naive.convertTo(dest_naive, CV_8UC1, 1.f, 0.5f);
			}

			// matrix redefine implementation
			Mat destf_matrix;
			Mat dest_matrix;
			{
				cudaEvent_t start, stop;
				float t = 0;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start);
				{
					RedefineGaussianCudaNotTCU(srcf, destf_matrix, sigma, r);
					cudaDeviceSynchronize();
				}
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);

				cudaEventElapsedTime(&t, start, stop);
				st_mat.push_back(t);

				destf_matrix.convertTo(dest_matrix, CV_8UC1, 1.f, 0.5f);
			}

			Mat src;
			srcf.convertTo(src, CV_8UC1);
			imshow("src", src);
			imshow("dest_naive", dest_naive);
			imshow("dest_matrix", dest_naive);
			imshow("ref", ref);
			ci(format("r: %d", r));
			ci(format("sigma: %f", sigma));
			ci(format("PSNR ref-naive: %f", PSNR(reff, destf_naive)));
			ci(format("PSNR ref-matrix: %f", PSNR(reff, destf_matrix)));
			ci(format("time naive: %f", st_naive.getMedian()));
			ci(format("time mat: %f", st_mat.getMedian()));
			ci.flush();
			key = waitKey(5);
		}
		return 0;
	}

	// CUDA without TCUのグレーBF
	if (0)
	{
		const int r = 5;
		const int loop = 100;

		cv::Mat src = imread("img/lenna.png", cv::IMREAD_GRAYSCALE);
		cv::Mat src_pd;
		copyMakeBorder(src, src_pd, r, r, r, r, cv::BORDER_REPLICATE);

		// naive cuda impl using global memory
		cv::Mat dest_global(src.rows, src.cols, CV_8UC1);
		{
			auto calcTimes = bilateralFilterCuda_naive_global(src_pd, dest_global, r, 1.f, 16.0f, loop);
			//auto calcTimes = bilateralFilterCuda_naive_global(src_pd, dest_global, r, 100.f, r/3.f, loop);

			std::sort(calcTimes.begin(), calcTimes.end());
			std::cout << "Time own global (median): " << calcTimes[calcTimes.size() / 2] << " ms" << std::endl;

			imshow("Own BF CUDA global", dest_global);
		}

		// naive cuda impl using texture memory
		cv::Mat dest_texture(src.rows, src.cols, CV_8UC1);
		{
			auto calcTimes = bilateralFilterCuda_naive_texture(src_pd, dest_texture, r, 1.f, 16.0f, loop);
			std::sort(calcTimes.begin(), calcTimes.end());
			std::cout << "Time own texture (median): " << calcTimes[calcTimes.size() / 2] << " ms" << std::endl;

			imshow("Own BF CUDA texture", dest_texture);
		}


		// opencv cv::bilateralFilter
		cv::Mat dest_cv;
		{
			vector<double> calcTimes(loop);
			cv::TickMeter time;
			int loop_ = loop;
			while (loop_--) {
				time.start();
				cv::bilateralFilter(src, dest_cv, 2 * r + 1, 1.f, 16.0f, cv::BORDER_REPLICATE);

				time.stop();
				calcTimes.push_back(time.getTimeMilli());
			}
			std::sort(calcTimes.begin(), calcTimes.end());
			std::cout << "Time CV::BF (median): " << calcTimes[calcTimes.size() / 2] << " ms" << std::endl;

			// show image
			imshow("CV::BF", dest_cv);
			// PSNR
			cout << "PSNR: " << PSNR(dest_global, dest_cv) << "dB" << endl;
		}


		// opencv cv::cuda::bilateralFilter
		cv::Mat dest_cv_cuda;
		{
			cv::cuda::GpuMat src_gpu(src), dest_gpu;
			vector<double> calcTimes(loop);
			cv::TickMeter time;
			int loop_ = loop;
			while (loop_--) {
				time.start();
				cv::cuda::bilateralFilter(src_gpu, dest_gpu, 2 * r + 1, 1.f, 16.0f, cv::BORDER_REPLICATE);

				time.stop();
				calcTimes.push_back(time.getTimeMilli());
			}
			std::sort(calcTimes.begin(), calcTimes.end());
			std::cout << "Time CV::CUDA::BF (median): " << calcTimes[calcTimes.size() / 2] << " ms" << std::endl;

			// show image
			dest_gpu.download(dest_cv_cuda);
			imshow("CV::CUDA::BF", dest_cv_cuda);
			// PSNR
			cout << "PSNR: " << PSNR(dest_global, dest_cv_cuda) << "dB" << endl;
			cudaDeviceReset();
		}
		cv::waitKey();
		imwrite("img/dest_own_pd_GLOBAL.png", dest_global);
		imwrite("img/dest_own_pd_TEXTURE.png", dest_texture);
		imwrite("img/dest_cv.png", dest_cv);
		imwrite("img/dest_cv_cuda.png", dest_cv_cuda);

		return 0;
	}

	// TCUのサンプル
	if (0)
	{
		tensorTest();
	}

	// スカラのRealtimeO(1)BilateralFilter
	if (1)
	{
		Mat src8u = imread("img/lenna.png", IMREAD_GRAYSCALE);
		Mat dest = Mat::zeros(src8u.size(), CV_64FC1);
		Mat dest8u = Mat::zeros(src8u.size(), CV_8UC1);

		Mat srcf_ori;
		src8u.convertTo(srcf_ori, CV_64FC1);
		int r = 5;
		// sigma_r=100.0なら桁落ちせず
		double sigma_s = r / 3.0, sigma_r = 100.0;
		// double double
		RealtimeO1Scala(srcf_ori, dest, r, sigma_r, sigma_s);
		dest.convertTo(dest, CV_8UC1);
		imwrite("img/realtimeO1BF/scala/realtimeO1_r5_sigmar100.png", dest);
	}

	return 0;
}