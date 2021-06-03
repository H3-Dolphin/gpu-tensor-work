#pragma once
#include <opencv2/opencv.hpp>

#define cudaSafeCall(call)\
do{\
	const cudaError_t err = call;\
	if (err != cudaSuccess) {\
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__, __LINE__, cudaGetErrorString(err));\
		exit(EXIT_FAILURE);\
	}\
}while(0)


#define cudaPitchPos(data, y, x, type, pitch) ((type*)((char*)(data) + (y) * (pitch)) + (x))



inline void showGPUInfo(const int deviceId = 0)
{
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
	const int cuda_devices_number = cv::cuda::getCudaEnabledDeviceCount();
	std::cout << "CUDA Device(s) Number: " << cuda_devices_number << std::endl;

	cv::cuda::DeviceInfo info(0);
	std::cout <<
		"[CUDA Device 0]" << std::endl <<
		"name: " << info.name() << std::endl <<
		"majorVersion: " << info.majorVersion() << std::endl <<
		"minorVersion: " << info.minorVersion() << std::endl <<
		"multiProcessorCount: " << info.multiProcessorCount() << std::endl <<
		"sharedMemPerBlock: " << info.sharedMemPerBlock() << std::endl <<
		"freeMemory: " << info.freeMemory() << std::endl <<
		"totalMemory: " << info.totalMemory() << std::endl <<
		"isCompatible: " << info.isCompatible() << std::endl <<
		"supports(FEATURE_SET_COMPUTE_10): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_10) << std::endl <<
		"supports(FEATURE_SET_COMPUTE_11): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_11) << std::endl <<
		"supports(FEATURE_SET_COMPUTE_12): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_12) << std::endl <<
		"supports(FEATURE_SET_COMPUTE_13): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_13) << std::endl <<
		"supports(FEATURE_SET_COMPUTE_20): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_20) << std::endl <<
		"supports(FEATURE_SET_COMPUTE_21): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_21) << std::endl <<
		"supports(FEATURE_SET_COMPUTE_30): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_30) << std::endl <<
		"supports(FEATURE_SET_COMPUTE_35): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_35) << std::endl;
}