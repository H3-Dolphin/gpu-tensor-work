#include "GaussianCuda.cuh"
#include "cuda_utils.hpp"
#include "utils.hpp"
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>
#include <mma.h>
#include "matMul.cuh"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include "NaiveIntegralCuda.cuh"

#define SUGGEST 1
#define Calc_BoxFiter 1
#define Save_Output 1

namespace tcu {
	constexpr int WMMA_M = 16;
	constexpr int WMMA_N = 16;
	constexpr int WMMA_K = 16;
	constexpr int WMMA_TILE_SIZE = (WMMA_M * WMMA_N);


	template<class MatType, class MajorType>
	using HalfFrag = nvcuda::wmma::fragment<MatType, WMMA_M, WMMA_N, WMMA_K, half, MajorType>;

	template<class MatType>
	using HalfRowMajorFrag = nvcuda::wmma::fragment<MatType, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>;

	template<class MatType>
	using HalfColMajorFrag = nvcuda::wmma::fragment < MatType, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major>;

	using HalfAccFrag = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>;

	using FloatAccFrag = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
}



using namespace cv;
using namespace std;


// ここで、長方形の足し引きでO(1)で畳みこませる
__global__ void BoxFilterKernelTCU(float* scan, float* dest, const size_t dpitch, const size_t spitch, const int dWidth, const int dHeight, const int r, const int scanSize)
{
	// 座標
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;

	// 端
	if (idx < 0 || idy < 0 || idx >= dWidth || idy >= dHeight)
	{
		return;
	}
	// 畳み込みの中心
	const float* scan_sp = (float*)((char*)scan + (idy + r + 1) * spitch) + (idx + r + 1);

	float sum = 0;
	const int l = (2 * r + 1);

	// 積分画像によりO(1)で求めている
	sum = scan_sp[scanSize * r + r] - scan_sp[scanSize * r - (r + 1)] - scan_sp[-scanSize * (r + 1) + r] + scan_sp[-scanSize * (r + 1) - (r + 1)];
	sum /= (l * l);

	// 格納したいdestの画素を指定して入れる
	float* dp = (float*)((char*)dest + idy * dpitch) + idx;
	*dp = sum;
}

// 上のカーネルのラッパー
__host__ void BoxFilterKernelTCUWrapper(float* scan, float* destDevice, const size_t dpitch, const size_t spitch, const int dWidth, const int dHeight, const int r, const int scanSize) {


	dim3 block(32, 4);
	//dim3 block(1, 1);

	dim3 grid(divUp(dWidth, block.x), divUp(dHeight, block.y));
	//dim3 grid(1);
	BoxFilterKernelTCU << <grid, block, 0 >> > (scan, destDevice, dpitch, spitch, dWidth, dHeight, r, scanSize);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}


#define SHOW_CALC_TIME 0

#define MEASURE_TIME(f, str)\
do{\
	cudaEvent_t start, stop;\
	float __t = 0;\
	cudaEventCreate(&start);\
	cudaEventCreate(&stop); \
	cudaEventRecord(start);\
	{\
		f();\
	}\
	cudaEventRecord(stop);\
	cudaEventSynchronize(stop);\
	cudaEventElapsedTime(&__t, start, stop);\
	cout << str <<": " << __t << endl;\
}while(0)

__host__ void BoxFilterTCU(Mat src, Mat& dest, const float sigma, const int r, const int scanSize)
{
	CV_Assert(src.channels() == 1);
	Mat temp, scan;
	cuda::copyMakeBorder(src, temp, r, r, r, r, BORDER_REPLICATE);

	//scan = Mat(src.rows, src.cols, CV_32FC1);
	scan = Mat(scanSize, scanSize, CV_32FC1);
	dest = Mat(src.rows, src.cols, CV_32FC1);

	const size_t kernel_size = (2 * r + 1) * (2 * r + 1);

	float sum_tmp = 0.0;
	int off = 0;
	// LAUのAを作る
	vector<float> A_Host(scanSize * scanSize);
	vector<half> A_Host16(scanSize * scanSize);

	Mat temp_tmp = temp;
	
	// 積分画像のtopとleftをgpuで0にさせる処理
	cuda::copyMakeBorder(temp_tmp, temp, 1, 0, 1, 0, BORDER_CONSTANT);
	if (scanSize > 256)
	{
		float* tptr = temp.ptr<float>();
#pragma omp parallel for
		for (int i = 0; i < scanSize * scanSize; i++) {
			A_Host[i] = tptr[i];
		}
	}
	else
	{
		float* tptr = temp.ptr<float>();
#pragma omp parallel for
		for (int i = 0; i < scanSize * scanSize; i++) {
			A_Host16[i] = __float2half(tptr[i]);
		}
	}

	// メモリプール
	// 宣言すれば、FromPoolとしなくてもPoolから確保してくれる
	cudaMemPool_t memPool0;
	int dev0 = 0;
	cudaSafeCall(cudaDeviceGetDefaultMemPool(&memPool0, dev0));
	uint64_t thresholdVal = ULONG_MAX;
	cudaSafeCall(cudaMemPoolSetAttribute(memPool0, cudaMemPoolAttrReleaseThreshold, (void*)&thresholdVal));

	// 出力画像用のpitchとか
	float* A_Device = nullptr;
	float* LA_Device = nullptr;
	float* LAU_Device = nullptr;

	half* A_Device16 = nullptr;
	half* LA_Device16 = nullptr;

	//float* scanafterDevice = nullptr;
	float* destDevice = nullptr;
	size_t tpitch = 0;
	size_t dpitch = 0;
	size_t destpitch = 0;

	// streasmによりmallocとmemcpyをAsync
	// CudaMallocPitchAsyncはまだ存在しないのでMallocAsyncとMemcpyAsyncを利用
	cudaStream_t stream_0, stream_1, stream_2;
	cudaStreamCreate(&stream_0);
	cudaStreamCreate(&stream_1);
	cudaStreamCreate(&stream_2);

	{
		auto f = [&]
		{
			if (scanSize > 256)
			{
				// AとLAを確保、H2D(TF32)
				cudaSafeCall(cudaMallocAsync(reinterpret_cast<void**>(&A_Device), scanSize * scanSize * sizeof(float), stream_0));
				cudaSafeCall(cudaMemcpyAsync(A_Device, &A_Host[0], scanSize * scanSize * sizeof(float), cudaMemcpyHostToDevice, stream_0));
				cudaSafeCall(cudaMallocAsync(reinterpret_cast<void**>(&LA_Device), scanSize * scanSize * sizeof(float), stream_1));
			}
			else
			{
				// AとLAの確保(FP16)
				cudaSafeCall(cudaMallocAsync(reinterpret_cast<void**>(&A_Device16), scanSize * scanSize * sizeof(half), stream_0));
				cudaSafeCall(cudaMemcpyAsync(A_Device16, &A_Host16[0], scanSize * scanSize * sizeof(half), cudaMemcpyHostToDevice, stream_0));
				cudaSafeCall(cudaMallocAsync(reinterpret_cast<void**>(&LA_Device16), scanSize * scanSize * sizeof(half), stream_1));
			}

			// LAUを確保(共通)
			cudaSafeCall(cudaMallocAsync(reinterpret_cast<void**>(&LAU_Device), scanSize * scanSize * sizeof(float), stream_2));

			// 出力画像用のcudaMallocPitch
			cudaSafeCall(cudaMallocPitch(reinterpret_cast<void**>(&destDevice), &destpitch, dest.cols * sizeof(float), scanSize));
		};
#if !SHOW_CALC_TIME
		f();
#else
		MEASURE_TIME(f, "malloc and copy");
#endif
	}

	{
		auto f = [&]
		{

#if !SUGGEST
			// 計測(PCSJの比較手法)
			{
				// ここで比較手法の、端の処理やH2D、D2Hを含めた全体的な処理の計測
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				int loop = 1000;
				std::vector<double> calcTimes_compare(loop);
				Mat cv_cuda_scan;
				vector<float> NaiveIntegral_Host;
				while (loop--) {
				    cudaEventRecord(start);
					Mat temp_fair;
					cuda::copyMakeBorder(src, temp_fair, r, r, r, r, BORDER_REPLICATE);
					temp_fair.convertTo(temp_fair, CV_8UC1);
					cv_cuda_scan = Mat(scanSize, scanSize, CV_32FC1);
					NaiveIntegral(temp_fair, cv_cuda_scan);
					NaiveIntegral_Host.resize(scanSize * scanSize);

					cudaEventRecord(stop);
					cudaEventSynchronize(stop);
					float milliseconds = 0;
					cudaEventElapsedTime(&milliseconds, start, stop);
					calcTimes_compare.push_back(milliseconds);
				}
				std::sort(calcTimes_compare.begin(), calcTimes_compare.end());
				std::cout << "Time Naive Integral(median): " << calcTimes_compare[calcTimes_compare.size() / 2] << " ms" << std::endl;

#if Calc_BoxFiter
				// ボックスフィルタリングしたいなら以下も追加
				// matを1行にreshapeしてcopyto(0)で送る
				cv_cuda_scan = cv_cuda_scan.reshape(0, scanSize * scanSize);
				cv_cuda_scan.col(0).copyTo(NaiveIntegral_Host);
				cudaSafeCall(cudaMemcpy(LAU_Device, &NaiveIntegral_Host[0], scanSize * scanSize * sizeof(float), cudaMemcpyHostToDevice));
#endif
			}
#else
			if (scanSize > 256)
			{
				// 提案手法(TF32コンスタントメモリ使用)
				matMulTcuHorizontalWrapperCmemTF32(A_Device, LA_Device, scanSize, scanSize, scanSize);
				matMulTcuVerticalWrapperCmemTF32(LA_Device, LAU_Device, scanSize, scanSize, scanSize);
			}
			else
			{
				// 提案手法(FP16コンスタントメモリ使用、256以下で実行)
				matMulTcuHorizontalWrapperCmemFP16(A_Device16, LA_Device16, scanSize, scanSize, scanSize);
				matMulTcuVerticalWrapperCmemFP16(LA_Device16, LAU_Device, scanSize, scanSize, scanSize);
			}
#endif	

			cudaSafeCall(cudaGetLastError());

#if Calc_BoxFiter
			// LAUDeviceを使って、積分画像で畳み込みした結果をdestDeviceに得る
			BoxFilterKernelTCUWrapper(LAU_Device, destDevice, destpitch, destpitch, dest.cols, dest.rows, r, scanSize);
#endif

		};
#if !SHOW_CALC_TIME
		f();
#else
		MEASURE_TIME(f, "calc mul");
#endif	
	}

	//copy device to host
	{
		auto f = [&]
		{
			cudaSafeCall(cudaMemcpy(scan.data, LAU_Device, scanSize * scanSize * sizeof(float), cudaMemcpyDeviceToHost));
#if Calc_BoxFiter
			// GPUでdestに結果を入れさせて、destを出力画像として戻す
			cudaSafeCall(cudaMemcpy2D(dest.data, (dest.cols) * sizeof(float), destDevice, destpitch, (dest.cols) * sizeof(float), dest.rows, cudaMemcpyDeviceToHost));
			cudaSafeCall(cudaDeviceSynchronize());
#endif
#if Save_Output
#if SUGGEST
			imwrite("img/ScanBoxTCUr" + to_string(r) + ".png", dest);
#else
			imwrite("img/ScanBoxNaiver" + to_string(r) + ".png", dest);
#endif
#endif
		};
#if !SHOW_CALC_TIME
		f();
#else
		MEASURE_TIME(f, "copy to host");
#endif	
	}

	//deallocate
	if (scanSize > 256)
	{
		cudaSafeCall(cudaFreeAsync(A_Device, stream_0));
		cudaSafeCall(cudaFreeAsync(LA_Device, stream_1));
	}
	else
	{
		cudaSafeCall(cudaFreeAsync(A_Device16, stream_0));
		cudaSafeCall(cudaFreeAsync(LA_Device16, stream_1));
	}
	cudaSafeCall(cudaFreeAsync(LAU_Device, stream_2));

	//cudaSafeCall(cudaFree(destDevice));
	//cudaSafeCall(cudaFree(scanafterDevice));

	cudaStreamDestroy(stream_0);
	cudaStreamDestroy(stream_1);
	cudaStreamDestroy(stream_2);
}
