#include "BilateralCuda_naive.cuh"

using namespace cv;
using namespace cv::cuda;
using namespace std;

#define TILE_X 32
#define TILE_Y 32

//constant memmory
__constant__ float cGaussian[64];

//texture memory
texture<uchar, 2, cudaReadModeElementType> tex;


__host__ void updateGaussian(const int r, const double sd)
{
	float fGaussian[64];
	for (int i = 0; i < 2 * r + 1; i++)
	{
		float x = i - r;
		fGaussian[i] = expf(-(x * x) / (2 * sd * sd));
	}
	// 外で宣言してあるコンスタントメモリに、空間的距離のガウシアンを用意
	cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float) * (2 * r + 1));
}

// 輝度差の方のガウシアン
__device__ inline float gaussian(const float x, const float sigma)
{
	return __expf(-(powf(x, 2)) / (2 * powf(sigma, 2)));
}

__global__ void gpuCalculation_texture(const uchar* src, uchar* dest, const int width, const int height, const int r, const float sigma_r, const float sigma_s)
{
	int idx = __mul24(blockIdx.x, TILE_X) + threadIdx.x;
	int idy = __mul24(blockIdx.y, TILE_Y) + threadIdx.y;

	if ((idx < width) && (idy < height))
	{
		double sum = 0;
		double wsum = 0;
		// 注目画素
		uchar tgt = tex2D(tex, idx + r, idy + r);

		for (int dy = -r; dy <= r; dy++) {
			for (int dx = -r; dx <= r; dx++) {
				// テクスチャ
				uchar ref = tex2D(tex, idx + dx + r, idy + dy + r);

				double w = (cGaussian[dy + r] * cGaussian[dx + r]) * gaussian(tgt - ref, sigma_r);
				sum += w * ref;
				wsum += w;
			}
		}
		dest[(idy)*width + idx] = sum / wsum;
	}
}

__global__ void gpuCalculation_global(const uchar* src, uchar* dest, const int width, const int height, const int r, const float sigma_r, const float sigma_s)
{
	int idx = __mul24(blockIdx.x, TILE_X) + threadIdx.x;
	int idy = __mul24(blockIdx.y, TILE_Y) + threadIdx.y;

	if ((idx < width) && (idy < height))
	{
		double sum = 0;
		double wsum = 0;
		// グローバルメモリの実装
		uchar tgt = src[(idy+r)*(width+2*r)+idx+r];


		for (int dy = -r; dy <= r; dy++) {
			for (int dx = -r; dx <= r; dx++) {
				// グローバル
				uchar ref = src[(idy + dy + r) * (width + 2 * r) + idx + dx + r];

				double w = (cGaussian[dy + r] * cGaussian[dx + r]) * gaussian(tgt - ref, sigma_r);
				sum += w * ref;
				wsum += w;
			}
		}
		dest[(idy)*width + idx] = sum / wsum;
	}
}

vector<double> bilateralFilterCuda_naive_texture(const Mat& src, Mat& dest, const int r, const float sigma_r, const float sigma_s, const int loop)
{
	// GPUの時間計測
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//int gray_size = src.cols * src.rows;
	int gray_size_dest = dest.cols * dest.rows;

	// メモリ確保
	// バンクコンフリクト防止
	size_t pitch;
	uchar* DevSrc = NULL;
	uchar* DevDest;

	// 空間のガウシアンを作成
	updateGaussian(r, sigma_s);

	// pitchとcpy2Dで確保とGPUへのコピーをして、バインドも張る
	cudaMallocPitch(&DevSrc, &pitch, sizeof(uchar) * src.step, src.rows);
	cudaMemcpy2D(DevSrc, pitch, src.ptr(), sizeof(uchar) * src.step, sizeof(uchar) * src.step, src.rows, cudaMemcpyHostToDevice);

	cudaBindTexture2D(0, tex, DevSrc, src.step, src.rows, pitch);
	// 出力用
	cudaMalloc<uchar>(&DevDest, gray_size_dest);

	dim3 block(TILE_X, TILE_Y);
	// 画像全体を覆うようなグリッドサイズ計算
	dim3 grid((dest.cols + block.x - 1) / block.x, (dest.rows + block.y - 1) / block.y);

	// 計測
	int loop_ = loop;
	vector<double> calcTimes(loop);
	while (loop_--) {
		cudaEventRecord(start, 0);
		gpuCalculation_texture<<<grid, block>>>(DevSrc, DevDest, dest.cols, dest.rows, r, sigma_r, sigma_s);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		calcTimes.push_back(time);
	}

	// DtoH
	cudaMemcpy(dest.ptr(), DevDest, gray_size_dest, cudaMemcpyDeviceToHost);

	// 解放
	cudaFree(DevSrc);
	cudaFree(DevDest);
	// バインドの解除
	cudaUnbindTexture(tex);
	cudaDeviceReset();

	return calcTimes;
}


vector<double> bilateralFilterCuda_naive_global(const Mat& src, Mat& dest, const int r, const float sigma_r, const float sigma_s, const int loop)
{
	// GPUの時間計測
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int gray_size = src.cols * src.rows;
	int gray_size_dest = dest.cols * dest.rows;

	size_t pitch;
	uchar* DevSrc = NULL;
	uchar* DevDest;

	updateGaussian(r, sigma_s);

	cudaMalloc<uchar>(&DevSrc, gray_size);
	cudaMemcpy(DevSrc, src.ptr(), gray_size, cudaMemcpyHostToDevice);

	// 出力用
	cudaMalloc<uchar>(&DevDest, gray_size_dest);

	dim3 block(TILE_X, TILE_Y);
	dim3 grid((dest.cols + block.x - 1) / block.x, (dest.rows + block.y - 1) / block.y);

	// 計測
	int loop_ = loop;
	vector<double> calcTimes(loop);
	while (loop_--) {
		cudaEventRecord(start, 0);
		gpuCalculation_global<< <grid, block >> > (DevSrc, DevDest, dest.cols, dest.rows, r, sigma_r, sigma_s);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		calcTimes.push_back(time);
	}
	// DtoH
	cudaMemcpy(dest.ptr(), DevDest, gray_size_dest, cudaMemcpyDeviceToHost);

	// 解放
	cudaFree(DevSrc);
	cudaFree(DevDest);
	// バインドの解除
	cudaDeviceReset();

	return calcTimes;
}