#include "matMul.cuh"
#include <assert.h>
#include <opencv2/core/utility.hpp>

namespace tcu {
	constexpr int WMMA_M = 16;
	constexpr int WMMA_N = 16;
	constexpr int WMMA_K = 16;

	template<class MatType, class MajorType>
	using HalfFrag = nvcuda::wmma::fragment<MatType, WMMA_M, WMMA_N, WMMA_K, half, MajorType>;

	template<class MatType>
	using HalfRowMajorFrag = nvcuda::wmma::fragment<MatType, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>;

	template<class MatType>
	using HalfColMajorFrag = nvcuda::wmma::fragment < MatType, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major>;

	using HalfAccFrag = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>;

	using FloatAccFrag = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
}


// C(MxN) <= A(MxK)xB(KxN)+C(MxN)
__global__ void matMulTcuKernel(const half* const dMatA, const half* const dMatB, half* dMatC, const int m, const int n, const int k)
{
	const auto warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	const auto warpN = (blockIdx.y * blockDim.y + threadIdx.y);

	const auto a_row = warpM * tcu::WMMA_M;
	const auto b_col = warpN * tcu::WMMA_N;
	if (a_row < 0 || a_row >= m || b_col < 0 || b_col >= n)
	{
		return;
	}

	tcu::HalfAccFrag acc_frag;
	nvcuda::wmma::fill_fragment(acc_frag, __float2half(.0f));

	for (auto i = 0; i < k; i += tcu::WMMA_K)
	{
		tcu::HalfRowMajorFrag<nvcuda::wmma::matrix_a> a_frag;
		tcu::HalfRowMajorFrag<nvcuda::wmma::matrix_b> b_frag;
		const half* a_ptr = dMatA + i + a_row * k;
		const half* b_ptr = dMatB + i * n + b_col;
		nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, k);
		nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, n);

		nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	}

	const auto c_row = warpM * tcu::WMMA_M;
	const auto c_col = warpN * tcu::WMMA_N;
	if ((c_row < m) && (c_col < n))
	{
		nvcuda::wmma::store_matrix_sync(dMatC + c_col + c_row * n, acc_frag, n, nvcuda::wmma::mem_row_major);
	}
}

//test matrix mul kernel
__global__ void matMulKernel(const half* dMatA, const half* dMatB, half* dMatC, const int m, const int n, const int k)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idy < 0 || idx < 0 || idy >= m || idx >= n)
	{
		return;
	}

	half ret = half(0.f);
	for (int i2 = 0; i2 < k; i2++)
	{
		ret = __hfma(dMatA[idy * k + i2], dMatB[i2 * n + idx], ret);
	}
	dMatC[idy * n + idx] = ret;
}

__host__ void matMulTcuWrapper(const half* const dMatA, const half* const dMatB, half* dMatC, const int m, const int n, const int k)
{
	assert(m % tcu::WMMA_M == 0 && n % tcu::WMMA_N == 0 && k % tcu::WMMA_K == 0);

	const int warpSize = 32;
	const dim3 grid(m / tcu::WMMA_M, n / tcu::WMMA_N);
	const dim3 block(warpSize);
	matMulTcuKernel << <grid, block >> > (dMatA, dMatB, dMatC, m, n, k);
}

__host__ void matMulWrapper(const half* dMatA, const half* dMatB, half* dMatC, const int m, const int n, const int k)
{
	const int gridSize = 8;
	const dim3 grid(cv::divUp(n, gridSize), cv::divUp(m, gridSize));
	const dim3 block(cv::divUp(n, grid.x), cv::divUp(m, grid.y));
	matMulKernel << <grid, block >> > (dMatA, dMatB, dMatC, m, n, k);
}

// show fragment contents
template<class MatType, class MajorType>
__device__ inline void print_fragment(const tcu::HalfFrag<MatType, MajorType>& frag, const char* name)
{
	if ((threadIdx.x & 0x1f) == 0)
	{
		if (name[0] != '\0')
		{
			printf("%s = \n", name);
		}
	}

	for (unsigned i = 0; i < warpSize; i++)
	{
		if (i == (threadIdx.x & 0x1f))
		{
			for (unsigned j = 0; j < frag.num_elements; j++)
			{
				const auto v = __half2float(frag.x[j]);
				if (v == 0.0f)
				{
					printf(" %f ", 0.0f);
				}
				else if (v > 0)
				{
					printf(" %f ", v);
				}
				else
				{
					printf("%f ", v);
				}
			}
			printf("\n");
		}
		__syncthreads();
	}
}
