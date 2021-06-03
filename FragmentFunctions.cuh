#ifndef FRAGMENTFUNCTIONS_CUH_INCLUDED
#define FRAGMENTFUNCTIONS_CUH_INCLUDED
#include <mma.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// shared_dim x shared_dim のShared Memoryを用いてfragmentの読み込みを行う
static constexpr std::size_t shared_dim = 16;

template<typename Layout>
__device__
void load_matrix_into_shared_memory(half* const shared_ptr, const half* const global_ptr, const unsigned ldm, const std::size_t mat_rows, const std::size_t mat_cols);

template<>
__device__
void load_matrix_into_shared_memory<nvcuda::wmma::row_major>(half* const shared_ptr, const half* const global_ptr, const unsigned ldm, const std::size_t mat_rows, const std::size_t mat_cols)
{
	// Warp内でのスレッドID
	const auto warp_thread_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;

	// warp内スレッドIDがnのスレッドが
	// Fig.1のGlobalMemoryの位置p_nの8要素をShared Memoryの位置p_nにコピー
	//
	// -      16x16 memory   -- -
	// |   p_0     |   p_1    | ^
	// |   p_2     |   p_3    | |
	// |   p_4     |   p_5    | 
	// |          ...         | 16
	// |   p_26    |   p_27   |
	// |   p_28    |   p_29   | |
	// |   p_30    |   p_31   | V
	// ------------------------ -
	// | <-- 8 --->|<-- 8 --->|
	//          Fig.1

	// 1スレッドがコピーする要素数は8 (16x16の256要素を1warp 32threadで分担)
	// 本当はshared_dim * shared_dim / warpSizeと書きたいけれど、組み込み変数であるwarpSizeが
	// コンパイル時に決まらないようなので8を決め打ち
	// 3×3でやりたいので、1スレあたり3に設定
	//constexpr std::size_t elements_per_thread = 8;
	constexpr std::size_t elements_per_thread = 3;

	// 1行16要素なので2スレッドで1行をコピー
	constexpr std::size_t threads_per_row = shared_dim / elements_per_thread; // 2

	// Fig.1の自分の担当の横長ブロックの先頭要素の位置(r0,c0)を計算
	const auto r0 = warp_thread_id / threads_per_row;
	const auto c0 = (warp_thread_id % threads_per_row) * elements_per_thread;
	// 各横長ブロックは8要素
	half* shared_ptr_head = shared_ptr + elements_per_thread * warp_thread_id;
	const half* global_ptr_head = global_ptr + r0 * ldm + c0;

#pragma unroll
	for (auto i = decltype(elements_per_thread)(0); i < elements_per_thread; i++)
	{
		if ((r0 < mat_rows) && (c0 + i < mat_cols))
		{
			shared_ptr_head[i] = global_ptr_head[i];
		}
		else
		{
			shared_ptr_head[i] = __float2half(.0f);
		}
	}

	__syncthreads();
}

template<>
__device__
void load_matrix_into_shared_memory<nvcuda::wmma::col_major>(half* const shared_ptr, const half* const global_ptr, const unsigned ldm, const std::size_t mat_rows, const std::size_t mat_cols)
{
	// 変数の説明等はload_matrix_into_shared_memory<nvcuda::wmma::row_major>内のコメントを参照
	const auto warp_thread_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;

	// Fig.1 の横長ブロックを縦長ブロックとして考える
	//constexpr std::size_t elements_per_thread = 8;
	// ここも3?
	constexpr std::size_t elements_per_thread = 3;

	constexpr std::size_t threads_per_col = shared_dim / elements_per_thread; // 2

	const auto c0 = warp_thread_id / threads_per_col;
	const auto r0 = (warp_thread_id % threads_per_col) * elements_per_thread;
	half* shared_ptr_head = shared_ptr + elements_per_thread * warp_thread_id;
	const half* global_ptr_head = global_ptr + c0 * ldm + r0;

#pragma unroll
	for (auto i = decltype(elements_per_thread)(0); i < elements_per_thread; i++)
	{
		if ((r0 + i < mat_rows) && (c0 < mat_cols))
		{
			shared_ptr_head[i] = global_ptr_head[i];
		}
		else
		{
			shared_ptr_head[i] = __float2half(.0f);
		}

	}
	__syncthreads();
}


template<typename Use, int m, int n, int k, typename T, typename Layout>
__device__
void load_irregular_matrix(nvcuda::wmma::fragment<Use, m, n, k, T, Layout>& frag, const T* const global_ptr, const unsigned ldm, const std::size_t mat_rows, const std::size_t mat_cols)
{
	__shared__ half tmp_shared_mem[shared_dim * shared_dim];
	// ここでsmemに16×16の値が入るので、この中でうまく3×3として入るようにしたい
	load_matrix_into_shared_memory<Layout>(tmp_shared_mem, global_ptr, ldm, mat_rows, mat_cols);
	nvcuda::wmma::load_matrix_sync(frag, tmp_shared_mem, shared_dim);
}


// row_major専用
__device__
//void store_irregular_matrix(half* const global_ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>& a, const unsigned ldm, const std::size_t mat_rows, const std::size_t mat_cols)
void store_irregular_matrix(float* const global_ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& a, const unsigned ldm, const std::size_t mat_rows, const std::size_t mat_cols)
{
	//__shared__ half tmp_shared_mem[shared_dim * shared_dim];
	__shared__ float tmp_shared_mem[shared_dim * shared_dim];

	// 変数の説明等はload_matrix_into_shared_memory<nvcuda::wmma::row_major>内のコメントを参照
	const auto warp_thread_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;
	nvcuda::wmma::store_matrix_sync(tmp_shared_mem, a, shared_dim, nvcuda::wmma::mem_row_major);

	//constexpr std::size_t num_store_elements = 8;
	constexpr std::size_t num_store_elements = 3;
	constexpr std::size_t threads_per_row = shared_dim / num_store_elements; // 2

	// Fig.1と同じ
	const auto r0 = warp_thread_id / threads_per_row;
	const auto c0 = (warp_thread_id % threads_per_row) * num_store_elements;
	//const half* shared_ptr_head = tmp_shared_mem + num_store_elements * warp_thread_id;
	const float* shared_ptr_head = tmp_shared_mem + num_store_elements * warp_thread_id;

	//half* global_ptr_head = global_ptr + r0 * ldm + c0;
	float* global_ptr_head = global_ptr + r0 * ldm + c0;

	if (!(r0 < mat_rows)) return;
#pragma unroll
	for (auto i = decltype(num_store_elements)(0); i < num_store_elements; i++)
	{
		if (c0 + i < mat_cols)
		{
			global_ptr_head[i]      = shared_ptr_head[i];
			global_ptr_head[i+512]  = shared_ptr_head[i+16];
			global_ptr_head[i+1024] = shared_ptr_head[i+32];
		}
		else
		{
			return;
		}
	}
}
#endif