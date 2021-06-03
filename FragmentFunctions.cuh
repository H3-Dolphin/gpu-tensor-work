#ifndef FRAGMENTFUNCTIONS_CUH_INCLUDED
#define FRAGMENTFUNCTIONS_CUH_INCLUDED
#include <mma.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// shared_dim x shared_dim ��Shared Memory��p����fragment�̓ǂݍ��݂��s��
static constexpr std::size_t shared_dim = 16;

template<typename Layout>
__device__
void load_matrix_into_shared_memory(half* const shared_ptr, const half* const global_ptr, const unsigned ldm, const std::size_t mat_rows, const std::size_t mat_cols);

template<>
__device__
void load_matrix_into_shared_memory<nvcuda::wmma::row_major>(half* const shared_ptr, const half* const global_ptr, const unsigned ldm, const std::size_t mat_rows, const std::size_t mat_cols)
{
	// Warp���ł̃X���b�hID
	const auto warp_thread_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;

	// warp���X���b�hID��n�̃X���b�h��
	// Fig.1��GlobalMemory�̈ʒup_n��8�v�f��Shared Memory�̈ʒup_n�ɃR�s�[
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

	// 1�X���b�h���R�s�[����v�f����8 (16x16��256�v�f��1warp 32thread�ŕ��S)
	// �{����shared_dim * shared_dim / warpSize�Ə�����������ǁA�g�ݍ��ݕϐ��ł���warpSize��
	// �R���p�C�����Ɍ��܂�Ȃ��悤�Ȃ̂�8�����ߑł�
	// 3�~3�ł�肽���̂ŁA1�X��������3�ɐݒ�
	//constexpr std::size_t elements_per_thread = 8;
	constexpr std::size_t elements_per_thread = 3;

	// 1�s16�v�f�Ȃ̂�2�X���b�h��1�s���R�s�[
	constexpr std::size_t threads_per_row = shared_dim / elements_per_thread; // 2

	// Fig.1�̎����̒S���̉����u���b�N�̐擪�v�f�̈ʒu(r0,c0)���v�Z
	const auto r0 = warp_thread_id / threads_per_row;
	const auto c0 = (warp_thread_id % threads_per_row) * elements_per_thread;
	// �e�����u���b�N��8�v�f
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
	// �ϐ��̐�������load_matrix_into_shared_memory<nvcuda::wmma::row_major>���̃R�����g���Q��
	const auto warp_thread_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;

	// Fig.1 �̉����u���b�N���c���u���b�N�Ƃ��čl����
	//constexpr std::size_t elements_per_thread = 8;
	// ������3?
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
	// ������smem��16�~16�̒l������̂ŁA���̒��ł��܂�3�~3�Ƃ��ē���悤�ɂ�����
	load_matrix_into_shared_memory<Layout>(tmp_shared_mem, global_ptr, ldm, mat_rows, mat_cols);
	nvcuda::wmma::load_matrix_sync(frag, tmp_shared_mem, shared_dim);
}


// row_major��p
__device__
//void store_irregular_matrix(half* const global_ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>& a, const unsigned ldm, const std::size_t mat_rows, const std::size_t mat_cols)
void store_irregular_matrix(float* const global_ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& a, const unsigned ldm, const std::size_t mat_rows, const std::size_t mat_cols)
{
	//__shared__ half tmp_shared_mem[shared_dim * shared_dim];
	__shared__ float tmp_shared_mem[shared_dim * shared_dim];

	// �ϐ��̐�������load_matrix_into_shared_memory<nvcuda::wmma::row_major>���̃R�����g���Q��
	const auto warp_thread_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;
	nvcuda::wmma::store_matrix_sync(tmp_shared_mem, a, shared_dim, nvcuda::wmma::mem_row_major);

	//constexpr std::size_t num_store_elements = 8;
	constexpr std::size_t num_store_elements = 3;
	constexpr std::size_t threads_per_row = shared_dim / num_store_elements; // 2

	// Fig.1�Ɠ���
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