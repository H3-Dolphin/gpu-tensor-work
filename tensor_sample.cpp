#include "tensor_sample.h"
#include "matMul.cuh"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
using namespace std;

#define SHOW_MAT 0
#define SHOW_DIFF 0

void tensorTest()
{
	const int sizeM = 32;
	const int sizeN = 32;
	const int sizeK = 32;
	half* A;
	half* B;
	half* C;
	half* rC;
	// 16Å~16ÇÃFP16ÇÃÉÅÉÇÉäÇämï€
	cudaSafeCall(cudaMallocManaged(reinterpret_cast<void**>(&A), sizeof(half) * sizeM * sizeK));
	cudaSafeCall(cudaMallocManaged(reinterpret_cast<void**>(&B), sizeof(half) * sizeK * sizeN));
	cudaSafeCall(cudaMallocManaged(reinterpret_cast<void**>(&C), sizeof(half) * sizeM * sizeN));
	cudaSafeCall(cudaMallocManaged(reinterpret_cast<void**>(&rC), sizeof(half) * sizeM * sizeN));

#if SHOW_MAT
	auto printMatrix = [](const half* m, const int rN, const int cN, const std::string prefix = "") ->void
	{
		cout << prefix << endl;
		for (int j = 0; j < rN * cN; j++)
		{
			cout << __half2float(m[j]) << ", ";
			if (j % cN == (cN - 1))
				cout << endl;
		}
		cout << endl;
	};
#endif

	//init
	{
		std::random_device rnd;
		std::mt19937 mt(rnd());
		std::normal_distribution<> norm(0.0, 5.0);
		for (int j = 0; j < sizeM; j++)
		{
			for (int i = 0; i < sizeK; i++)
			{
				A[j * sizeK + i] = __float2half(
					norm(mt)
				);
			}
		}
		for (int j = 0; j < sizeK; j++)
		{
			for (int i = 0; i < sizeN; i++)
			{
				B[j * sizeN + i] = __float2half(
					norm(mt)
				);
			}
		}
		for (int i = 0; i < sizeM * sizeN; i++)
		{
			C[i] = __float2half(0.f);
		}
	}

#if SHOW_MAT
	printMatrix(A, sizeM, sizeK, "A");
	printMatrix(B, sizeK, sizeN, "B");
	printMatrix(C, sizeM, sizeN, "C");
#endif

	cudaEvent_t start, stop;
	cudaSafeCall(cudaEventCreate(&start));
	cudaSafeCall(cudaEventCreate(&stop));
	{
		cudaSafeCall(cudaEventRecord(start));
		{
			matMulTcuWrapper(A, B, C, sizeM, sizeN, sizeK);
			cudaSafeCall(cudaGetLastError());
		}
		cudaSafeCall(cudaEventRecord(stop));
		cudaSafeCall(cudaEventSynchronize(stop));
		float milliseconds = 0;
		cudaSafeCall(cudaEventElapsedTime(&milliseconds, start, stop));
		cout << "calcTime w/ TCU: " << milliseconds << "ms" << endl;
	}
	{
		cudaSafeCall(cudaEventRecord(start));
		{
			matMulWrapper(A, B, rC, sizeM, sizeN, sizeK);
			cudaSafeCall(cudaGetLastError());
		}
		cudaSafeCall(cudaEventRecord(stop));
		cudaSafeCall(cudaEventSynchronize(stop));
		float milliseconds = 0;
		cudaSafeCall(cudaEventElapsedTime(&milliseconds, start, stop));
		cout << "calcTime w/o TCU: " << milliseconds << "ms" << endl;
	}

#if SHOW_MAT
	printMatrix(A, sizeM, sizeK, "A");
	printMatrix(B, sizeK, sizeN, "B");
	printMatrix(C, sizeM, sizeN, "C");
	printMatrix(rC, sizeM, sizeN, "rC");
#endif

#if SHOW_DIFF
	auto printMatrixDiff = [](const half* m0, const half* m1, const int rN, const int cN, const std::string prefix = "") ->void
	{
		cout << prefix << endl;
		for (int j = 0; j < rN * cN; j++)
		{
			cout << __half2float(m0[j]) - __half2float(m1[j]) << ", ";
			if (j % cN == (cN - 1))
				cout << endl;
		}
		cout << endl;
	};
	printMatrixDiff(C, rC, sizeM, sizeN, "diff C");
#endif

	cudaSafeCall(cudaFree(A));
	cudaSafeCall(cudaFree(B));
	cudaSafeCall(cudaFree(C));
	cudaSafeCall(cudaFree(rC));
}