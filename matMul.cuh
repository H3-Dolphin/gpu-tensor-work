#pragma once
#include <cuda_runtime.h>
#include <mma.h>

// C(MxN) <= A(MxK)xB(KxN)+C(MxN)
__host__ void matMulTcuWrapper(const half* const dMatA, const half* const dMatB, half* dMatC, const int m, const int n, const int k);
__host__ void matMulWrapper(const half* dMatA, const half* dMatB, half* dMatC, const int m, const int n, const int k);

