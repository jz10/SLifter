#include <cuda_runtime.h>

// Kernel: C[i] = A[i] + B[i]
// Matches symbol: _Z9vectorAddPKfS0_Pfi
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

