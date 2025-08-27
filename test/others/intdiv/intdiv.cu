#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 

using DTYPE = int64_t;

__global__ void intdiv(const DTYPE* A, const DTYPE* B, DTYPE* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    C[idx] = A[idx] + B[idx];
}

int main() {
    int n = 1 << 20;
    size_t sz = n * sizeof(DTYPE);
    DTYPE *h_A = (DTYPE*)malloc(sz), *h_B = (DTYPE*)malloc(sz), *h_C = (DTYPE*)malloc(sz);
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    DTYPE *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sz);
    cudaMalloc(&d_B, sz);
    cudaMalloc(&d_C, sz);
    cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    intdiv<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaMemcpy(h_C, d_C, sz, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++)
        printf("%d ", h_C[i]);
    printf("\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
