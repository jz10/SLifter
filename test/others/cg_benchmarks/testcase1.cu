// Testcase1: Evaluating thread_group and basic synchronization functions 
// with tiled partitioning.

#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

__global__ void cgkernel() {
    printf("A (thread %d)\n", threadIdx.x);
    thread_group g = tiled_partition(this_thread_block(), 16);
    printf("B (thread %d)\n", threadIdx.x);
    g.sync();
    printf("C (thread %d)\n", threadIdx.x);
    __syncthreads();
    printf("D (thread %d)\n", threadIdx.x);
}

int main() {
    printf("Executing Test Case 1: Testing thread_group and basic synchronization with tiled partitioning.\n");
    cgkernel<<<1, 128>>>();
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    printf("Device finished successfully.\n");
    return 0;
}
