// Testcase4: Evaluating variable transformation.

#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

__global__ void cgkernel() {
    int i = threadIdx.x;
    int j = i + 1;

    printf("A (thread %d)\n", i);

    thread_group g = tiled_partition(this_thread_block(), 16);
    
    if (i % 2 == 1) 
    {
        printf("B (thread %d)\n", threadIdx.x * 2);
    }
    else
    {
        printf("B (thread %d)\n", threadIdx.x);
    }
    g.sync();
    
    printf("C (thread %d)\n", i + j);
}

int main() {
    printf("Executing Test Case 4: Variable Transform Test.\n");
    
    cgkernel<<<1, 128>>>();
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    printf("Device finished\n");
    return 0;
}
