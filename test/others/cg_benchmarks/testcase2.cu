// Testcase2: Evaluating synchronization within a branch

#include <stdio.h>
#include <cuda_runtime.h>
#include "cooperative_groups.h"

using namespace cooperative_groups;

__global__ void cgkernel() {
    int i = threadIdx.x;

    // Initial thread output
    printf("A (thread %d)\n", i);

    thread_group g = tiled_partition(this_thread_block(), 16);

    int tile_id = i / g.size();
    
    if (tile_id % 2 == 0) {
        printf("B (thread %d)\n", i);
        g.sync();
        printf("C (thread %d)\n", i);
    } else {
        printf("D (thread %d)\n", i);
    }

    if (i < 4) {
        printf("E (thread %d)\n", i);
    }
}

int main() {
    printf("Executing Test Case 2: Synchronization inside a branch without external variables.\n");
    
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
