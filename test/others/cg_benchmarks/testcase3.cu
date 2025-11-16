// Testcase3: Evaluating synchronization within nested branches.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace cooperative_groups;

__global__ void cgkernel() {
    int i = threadIdx.x;
    
    // Initial thread output
    printf("A (thread %d)\n", i);

    thread_group g = tiled_partition(this_thread_block(), 1);    
    if (i < 8) {
        printf("B (thread %d)\n", i);
        if (i < 4) {
            g.sync();
            printf("C (thread %d)\n", i);
            if (i < 2) {
                g.sync();
                printf("D (thread %d)\n", i);
                if (i < 1) {
                    g.sync();
                    printf("E (thread %d)\n", i);
                }
            }
        } else {
            printf("F (thread %d)\n", i);
            g.sync();
        }
    }
}

int main() {
    printf("Executing Test Case 3: Synchronization inside nested branches.\n");
    
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
