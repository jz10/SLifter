// Testcase0: Evaluating thread_block and basic synchronization functions
// without employing group partitioning. 

#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

__global__ void cgkernel() {
    thread_block g = this_thread_block();
    int i = g.thread_rank();

    printf("Thread %d launched\n", i);
    g.sync();
    printf("Thread %d synchronized\n", i);
    sync(this_thread_block());
    printf("Thread %d synchronized again\n", i);
}

__host__ int main() {
    printf("Executing Test Case: Testing thread_block and basic synchronization.\n");
    cgkernel<<<1, 128>>>();
    cudaDeviceSynchronize();
    return 0;
}
