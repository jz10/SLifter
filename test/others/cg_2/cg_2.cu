#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void cooperative_scan_kernel(int* data, int size) {
    cg::thread_block block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    extern __shared__ int temp[];

    int tid = block.thread_rank();

    // Load input
    int global_tid = blockIdx.x * blockDim.x + tid;
    int val = (global_tid < size) ? data[global_tid] : 0;

    // Warp-level scan
    if (warp.meta_group_rank() == 0) {
        // Intra-warp inclusive scan using shfl
        for (int offset = 1; offset < 32; offset *= 2) {
            int n = warp.shfl_up(val, offset);
            if (warp.thread_rank() >= offset) {
                val += n;
            }
        }
        temp[tid] = val;
    }

    cg::sync(block);

    // Store result
    if (global_tid < size) {
        data[global_tid] = temp[tid];
    }
}