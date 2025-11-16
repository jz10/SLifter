

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void badMemAccessPattern(float* input, float* output, int n) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    // Strided access within warp - destroys coalescing
    int stride = warp.thread_rank() * 4;  // 4-element stride
    if (block.thread_rank() + stride < n) {
        float val = input[block.thread_rank() + stride];  // POOR COALESCING
        output[block.thread_rank()] = val * 2.0f;
    }
    
    warp.sync();
    
   int idx = block.thread_rank() * 2;  // Even worse stride
   if (idx < n) {
       output[idx] = input[idx] + 1.0f;
   }
}