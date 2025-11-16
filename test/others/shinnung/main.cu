#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void tile4_reduce_kernel(const float* A, float* B, int N) {

    // thread block tile size = 4
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(block);

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = 0;

    if (gid < N) {
        x = A[gid];

        // tile.size() == 4, tile.thread_rank() 0~3
        for (int offset = tile4.size() / 2; offset > 0; offset /= 2) {
            float y = tile4.shfl_down(x, offset);
            if (tile4.thread_rank() < offset) {
                x += y;
            }
        }
    }

    // useless sync -> can we find this?
    // tile4.sync();

    if (gid < N && tile4.thread_rank() == 0) {
        int tile_id = gid / 4;
        B[tile_id] = x;
    }
}
