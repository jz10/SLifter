#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void cooperative_groups_basic_kernel(int* data, int size)
{
    cg::thread_block block = cg::this_thread_block();
    int tid = block.thread_rank();
    int bid = block.group_index().x;

    if (tid < size)
    {
        data[tid] = tid * 2;
    }

    cg::sync(block);

    auto tile16 = cg::tiled_partition<16>(block);

    if (tile16.thread_rank() == 0)
    {
        data[bid * 16] += 1000;
    }

    cg::sync(tile16);
}