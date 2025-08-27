#pragma once
#include <cstdint>
#include <type_traits>
#include <bit>
#include <iostream>

extern "C" int const_mem[1024] = {0};

enum : int {
    SR_NTID_X     = 0x0,
    SR_GRID_DIM_X = 0x14,
    SR_CTAID_X    = 0x20,
    SR_TID_X      = 0x2C,
    SR_LANE_ID    = 0x38,
    ARG_BASE      = 0x160
};

static int off = ARG_BASE;


static constexpr size_t align_up(size_t x, size_t a)
{
    return (x + a - 1) & ~(a - 1);
}

template<typename T>
inline void writeArg(int idx, T v)
{

    if constexpr (std::is_pointer_v<T>) {
        uint64_t raw = reinterpret_cast<uint64_t>(v);
        int aligned_off = align_up(off, sizeof(T));
        const_mem[aligned_off]     = static_cast<uint32_t>(raw);
        const_mem[aligned_off + 1] = static_cast<uint32_t>(raw >> 32);
    } else if constexpr (sizeof(T) == 8) {
        uint64_t raw = std::bit_cast<uint64_t>(v);
        int aligned_off = align_up(off, sizeof(T));
        const_mem[aligned_off]     = static_cast<uint32_t>(raw);
        const_mem[aligned_off + 1] = static_cast<uint32_t>(raw >> 32);
    } else {
        const_mem[off]     = std::bit_cast<uint32_t>(v);
    }

    std::cout<<"Arg "<<off<<", type="<<typeid(T).name()<<", size="<<sizeof(T)<<std::endl;

    off += sizeof(T);
}

template<typename Func, typename... Args>
void launchKernel(Func func, int gridDim, int blockDim, Args... args)
{
    int argIdx = 0;
    (writeArg(argIdx++, args), ...);

    for (int cta = 0; cta < gridDim; ++cta) {
        for (int tid = 0; tid < blockDim; ++tid) {
            const_mem[SR_NTID_X]     = blockDim;
            const_mem[SR_GRID_DIM_X] = gridDim;
            const_mem[SR_CTAID_X]    = cta;
            const_mem[SR_TID_X]      = tid;
            const_mem[SR_LANE_ID]    = (tid & 31);
            func(); 
        }
    }
}
