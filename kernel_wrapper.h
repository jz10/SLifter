#pragma once
#include <cstdint>
#include <type_traits>

extern "C" int const_mem[1024] = {0};

enum : int {
    SR_NTID_X     = 0x08,
    SR_GRID_DIM_X = 0x14,
    SR_CTAID_X    = 0x20,
    SR_TID_X      = 0x2C,
    ARG_BASE      = 0x140
};

template<typename T>
inline void writeArg(int idx, T v)
{
    const int off = ARG_BASE + idx * 8;

    if constexpr (std::is_pointer_v<T> || sizeof(T) == 8) {
        uint64_t raw = std::is_pointer_v<T> ?
                       reinterpret_cast<uint64_t>(v) :
                       *reinterpret_cast<uint64_t*>(&v);

        const_mem[off]     = uint32_t(raw);
        const_mem[off + 1] = uint32_t(raw >> 32);
    } else {
        const_mem[off]     = uint32_t(v);
        const_mem[off + 1] = 0;
    }
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
            func(); 
        }
    }
}