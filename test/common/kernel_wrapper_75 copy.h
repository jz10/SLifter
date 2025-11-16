// host_runtime.hpp
// Runtime support for lifted CUDA SASS: const_mem, shared_mem, local_mem,
// launchKernel, warp_shfl, vote_any, syncthreads.
//
// Assumes your lifted IR has declarations like:
//
//   @"const_mem"   = external thread_local global [5 x [4096 x i8]]
//   @"shared_mem"  = external thread_local global [49152 x i32]
//   @"local_mem"   = external thread_local global [32768 x i8]
//   declare void @"syncthreads"()
//
// NOTE: This defines shared_mem/local_mem exactly as in the IR so the linker
//       is happy. shared_mem is thread-local here; to get true per-block
//       shared memory semantics you’ll eventually want to change the IR
//       (remove thread_local from @shared_mem) and adjust this runtime.
//
// Build with the lifted kernel module so these symbols get resolved.

#pragma once
#include <cstdint>
#include <type_traits>
#include <bit>
#include <iostream>
#include <cstring>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <array>
#include <algorithm>
#include <typeinfo>

constexpr int WARP_SIZE = 32;

// -----------------------------------------------------------------------------
// Global memory regions matching lifted IR symbols
// -----------------------------------------------------------------------------

// const_mem as seen by the lifted kernels: [5 x [4096 x i8]]
// const_mem[0] corresponds to c[0], which we use for parameters + "special regs".
extern "C" alignas(8) thread_local uint8_t const_mem[5][4096] = {0};

extern "C" alignas(8) uint8_t shared_mem[32768] = {0};

// local_mem as seen by the lifted kernels: [32768 x i8]
// Matches: @"local_mem" = external thread_local global [32768 x i8]
extern "C" alignas(8) thread_local uint8_t local_mem[32768] = {0};

// Simple helper to get base of c[0]
inline uint8_t* cmem0() {
    return &const_mem[0][0];
}

// -----------------------------------------------------------------------------
// Special-register layout in c[0]
// -----------------------------------------------------------------------------

enum : uint32_t {
    SR_NTID_X     = 0x0,
    SR_GRID_DIM_X = 0xC,
    SR_CTAID_X    = 0x20,
    SR_TID_X      = 0x2C,
    SR_LANE_ID    = 0x38,
    ARG_BASE      = 0x160
};

static uint32_t off = ARG_BASE;

static constexpr uint32_t align_up(uint32_t x, uint32_t a) {
    return (x + (a - 1u)) & ~(a - 1u);
}

// -----------------------------------------------------------------------------
// Argument marshalling into c[0][ARG_BASE...]
// -----------------------------------------------------------------------------

template<typename T>
inline void writeArg(T v)
{
    uint32_t aligned_off = align_up(off, static_cast<uint32_t>(sizeof(T)));
    uint8_t* base = cmem0();

    if constexpr (std::is_pointer_v<T>) {
        uint64_t raw = reinterpret_cast<uint64_t>(v);
        std::memcpy(base + aligned_off, &raw, sizeof(raw));
        std::cout << " ptr=0x" << std::hex << raw << std::dec;
    } else if constexpr (sizeof(T) == 8) {
        uint64_t raw = std::bit_cast<uint64_t>(v);
        std::memcpy(base + aligned_off, &raw, sizeof(raw));
        std::cout << " u64=0x" << std::hex << raw << std::dec;
    } else if constexpr (sizeof(T) == 4) {
        uint32_t raw = std::bit_cast<uint32_t>(v);
        std::memcpy(base + aligned_off, &raw, sizeof(raw));
        std::cout << " u32=0x" << std::hex << raw << std::dec;
    } else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Unexpected arg size");
    }

    std::cout << " Arg byte_off=" << aligned_off << ", type=" << typeid(T).name()
              << ", size=" << sizeof(T) << std::endl;

    off = aligned_off + sizeof(T);
}

// -----------------------------------------------------------------------------
// Per-thread context (used by warp primitives and syncthreads)
// -----------------------------------------------------------------------------

struct ThreadCtx {
    int cta         = 0;
    int tid         = 0;
    int lane        = 0;
    int warpInBlock = 0;
    int warpGlobal  = 0;
};

thread_local ThreadCtx g_threadCtx;

// -----------------------------------------------------------------------------
// Per-warp contexts for shuffles and votes
// -----------------------------------------------------------------------------

struct WarpShuffleCtx {
    std::mutex m;
    std::condition_variable cv;
    int   arrived    = 0;
    int   generation = 0;
    float vals[WARP_SIZE]{};
    float results[WARP_SIZE]{};
};

struct WarpVoteCtx {
    std::mutex m;
    std::condition_variable cv;
    int  arrived    = 0;
    int  generation = 0;
    bool any        = false;
    bool result     = false;
};

// -----------------------------------------------------------------------------
// Per-block context for syncthreads
// -----------------------------------------------------------------------------

struct BlockSyncCtx {
    std::mutex m;
    std::condition_variable cv;
    int arrived    = 0;
    int generation = 0;
};

// Global execution config visible to primitives
static WarpShuffleCtx* g_warpCtx       = nullptr;
static WarpVoteCtx*    g_warpVoteCtx   = nullptr;
static BlockSyncCtx*   g_blockSyncCtx  = nullptr;
static int             g_blockDim      = 0;
static int             g_gridDim       = 0;
static int             g_warpsPerBlock = 0;

// Snapshot of const_mem after argument marshalling
static std::array<uint8_t, sizeof(const_mem)> g_constMemTemplate{};

// -----------------------------------------------------------------------------
// warp_shfl_down_f32 implementation
//
// Matches IR decl:
//   declare float @"warp_shfl_down_f32"(i32 %mask, float %val, i32 %offset, i32 %clamp)
//
// We interpret clamp as the SASS clamp (max lane index, 0..31). Effective width = clamp+1.
// -----------------------------------------------------------------------------

extern "C" float warp_shfl_down_f32(uint32_t mask,
                                    float    val,
                                    int32_t  offset,
                                    int32_t  clamp)
{
    (void)mask; // ignoring mask for now (assume all lanes participate)

    const int lane        = g_threadCtx.lane;
    const int warpInBlock = g_threadCtx.warpInBlock;
    const int warpGlobal  = g_threadCtx.warpGlobal;

    // Handle partial last warp in a block
    const int warpStartTid = warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 0) {
        return val;
    }

    const int effectiveWidth = std::min(clamp + 1, lanesInWarp);

    WarpShuffleCtx& ctx = g_warpCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;

    ctx.vals[lane] = val;
    ctx.arrived++;

    if (ctx.arrived == lanesInWarp) {
        // Last lane performs the shuffle for the warp
        for (int l = 0; l < lanesInWarp; ++l) {
            float out;
            if (l < effectiveWidth) {
                int srcLane = l + offset;
                if (srcLane < effectiveWidth && srcLane < lanesInWarp) {
                    out = ctx.vals[srcLane];
                } else {
                    out = ctx.vals[l];
                }
            } else {
                out = ctx.vals[l];
            }
            ctx.results[l] = out;
        }

        ctx.arrived = 0;
        ++ctx.generation;
        lock.unlock();
        ctx.cv.notify_all();
    } else {
        ctx.cv.wait(lock, [&]{ return ctx.generation != myGen; });
    }

    return ctx.results[lane];
}

// -----------------------------------------------------------------------------
// vote_any implementation
//
// Matches IR decl:
//   declare i1 @"vote_any"(i32 %mask, i1 %pred)
// -----------------------------------------------------------------------------

extern "C" bool vote_any(uint32_t mask, bool pred)
{
    (void)mask; // ignoring lane mask for now

    const int warpGlobal  = g_threadCtx.warpGlobal;
    const int warpInBlock = g_threadCtx.warpInBlock;

    const int warpStartTid = warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 0) {
        return pred;
    }

    WarpVoteCtx& ctx = g_warpVoteCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;

    if (pred) {
        ctx.any = true;
    }
    ctx.arrived++;

    if (ctx.arrived == lanesInWarp) {
        ctx.result   = ctx.any;
        ctx.any      = false;
        ctx.arrived  = 0;
        ++ctx.generation;

        lock.unlock();
        ctx.cv.notify_all();
    } else {
        ctx.cv.wait(lock, [&]{ return ctx.generation != myGen; });
    }

    return ctx.result;
}

// -----------------------------------------------------------------------------
// syncthreads implementation
//
// Matches IR decl:
//   declare void @"syncthreads"()
//
// This is a per-block barrier: all threads in a block (CTA) must reach it
// before any are allowed to continue.
// -----------------------------------------------------------------------------

extern "C" void syncthreads()
{
    const int cta = g_threadCtx.cta;

    BlockSyncCtx& ctx = g_blockSyncCtx[cta];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;

    ctx.arrived++;

    if (ctx.arrived == g_blockDim) {
        // Last thread in block arrives; release everyone
        ctx.arrived = 0;
        ++ctx.generation;
        lock.unlock();
        ctx.cv.notify_all();
    } else {
        ctx.cv.wait(lock, [&]{ return ctx.generation != myGen; });
    }
}

// -----------------------------------------------------------------------------
// Kernel launcher with real OS-thread concurrency
//
// Interface is:
//
//   template<typename Func, typename... Args>
//   void launchKernel(Func func, int gridDim, int blockDim, Args... args)
// -----------------------------------------------------------------------------

template<typename Func, typename... Args>
void launchKernel(Func func, int gridDim, int blockDim, Args... args)
{
    off = ARG_BASE;

    // Marshal arguments into main thread's const_mem (c[0])
    (writeArg(args), ...);

    // Snapshot the entire const_mem (all 5 banks) into a template
    std::memcpy(g_constMemTemplate.data(), const_mem, sizeof(const_mem));

    // Global config
    g_blockDim      = blockDim;
    g_gridDim       = gridDim;
    g_warpsPerBlock = (blockDim + WARP_SIZE - 1) / WARP_SIZE;
    const int totalWarps   = g_warpsPerBlock * gridDim;
    const int totalThreads = gridDim * blockDim;

    // Allocate per-warp contexts
    std::vector<WarpShuffleCtx> warpCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpVoteCtx>    voteCtx(static_cast<size_t>(totalWarps));
    g_warpCtx     = warpCtx.data();
    g_warpVoteCtx = voteCtx.data();

    // Allocate per-block barriers (for syncthreads)
    std::vector<BlockSyncCtx> blockCtx(static_cast<size_t>(gridDim));
    g_blockSyncCtx = blockCtx.data();

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(totalThreads));

    for (int cta = 0; cta < gridDim; ++cta) {
        for (int tid = 0; tid < blockDim; ++tid) {
            threads.emplace_back([=]() {
                // Initialize this thread's TLS const_mem from the template
                std::memcpy(const_mem, g_constMemTemplate.data(), sizeof(const_mem));

                // NOTE: shared_mem and local_mem are thread-local and start zeroed.
                // If you want them zeroed per kernel launch, this is fine.
                // If you need them uninitialized, remove these memset calls.
                std::memset(shared_mem, 0, sizeof(shared_mem));
                std::memset(local_mem,  0, sizeof(local_mem));

                // Set special registers in c[0]
                uint8_t* c0 = cmem0();
                *reinterpret_cast<uint32_t*>(c0 + SR_NTID_X)     = static_cast<uint32_t>(blockDim);
                *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_X) = static_cast<uint32_t>(gridDim);
                *reinterpret_cast<uint32_t*>(c0 + SR_CTAID_X)    = static_cast<uint32_t>(cta);
                *reinterpret_cast<uint32_t*>(c0 + SR_TID_X)      = static_cast<uint32_t>(tid);
                *reinterpret_cast<uint32_t*>(c0 + SR_LANE_ID)    = static_cast<uint32_t>(tid & (WARP_SIZE - 1));

                // Fill per-thread context
                g_threadCtx.cta         = cta;
                g_threadCtx.tid         = tid;
                g_threadCtx.lane        = tid & (WARP_SIZE - 1);
                g_threadCtx.warpInBlock = tid / WARP_SIZE;
                g_threadCtx.warpGlobal  = cta * g_warpsPerBlock + g_threadCtx.warpInBlock;

                // Run the lifted kernel for this logical thread
                func();
            });
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    g_warpCtx       = nullptr;
    g_warpVoteCtx   = nullptr;
    g_blockSyncCtx  = nullptr;
}
