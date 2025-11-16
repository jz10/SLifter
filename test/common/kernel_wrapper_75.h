#pragma once

// -----------------------------------------------------------------------------
// Runtime support for lifted CUDA SASS (SM75)
// - const_mem  : per-thread "constant memory" (c[0]..c[4])
// - shared_mem : per-block shared memory (single segment, reused per CTA)
// - local_mem  : per-thread local memory
// - warp_shfl / vote_any / syncthreads
// - launchKernel: runs a lifted kernel on CPU threads
// -----------------------------------------------------------------------------

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
//
// IR expects:
//
//   @"const_mem"  = external thread_local global [5 x [4096 x i8]]
//   @"shared_mem" = external global              [49152 x i32]
//   @"local_mem"  = external thread_local global [32768 x i8]
//
// We model:
//
//   - const_mem: thread_local (per CUDA thread)
//   - shared_mem: one global segment reused per CTA (we run CTAs sequentially)
//   - local_mem: thread_local (per CUDA thread)
// -----------------------------------------------------------------------------

// const_mem[0] corresponds to c[0], used for params and "special regs".
extern "C" alignas(8) thread_local uint8_t const_mem[5][4096] = {0};

// ONE shared segment reused for each CTA. We run CTAs sequentially to give
// per-block semantics even though IR has a single global symbol.
extern "C" alignas(16) int32_t shared_mem[49152] = {0};

// per-thread "local" memory
extern "C" alignas(16) thread_local uint8_t local_mem[32768] = {0};

// Helper: base of c[0]
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
// Per-thread context (for shuffles, votes, syncthreads)
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
    // Raw 32-bit payload; we bit_cast between int32/float and this.
    uint32_t vals[WARP_SIZE]{};
    uint32_t results[WARP_SIZE]{};
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
// Shuffle helpers
// -----------------------------------------------------------------------------

// Interpret clamp as used by the lifted IR:
//  - clamp == 0 comes from SASS "RZ" and means "full width" (whole warp).
//  - otherwise: width = min(clamp + 1, lanesInWarp).
inline int shfl_effective_width(int clamp, int lanesInWarp) {
    if (clamp == 0)
        return lanesInWarp;
    int width = clamp + 1;
    if (width > lanesInWarp)
        width = lanesInWarp;
    return width;
}

template<typename T>
T warp_shfl_down(uint32_t mask,
                 T       val,
                 int32_t offset,
                 int32_t clamp)
{
    (void)mask; // lane mask ignored for now (assume all participate)
    static_assert(sizeof(T) == 4, "warp_shfl_down only supports 32-bit types");

    const int lane        = g_threadCtx.lane;
    const int warpInBlock = g_threadCtx.warpInBlock;
    const int warpGlobal  = g_threadCtx.warpGlobal;

    const int warpStartTid = warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 0)
        return val;

    const int width = shfl_effective_width(clamp, lanesInWarp);

    WarpShuffleCtx& ctx = g_warpCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;

    ctx.vals[lane] = std::bit_cast<uint32_t>(val);
    ctx.arrived++;

    if (ctx.arrived == lanesInWarp) {
        // Last lane performs the shuffle for the warp
        for (int l = 0; l < lanesInWarp; ++l) {
            uint32_t outRaw;
            if (l < width) {
                int srcLane = l + offset;
                if (srcLane < width && srcLane < lanesInWarp) {
                    outRaw = ctx.vals[srcLane];
                } else {
                    outRaw = ctx.vals[l];
                }
            } else {
                outRaw = ctx.vals[l];
            }
            ctx.results[l] = outRaw;
        }

        ctx.arrived = 0;
        ++ctx.generation;
        lock.unlock();
        ctx.cv.notify_all();
    } else {
        ctx.cv.wait(lock, [&]{ return ctx.generation != myGen; });
    }

    return std::bit_cast<T>(ctx.results[lane]);
}

template<typename T>
T warp_shfl_up(uint32_t mask,
               T       val,
               int32_t offset,
               int32_t clamp)
{
    (void)mask; // lane mask ignored for now
    static_assert(sizeof(T) == 4, "warp_shfl_up only supports 32-bit types");

    const int lane        = g_threadCtx.lane;
    const int warpInBlock = g_threadCtx.warpInBlock;
    const int warpGlobal  = g_threadCtx.warpGlobal;

    const int warpStartTid = warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 0)
        return val;

    const int width = shfl_effective_width(clamp, lanesInWarp);

    WarpShuffleCtx& ctx = g_warpCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;

    ctx.vals[lane] = std::bit_cast<uint32_t>(val);
    ctx.arrived++;

    if (ctx.arrived == lanesInWarp) {
        for (int l = 0; l < lanesInWarp; ++l) {
            uint32_t outRaw;
            if (l < width) {
                int srcLane = l - offset;
                if (srcLane >= 0 && srcLane < width && srcLane < lanesInWarp) {
                    outRaw = ctx.vals[srcLane];
                } else {
                    outRaw = ctx.vals[l];
                }
            } else {
                outRaw = ctx.vals[l];
            }
            ctx.results[l] = outRaw;
        }

        ctx.arrived = 0;
        ++ctx.generation;
        lock.unlock();
        ctx.cv.notify_all();
    } else {
        ctx.cv.wait(lock, [&]{ return ctx.generation != myGen; });
    }

    return std::bit_cast<T>(ctx.results[lane]);
}

// C wrappers matching the lifted IR declarations

extern "C" int32_t warp_shfl_down_i32(uint32_t mask,
                                      int32_t  val,
                                      int32_t  offset,
                                      int32_t  clamp)
{
    return warp_shfl_down<int32_t>(mask, val, offset, clamp);
}

extern "C" float warp_shfl_down_f32(uint32_t mask,
                                    float    val,
                                    int32_t  offset,
                                    int32_t  clamp)
{
    return warp_shfl_down<float>(mask, val, offset, clamp);
}

extern "C" int32_t warp_shfl_up_i32(uint32_t mask,
                                    int32_t  val,
                                    int32_t  offset,
                                    int32_t  clamp)
{
    return warp_shfl_up<int32_t>(mask, val, offset, clamp);
}

extern "C" float warp_shfl_up_f32(uint32_t mask,
                                  float    val,
                                  int32_t  offset,
                                  int32_t  clamp)
{
    return warp_shfl_up<float>(mask, val, offset, clamp);
}

// -----------------------------------------------------------------------------
// vote_any
// Matches IR:
//   declare i1 @"vote_any"(i32 %mask, i1 %pred)
// -----------------------------------------------------------------------------

extern "C" bool vote_any(uint32_t mask, bool pred)
{
    (void)mask;

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
// syncthreads
// Matches IR:
//   declare void @"syncthreads"()
// -----------------------------------------------------------------------------

extern "C" void syncthreads()
{
    const int cta = g_threadCtx.cta;

    BlockSyncCtx& ctx = g_blockSyncCtx[cta];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;

    ctx.arrived++;

    if (ctx.arrived == g_blockDim) {
        ctx.arrived = 0;
        ++ctx.generation;
        lock.unlock();
        ctx.cv.notify_all();
    } else {
        ctx.cv.wait(lock, [&]{ return ctx.generation != myGen; });
    }
}

// -----------------------------------------------------------------------------
// Kernel launcher
//
// Interface:
//   template<typename Func, typename... Args>
//   void launchKernel(Func func, int gridDim, int blockDim, Args... args)
//
// Important: we give *per-block* shared memory semantics by running CTAs
// sequentially and zeroing shared_mem once per CTA, not per thread.
// -----------------------------------------------------------------------------

template<typename Func, typename... Args>
void launchKernel(Func func, int gridDim, int blockDim, Args... args)
{
    off = ARG_BASE;

    // Marshal arguments into this (main) thread's const_mem (c[0])
    (writeArg(args), ...);

    // Snapshot entire const_mem (all 5 banks) as a template
    std::memcpy(g_constMemTemplate.data(), const_mem, sizeof(const_mem));

    // Global config
    g_blockDim      = blockDim;
    g_gridDim       = gridDim;
    g_warpsPerBlock = (blockDim + WARP_SIZE - 1) / WARP_SIZE;
    const int totalWarps = g_warpsPerBlock * gridDim;

    // Per-warp contexts (enough for all warps in the grid)
    std::vector<WarpShuffleCtx> warpCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpVoteCtx>    voteCtx(static_cast<size_t>(totalWarps));
    g_warpCtx     = warpCtx.data();
    g_warpVoteCtx = voteCtx.data();

    // Per-block barrier contexts (one per CTA)
    std::vector<BlockSyncCtx> blockCtx(static_cast<size_t>(gridDim));
    g_blockSyncCtx = blockCtx.data();

    // Run CTAs sequentially to give each one a fresh shared_mem
    for (int cta = 0; cta < gridDim; ++cta) {
        // Per-CTA shared memory init (NOT per thread)
        std::memset(shared_mem, 0, sizeof(shared_mem));

        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(blockDim));

        for (int tid = 0; tid < blockDim; ++tid) {
            threads.emplace_back([=]() {
                // Initialize this thread's TLS const_mem from template
                std::memcpy(const_mem, g_constMemTemplate.data(), sizeof(const_mem));

                // local_mem is thread-local; zero if you want deterministic state
                std::memset(local_mem, 0, sizeof(local_mem));

                // Set special registers in c[0]
                uint8_t* c0 = cmem0();
                *reinterpret_cast<uint32_t*>(c0 + SR_NTID_X)     = static_cast<uint32_t>(blockDim);
                *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_X) = static_cast<uint32_t>(gridDim);
                *reinterpret_cast<uint32_t*>(c0 + SR_CTAID_X)    = static_cast<uint32_t>(cta);
                *reinterpret_cast<uint32_t*>(c0 + SR_TID_X)      = static_cast<uint32_t>(tid);
                *reinterpret_cast<uint32_t*>(c0 + SR_LANE_ID)    = static_cast<uint32_t>(tid & (WARP_SIZE - 1));

                // Per-thread context for warp primitives & syncthreads
                g_threadCtx.cta         = cta;
                g_threadCtx.tid         = tid;
                g_threadCtx.lane        = tid & (WARP_SIZE - 1);
                g_threadCtx.warpInBlock = tid / WARP_SIZE;
                g_threadCtx.warpGlobal  = cta * g_warpsPerBlock + g_threadCtx.warpInBlock;

                // Run the lifted kernel for this logical CUDA thread
                func();
            });
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    g_warpCtx       = nullptr;
    g_warpVoteCtx   = nullptr;
    g_blockSyncCtx  = nullptr;
}
