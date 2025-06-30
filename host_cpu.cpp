#include <cstdint>
#include <iostream>
#include <cmath>

extern "C" void _Z9vectorAddPKfS0_Pfi(
    uint64_t A, uint64_t B,
    uint64_t C, uint64_t N);

extern int g_thread_idx;
extern int g_block_dim;
extern int g_block_idx;
extern int g_lane_id;
extern int g_warp_id;

int const_mem[1024];


enum : int {
    SR_NTID_X     = 0x08,
    SR_GRID_DIM_X = 0x14,
    SR_CTAID_X    = 0x20,
    SR_TID_X      = 0x2C
};

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim  = (N + blockDim - 1) / blockDim;

    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];

    for(int i = 0; i < N; ++i) {
        A[i] = float(i);
        B[i] = float(i)*2;
    }

    for(int b = 0; b < gridDim; ++b) {
        for(int t = 0; t < blockDim; ++t) {
            // g_block_idx  = b;
            // g_block_dim  = blockDim;
            // g_thread_idx = t;
            // g_warp_id    = t / 32;
            // g_lane_id    = t % 32;
            const_mem[SR_NTID_X]     = blockDim;
            const_mem[SR_GRID_DIM_X] = gridDim;
            const_mem[SR_CTAID_X]    = b;
            const_mem[SR_TID_X]      = t;
            _Z9vectorAddPKfS0_Pfi(
                (uint64_t)A,
                (uint64_t)B,
                (uint64_t)C,
                (uint64_t)N
            );
        }
    }

    bool ok = true;
    for(int i = 0; i < N; ++i) {
        float expected = A[i] + B[i];
        if (std::fabs(C[i] - expected) > 1e-6f) {
            std::cout << "FAIL at index " << i
                      << ": got " << C[i]
                      << ", expected " << expected << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "TEST PASSED\n";
    }

    delete[] A;
    delete[] B;
    delete[] C;
    return ok ? 0 : 1;
}
