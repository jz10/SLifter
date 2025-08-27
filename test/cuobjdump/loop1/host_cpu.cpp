#include <cstdint>
#include <iostream>
#include <cmath>
#include <random>
#include "kernel_wrapper.h"

extern "C" void _Z5loop1PKfS0_Pfi();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim  = (N + blockDim - 1) / blockDim;

    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];


    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for(int i = 0; i < N; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    launchKernel(_Z5loop1PKfS0_Pfi, gridDim, blockDim,
                          A, B, C, N);

    bool ok = true;
    for (int idx = 0; idx < N; ++idx) {
        double expected = (static_cast<double>(N - 1) + idx) * (N - idx) * 0.5; // sum_{i=idx}^{N-1} i
        if (std::fabs(static_cast<double>(C[idx]) - expected) > 1e-3 * expected) {
            std::cerr << "Mismatch at " << idx << ": got " << C[idx]
                      << ", expected " << expected << std::endl;
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
