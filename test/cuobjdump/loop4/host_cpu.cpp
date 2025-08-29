#include <cstdint>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include "kernel_wrapper.h"

extern "C" void _Z5loop4PKfS0_Pfi();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim  = (N + blockDim - 1) / blockDim;
    const int totalThreads = gridDim * blockDim;

    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    std::fill(C, C + N, 0.0f);

    launchKernel(_Z5loop4PKfS0_Pfi, gridDim, blockDim, A, B, C, N);

    bool ok = true;
    for (int j = 0; j < N; ++j) {
        int updates = std::min(j + 1, totalThreads);
        double expected = static_cast<double>(updates) *
                          (static_cast<double>(A[j]) + static_cast<double>(B[j]));

        double got = static_cast<double>(C[j]);
        double tol = 1e-3 * std::fabs(expected);
        if (std::fabs(got - expected) > tol) {
            std::cerr << "Mismatch at " << j << ": got " << C[j]
                      << ", expected " << expected << " (updates=" << updates << ")\n";
            ok = false;
            break;
        }
    }

    if (ok) std::cout << "TEST PASSED\n";

    delete[] A;
    delete[] B;
    delete[] C;
    return ok ? 0 : 1;
}
