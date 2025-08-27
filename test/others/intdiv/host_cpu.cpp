#include <cstdint>
#include <iostream>
#include <cmath>
#include <random>
#include "kernel_wrapper.h"

extern "C" void _Z6intdivPKiS0_Pii();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim  = (N + blockDim - 1) / blockDim;

    int *A = new int[N];
    int *B = new int[N];
    int *C = new int[N];


    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<int> dist(1, 100);  // Avoid division by zero
    for(int i = 0; i < N; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    launchKernel(_Z6intdivPKiS0_Pii, gridDim, blockDim,
                          A, B, C, N);

    bool ok = true;
    for (int idx = 0; idx < gridDim * blockDim; ++idx) {
        if (idx < N) {  // Only check valid indices
            int expected = A[idx] / B[idx];
            
            if (C[idx] != expected) {
                std::cerr << "Mismatch at thread " << idx << ": got " << C[idx]
                          << ", expected " << expected << std::endl;
                ok = false;
                break;
            }
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
