#!/usr/bin/env bash

# TESTS=("vecadd" "loop" "resnet18" "tensor" "hetero_mark_aes" "hetero_mark_fir" "hetero_mark_pr" "spaxy" "spaxy1")
TESTS=("loop2")
# TESTS=("vecadd" "spaxy" "spaxy1")

for TEST in "${TESTS[@]}"; do
  echo "Running test: ${TEST}"
  # Generate LLVM IR
  python ../main.py -i "${TEST}/${TEST}.sass" -o kernel

  # Compile and link
  clang -O2 -target x86_64 -c kernel.ll -o kernel.o
  clang++ -std=c++20 -O2 ./${TEST}/host_cpu.cpp kernel.o -I.. -o "${TEST}.out"

  echo "Executing ${TEST}.out..."
  "./${TEST}.out"

  # Clean up artifacts
  rm -f kernel.o kernel.ll "${TEST}.out"
done