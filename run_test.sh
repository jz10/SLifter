#!/usr/bin/env bash

TEST=vecadd

python main.py -i ./test/vecaddtest.sass -o kernel

clang++ -O2 -c intrinsics.cpp -o intrinsics.o
clang -O2 -target x86_64 -c kernel.ll   -o kernel.o
clang++ -O2 host_cpu.cpp kernel.o intrinsics.o -o ${TEST}.out

echo "Running ${TEST}..."
./${TEST}.out
