#!/usr/bin/env bash
set -euo pipefail

# TESTS=("vecadd" "loop" "resnet18" "tensor" "hetero_mark_aes" "hetero_mark_fir" "hetero_mark_pr" "spaxy" "spaxy1")
TESTS=("loop2")
SM=${SM:-75}
SUITE="cuobjdump" # this script is for cuobjdump only

# Fixed timeout in seconds for each step (compile/run)
TIMEOUT_SECS=10

if ! command -v timeout >/dev/null 2>&1; then
  echo "error: 'timeout' command not found; please install coreutils" >&2
  exit 1
fi

run_t() {
  local secs="$1"; shift
  timeout --preserve-status "${secs}s" "$@"
}

cleanup_link() {
  if [ "${LINK_CREATED:-0}" -eq 1 ]; then
    rm -f "common/kernel_wrapper.h" || true
  fi
}
trap cleanup_link EXIT

for TEST in "${TESTS[@]}"; do
  echo "[cuobjdump] Running test: ${TEST} (sm_${SM})"

  TEST_DIR="${SUITE}/${TEST}"
  OUT_BASE="${TEST}_sm${SM}"

  # Always recompile .cu to SASS for this SM
  echo "Compiling ${TEST}.cu -> ${OUT_BASE}.sass"
  run_t "${TIMEOUT_SECS}" nvcc -arch=sm_${SM} -c "./${TEST_DIR}/${TEST}.cu" -o "${TEST_DIR}/${OUT_BASE}.tmp.o"
  run_t "${TIMEOUT_SECS}" cuobjdump --dump-sass "${TEST_DIR}/${OUT_BASE}.tmp.o" > "${TEST_DIR}/${OUT_BASE}.sass"

  # Symlink correct kernel wrapper
  LINK_CREATED=0
  if [ -f "common/kernel_wrapper_${SM}.h" ]; then
    rm -f "common/kernel_wrapper.h"
    ln -s "kernel_wrapper_${SM}.h" "common/kernel_wrapper.h"
    LINK_CREATED=1
  fi

  # Generate LLVM IR
  run_t "${TIMEOUT_SECS}" python ../main.py -i "${TEST_DIR}/${OUT_BASE}.sass" -o "${TEST_DIR}/${OUT_BASE}"

  # Compile and link in the test subfolder
  run_t "${TIMEOUT_SECS}" clang -O2 -target x86_64 -c "${TEST_DIR}/${OUT_BASE}.ll" -o "${TEST_DIR}/${OUT_BASE}.o"
  run_t "${TIMEOUT_SECS}" clang++ -std=c++20 -O2 -D__CUDA_ARCH__=${SM}0 "./${TEST_DIR}/host_cpu.cpp" "${TEST_DIR}/${OUT_BASE}.o" -I./common -o "${TEST_DIR}/${OUT_BASE}.out"

  echo "Executing ${TEST_DIR}/${OUT_BASE}.out..."
  run_t "${TIMEOUT_SECS}" "./${TEST_DIR}/${OUT_BASE}.out"

  # Clean up artifacts: remove .ll/.o/.out, temp obj, and legacy unsuffixed sass
  rm -f "${TEST_DIR}/${OUT_BASE}.ll" "${TEST_DIR}/${OUT_BASE}.o" "${TEST_DIR}/${OUT_BASE}.out" "${TEST_DIR}/${OUT_BASE}.tmp.o" "${TEST_DIR}/${TEST}.sass"

  # Clean up symlink for this test
  cleanup_link
  trap - EXIT
done
