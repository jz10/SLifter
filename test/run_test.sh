#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# SLifter Unified Test Runner
# ============================================================================
#
# Supports cuobjdump, nvbit, others, and hecbench suites in a single script.
# Configure via environment variables:
#   SM=75 SUITE=cuobjdump TESTS=("loop3") bash run_tests_combined.sh
#   SM=75 SUITE=hecbench TESTS=("bfs" "b+tree") bash run_tests_combined.sh
#   TIMEOUT_SECS=5 ... (optional)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SM=${SM:-75}
SUITE=${SUITE:-"cuobjdump"}
TIMEOUT_SECS=${TIMEOUT_SECS:-10}

if [ -z "${TESTS+x}" ]; then
  case "${SUITE}" in
    cuobjdump) TESTS=("loop3") ;;
    hecbench) TESTS=("bfs") ;;
    nvbit) TESTS=("resnet18") ;;
    others) TESTS=("spaxy") ;;
    *)
      echo "error: no default tests for suite '${SUITE}'. Please set TESTS=(...)" >&2
      exit 1
      ;;
  esac
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "error: 'timeout' command not found; please install coreutils" >&2
  exit 1
fi

run_t() {
  local secs="$1"
  shift
  timeout --preserve-status "${secs}s" "$@"
}

cleanup_link() {
  if [ "${LINK_CREATED:-0}" -eq 1 ]; then
    rm -f "common/kernel_wrapper.h" || true
  fi
}

for RAW_TEST in "${TESTS[@]}"; do
  echo "[${SUITE}] Running test: ${RAW_TEST} (sm_${SM})"

  # Normalise hecbench names (strip optional -cuda suffix)
  if [ "${SUITE}" = "hecbench" ]; then
    TEST="${RAW_TEST%-cuda}"
    while [[ "${TEST}" == *"-cuda" ]]; do
      TEST="${TEST%-cuda}"
    done
    TEST_DIR="${SUITE}/${TEST}-cuda"
  else
    TEST="${RAW_TEST}"
    TEST_DIR="${SUITE}/${TEST}"
  fi

  OUT_BASE="${TEST}_sm${SM}"
  SASS_FILE="${TEST_DIR}/${OUT_BASE}.sass"

  if [ ! -d "${TEST_DIR}" ]; then
    echo "error: test directory '${TEST_DIR}' not found" >&2
    continue
  fi

  if [ ! -f "${SASS_FILE}" ]; then
    echo "error: expected SASS file '${SASS_FILE}' not found" >&2
    continue
  fi

  # Handle kernel wrapper symlink for cuobjdump suite
  LINK_CREATED=0
  if [ "${SUITE}" = "cuobjdump" ] && [ -f "common/kernel_wrapper_${SM}.h" ]; then
    rm -f "common/kernel_wrapper.h"
    ln -s "kernel_wrapper_${SM}.h" "common/kernel_wrapper.h"
    LINK_CREATED=1
  fi

  trap cleanup_link EXIT

  echo "Generating LLVM IR from ${SASS_FILE}..."
  run_t "${TIMEOUT_SECS}" python ../main.py -i "${SASS_FILE}" -o "${TEST_DIR}/${OUT_BASE}"

  HOST_CPP="${TEST_DIR}/host_cpu.cpp"
  if [ -f "${HOST_CPP}" ]; then
    echo "Compiling and linking host code (${HOST_CPP})..."
    run_t "${TIMEOUT_SECS}" clang -O2 -target x86_64 -c "${TEST_DIR}/${OUT_BASE}.ll" -o "${TEST_DIR}/${OUT_BASE}.o"
    run_t "${TIMEOUT_SECS}" clang++ -std=c++20 -O2 -D__CUDA_ARCH__=${SM}0 "./${HOST_CPP}" "${TEST_DIR}/${OUT_BASE}.o" -I./common -o "${TEST_DIR}/${OUT_BASE}.out"

    echo "Executing ${TEST_DIR}/${OUT_BASE}.out..."
    run_t "${TIMEOUT_SECS}" "./${TEST_DIR}/${OUT_BASE}.out"

    rm -f "${TEST_DIR}/${OUT_BASE}.o" "${TEST_DIR}/${OUT_BASE}.out"
  else
    echo "Lifting-only test (no host execution)"
  fi

  rm -f "${TEST_DIR}/${OUT_BASE}.ll"

  cleanup_link
  trap - EXIT

  echo "Test ${RAW_TEST} completed successfully!"
done
