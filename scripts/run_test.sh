#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# SLifter Unified Test Runner
# ============================================================================
#
# Example: SM=75 SUITE=hecbench TESTS=atomicCAS LIFTER=NVVM bash scripts/run_test.sh &>results.txt
# Supports cuobjdump, nvbit, others, and hecbench suites in a single script.
# Configure via environment variables (SM, SUITE, TESTS, LIFTER, etc.)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_ROOT="${REPO_ROOT}/test"
cd "${TEST_ROOT}"

SM=${SM:-75}
SUITE=${SUITE:-"cuobjdump"}
LIFTER=${LIFTER:-"x86"}
LIFTER=$(printf "%s" "${LIFTER}" | tr '[:upper:]' '[:lower:]')
TIMEOUT_SECS=${TIMEOUT_SECS:-10}
EXTRA_LIFTER_ARGS=("--verbose")

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

wrapper_include_dir=""
cleanup_wrapper_dir() {
  if [ -n "${wrapper_include_dir:-}" ]; then
    rm -rf "${wrapper_include_dir}" || true
    wrapper_include_dir=""
  fi
}
trap cleanup_wrapper_dir EXIT

for RAW_TEST in "${TESTS[@]}"; do
  echo "[${SUITE}] Running test: ${RAW_TEST} (sm_${SM})"
  cleanup_wrapper_dir

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

  OUT_BASE="${TEST}_${LIFTER}_sm${SM}"
  SASS_FILE="${TEST_DIR}/${TEST}_sm${SM}.sass"

  if [ ! -d "${TEST_DIR}" ]; then
    echo "error: test directory '${TEST_DIR}' not found" >&2
    continue
  fi

  if [ ! -f "${SASS_FILE}" ]; then
    echo "error: expected SASS file '${SASS_FILE}' not found" >&2
    continue
  fi

  echo "Generating LLVM IR from ${SASS_FILE}..."
  run_t "${TIMEOUT_SECS}" python ../main.py -i "${SASS_FILE}" -o "${TEST_DIR}/${OUT_BASE}" --lifter "${LIFTER}" "${EXTRA_LIFTER_ARGS[@]}"

  HOST_CPP="${TEST_DIR}/host_cpu.cpp"
  HOST_CUDA="${TEST_DIR}/host_cuda.cu"

  if [ "${LIFTER}" = "x86" ] && [ -f "${HOST_CPP}" ]; then
    echo "Compiling and linking host code (${HOST_CPP})..."
    if [ -f "common/kernel_wrapper_${SM}.h" ]; then
      wrapper_include_dir="$(mktemp -d "common/kernel_wrapper_tmp.XXXXXX")"
      cp "common/kernel_wrapper_${SM}.h" "${wrapper_include_dir}/kernel_wrapper.h"
    fi
    extra_include=()
    if [ -n "${wrapper_include_dir:-}" ]; then
      extra_include+=("-I./${wrapper_include_dir}")
    fi
    run_t "${TIMEOUT_SECS}" clang -O2 -target x86_64 -c "${TEST_DIR}/${OUT_BASE}.ll" -o "${TEST_DIR}/${OUT_BASE}.o"
    run_t "${TIMEOUT_SECS}" clang++ -std=c++20 -O2 -D__CUDA_ARCH__=${SM}0 "${extra_include[@]}" -I./common "./${HOST_CPP}" "${TEST_DIR}/${OUT_BASE}.o" -o "${TEST_DIR}/${OUT_BASE}.out"

    echo "Executing ${TEST_DIR}/${OUT_BASE}.out..."
    run_t "${TIMEOUT_SECS}" "./${TEST_DIR}/${OUT_BASE}.out"

    # rm -f "${TEST_DIR}/${OUT_BASE}.o" "${TEST_DIR}/${OUT_BASE}.out"
    cleanup_wrapper_dir
  elif [ "${LIFTER}" = "nvvm" ] && [ -f "${HOST_CUDA}" ]; then
    PTX_FILE="${TEST_DIR}/${OUT_BASE}.ptx"
    DEVICE_IMAGE="${PTX_FILE}"

    echo "Lowering NVVM IR to PTX..."
    run_t "${TIMEOUT_SECS}" llc -march=nvptx64 -mcpu=sm_${SM} -mattr=+ptx${SM} -o "${PTX_FILE}" "${TEST_DIR}/${OUT_BASE}.ll"

    if [ -n "${USE_PTXAS:-}" ]; then
      CUBIN_FILE="${TEST_DIR}/${OUT_BASE}.cubin"
      echo "Assembling PTX to cubin (USE_PTXAS set)..."
      run_t "${TIMEOUT_SECS}" ptxas -arch=sm_${SM} "${PTX_FILE}" -o "${CUBIN_FILE}"
      DEVICE_IMAGE="${CUBIN_FILE}"
    fi

    echo "Compiling CUDA host (${HOST_CUDA})..."
    run_t "${TIMEOUT_SECS}" nvcc -std=c++17 -I./common "./${HOST_CUDA}" -lcuda -o "${TEST_DIR}/${OUT_BASE}.out"

    echo "Executing ${TEST_DIR}/${OUT_BASE}.out..."
    run_t "${TIMEOUT_SECS}" "./${TEST_DIR}/${OUT_BASE}.out" "${DEVICE_IMAGE}"

    # rm -f "${TEST_DIR}/${OUT_BASE}.out" "${TEST_DIR}/${OUT_BASE}.ptx" "${TEST_DIR}/${OUT_BASE}.cubin"
  else
    echo "Lifting-only test (no host execution)"
  fi

  rm -f "${TEST_DIR}/${OUT_BASE}.ll"

  echo "Test ${RAW_TEST} completed successfully!"
done
cleanup_wrapper_dir
trap - EXIT
