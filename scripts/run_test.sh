#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# SLifter Unified Test Runner
# ============================================================================
# Usage: SM=75 SUITE=cuobjdump TESTS=loop3 LIFTER=nvvm bash scripts/run_test.sh &>results.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_ROOT="${REPO_ROOT}/test"

# Config
SM="${SM:-75}"
SUITE="${SUITE:-cuobjdump}"
LIFTER="${LIFTER:-x86}"
TIMEOUT_SECS="${TIMEOUT_SECS:-10}"
EXTRA_LIFTER_ARGS=("${EXTRA_LIFTER_ARGS[@]:---verbose}")

LIFTER=$(echo "${LIFTER}" | tr '[:upper:]' '[:lower:]')

log() { echo "[INFO] $1"; }
error() { echo "[FAIL] $1" >&2; }

if [ -z "${TESTS+x}" ]; then
  error "No tests specified. Set 'TESTS' environment variable."
  exit 1
fi

if ! command -v timeout >/dev/null 2>&1; then
  error "'timeout' command missing."
  exit 1
fi

run_t() {
  local duration="$1"
  shift
  timeout --preserve-status "${duration}s" "$@"
}

WRAPPER_TMP_DIR=""
cleanup() {
  if [ -n "${WRAPPER_TMP_DIR}" ]; then
    rm -rf "${WRAPPER_TMP_DIR}"
  fi
}
trap cleanup EXIT INT TERM

# --- Compilation Logic ---

compile_artifacts_if_needed() {
  local src_file="${CURRENT_TEST_DIR}/${CURRENT_TEST_NAME}.cu"
  
  # Only compile if missing (primary use case: cuobjdump suite)
  if [[ -f "${CURRENT_SASS_FILE}" ]] && [[ -f "${CURRENT_ELF_FILE}" ]]; then
    return 0
  fi

  if [[ ! -f "${src_file}" ]]; then
    error "Missing SASS/ELF and no source file found at ${src_file}"
    return 1
  fi

  log "Compiling SASS/ELF from ${src_file}..."
  
  local build_tmp
  build_tmp=$(mktemp -d)
  local tmp_exe="${build_tmp}/temp.exe"

  # Compile to temp executable
  run_t "${TIMEOUT_SECS}" nvcc -arch="sm_${SM}" \
    -Ddfloat=double -Ddlong=int -std=c++14 -w \
    -I. -I"${CURRENT_TEST_DIR}" \
    "${src_file}" -o "${tmp_exe}"

  # Extract artifacts
  run_t "${TIMEOUT_SECS}" cuobjdump --dump-sass "${tmp_exe}" > "${CURRENT_SASS_FILE}"
  run_t "${TIMEOUT_SECS}" cuobjdump --dump-elf "${tmp_exe}" > "${CURRENT_ELF_FILE}"

  rm -rf "${build_tmp}"
}

# --- Execution Stages ---

resolve_paths() {
  local raw_name="$1"
  local test_name="${raw_name}"
  local test_dir_rel="${SUITE}/${test_name}"

  # HecBench directory normalization
  if [ "${SUITE}" = "hecbench" ]; then
    test_name="${raw_name%-cuda}"
    while [[ "${test_name}" == *"-cuda" ]]; do test_name="${test_name%-cuda}"; done
    test_dir_rel="${SUITE}/${test_name}-cuda"
  fi

  CURRENT_TEST_NAME="${test_name}"
  CURRENT_TEST_DIR="${TEST_ROOT}/${test_dir_rel}"
  CURRENT_SASS_FILE="${CURRENT_TEST_DIR}/${test_name}_sm${SM}.sass"
  CURRENT_ELF_FILE="${CURRENT_TEST_DIR}/${test_name}_sm${SM}.elf"
  CURRENT_OUT_BASE="${test_name}_${LIFTER}_sm${SM}"
  
  if [ ! -d "${CURRENT_TEST_DIR}" ]; then
    error "Directory not found: ${test_dir_rel}"
    return 1
  fi
  return 0
}

stage_lift() {
  log "Lifting SASS + ELF to LLVM IR..."
  # Pass both SASS and ELF to -i
  run_t "${TIMEOUT_SECS}" \
    python "${REPO_ROOT}/main.py" \
      -i "${CURRENT_SASS_FILE}" "${CURRENT_ELF_FILE}" \
      -o "${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}" \
      --lifter "${LIFTER}" \
      "${EXTRA_LIFTER_ARGS[@]}"
}

stage_run_x86() {
  local host_src="${CURRENT_TEST_DIR}/host_cpu.cpp"
  [ ! -f "${host_src}" ] && return 0

  log "Compiling X86 Host..."
  
  local wrapper_src="${TEST_ROOT}/common/kernel_wrapper_${SM}.h"
  local include_args=()
  
  cleanup
  if [ -f "${wrapper_src}" ]; then
    WRAPPER_TMP_DIR="$(mktemp -d "${TEST_ROOT}/common/wrapper.XXXXXX")"
    cp "${wrapper_src}" "${WRAPPER_TMP_DIR}/kernel_wrapper.h"
    include_args+=("-I${WRAPPER_TMP_DIR}")
  fi

  run_t "${TIMEOUT_SECS}" clang -O2 -target x86_64 -c \
    "${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}.ll" \
    -o "${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}.o"

  run_t "${TIMEOUT_SECS}" clang++ -std=c++20 -O2 -D__CUDA_ARCH__="${SM}0" \
    "${include_args[@]}" -I"${TEST_ROOT}/common" \
    "${host_src}" \
    "${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}.o" \
    -o "${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}.out"

  log "Running..."
  run_t "${TIMEOUT_SECS}" "${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}.out"
  
  rm -f "${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}.o" 
}

stage_run_nvvm() {
  local host_src="${CURRENT_TEST_DIR}/host_cuda.cu"
  [ ! -f "${host_src}" ] && return 0

  log "Compiling NVVM Host..."
  local ptx="${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}.ptx"
  local img="${ptx}"

  run_t "${TIMEOUT_SECS}" llc -march=nvptx64 -mcpu="sm_${SM}" -mattr="+ptx${SM}" \
    -o "${ptx}" "${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}.ll"

  if [ -n "${USE_PTXAS:-}" ]; then
    local cubin="${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}.cubin"
    run_t "${TIMEOUT_SECS}" ptxas -arch="sm_${SM}" "${ptx}" -o "${cubin}"
    img="${cubin}"
  fi

  run_t "${TIMEOUT_SECS}" nvcc -std=c++17 -I"${TEST_ROOT}/common" \
    "${host_src}" -lcuda -o "${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}.out"

  log "Running..."
  run_t "${TIMEOUT_SECS}" "${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}.out" "${img}"
}

# --- Main ---

cd "${TEST_ROOT}"

# Split TESTS string into array
IFS=' ' read -r -a TEST_ARRAY <<< "${TESTS}"

for RAW_TEST in "${TEST_ARRAY[@]}"; do
  echo "------------------------------------------------------------"
  log "Test: ${SUITE}/${RAW_TEST} (SM=${SM}, Lifter=${LIFTER})"

  if ! resolve_paths "${RAW_TEST}"; then continue; fi
  
  if ! compile_artifacts_if_needed; then continue; fi

  stage_lift

  case "${LIFTER}" in
    x86)  stage_run_x86 ;;
    nvvm) stage_run_nvvm ;;
    *)    log "Lifter '${LIFTER}' execution not supported." ;;
  esac

  # Cleanup run artifacts
  rm -f "${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}.ll" \
        "${CURRENT_TEST_DIR}/${CURRENT_OUT_BASE}.out"

  echo "[PASS] ${RAW_TEST} completed."
done
