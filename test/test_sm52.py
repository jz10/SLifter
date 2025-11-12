"""
Test suite for SM 52 architecture.
"""
import pytest

from .test_utils import (
    get_test_bases,
    get_bases_for_sm_suite,
    execute_test_case,
    LIFTERS,
)

SM = "52"

# Initialize test bases (identification based on cu or sass presence)
BASES = get_test_bases()

pytest.skip("SM 52 test suite disabled", allow_module_level=True)


def _parametrize_suite(suite: str):
    return pytest.mark.parametrize(
        "base",
        get_bases_for_sm_suite(BASES, SM, suite),
        ids=lambda b: b.split("/", 1)[1],
    )


@pytest.mark.parametrize("lifter", LIFTERS, ids=lambda l: l)
@_parametrize_suite("nvbit")
def test_nvbit(base, lifter, tmp_path):
    execute_test_case(BASES, base, SM, lifter)


@_parametrize_suite("cuobjdump")
def test_cuobjdump_x86(base, tmp_path):
    execute_test_case(BASES, base, SM, "x86")


@_parametrize_suite("cuobjdump")
def test_cuobjdump_nvvm(base, tmp_path):
    execute_test_case(BASES, base, SM, "nvvm")


@_parametrize_suite("hecbench")
def test_hecbench_x86(base, tmp_path):
    execute_test_case(BASES, base, SM, "x86")


@_parametrize_suite("hecbench")
def test_hecbench_nvvm(base, tmp_path):
    execute_test_case(BASES, base, SM, "nvvm")
