"""
Test suite for SM 75 architecture.
"""
from test_utils import (
    get_test_bases,
    get_bases_for_sm_suite, execute_test_case, pytest
)

SM = "75"

# Initialize test bases (identification based on cu or sass presence)
BASES = get_test_bases()

if pytest:
    @pytest.mark.parametrize("base", get_bases_for_sm_suite(BASES, SM, "nvbit"), 
                            ids=lambda b: b.split("/", 1)[1])
    def test_nvbit(base, tmp_path):
        execute_test_case(BASES, base, SM)

    @pytest.mark.parametrize("base", get_bases_for_sm_suite(BASES, SM, "cuobjdump"), 
                            ids=lambda b: b.split("/", 1)[1])
    def test_cuobjdump(base, tmp_path):
        execute_test_case(BASES, base, SM)

    @pytest.mark.parametrize("base", get_bases_for_sm_suite(BASES, SM, "others"), 
                            ids=lambda b: b.split("/", 1)[1])
    def test_others(base, tmp_path):
        execute_test_case(BASES, base, SM)