import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
# TESTS = ["vecadd", "spaxy", "spaxy1"] 
TESTS = [p.name for p in (TEST_DIR).iterdir() if (p / f"{p.name}.sass").exists()]

def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    assert r.returncode == 0, f"rc={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    return r.stdout

import pytest

@pytest.mark.parametrize("name", TESTS)
def test_end_to_end(name, tmp_path):
    # 1) lift to LLVM
    run(["python", "../main.py", "-i", f"{name}/{name}.sass", "-o", "kernel"], cwd=TEST_DIR)

    # 2) compile & link
    run(["clang", "-O2", "-target", "x86_64", "-c", "kernel.ll", "-o", "kernel.o"], cwd=TEST_DIR)
    run(["clang++", "-std=c++20", "-O2", f"./{name}/host_cpu.cpp", "kernel.o", "-I..", "-o", f"{name}.out"], cwd=TEST_DIR)

    # 3) execute
    out = run([f"./{name}.out"], cwd=TEST_DIR)
    assert "TEST PASSED" in out

    # Optional: compare produced IR against a golden file
    # golden = (ROOT / "tests" / "golden" / f"{name}.ll").read_text()
    # current = (TEST_DIR / "kernel.ll").read_text()
    # assert current == golden

    # Clean (or keep tmp_path artifacts if you want)
    for f in ["kernel.o", "kernel.ll", f"{name}.out"]:
        (TEST_DIR / f).unlink(missing_ok=True)
