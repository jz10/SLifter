import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
LIFTERS = ("x86", "nvvm")

def run_cmd(cmd: List[str], cwd: Path, timeout: int = 120) -> str:
    try:
        r = subprocess.run(
            cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            f"Timeout ({timeout}s): {' '.join(map(str, cmd))}\n{e.stdout}\n{e.stderr}"
        )

    assert r.returncode == 0, (
        f"Failed (rc={r.returncode}): {' '.join(map(str, cmd))}\n{r.stdout}\n{r.stderr}"
    )
    return r.stdout

def _dump_elf_from_exe(exe_path: Path, out_path: Path, cwd: Path):
    out = run_cmd(["cuobjdump", "--dump-elf", str(exe_path)], cwd)
    out_path.write_bytes(out.encode("utf-8") if isinstance(out, str) else out)

def discover_bases() -> Dict[str, Dict]:
    by_base = {}
    suites = ["nvbit", "cuobjdump", "hecbench"]
    sms = ["75", "90"]

    for suite in suites:
        suite_dir = TEST_DIR / suite
        if not suite_dir.exists(): continue

        for test_dir in suite_dir.iterdir():
            if not test_dir.is_dir(): continue

            testname = test_dir.name
            if suite == "hecbench":
                if not testname.endswith("-cuda"): continue
                testname = testname[:-5]

            cu_file = test_dir / f"{testname}.cu"
            if suite == "hecbench" and not cu_file.exists():
                cu_file = test_dir / "main.cu"
            
            # Ensure we only count SASS if the corresponding ELF also exists
            has_valid_sass = any(s.exists() and s.with_suffix(".elf").exists() for s in test_dir.glob("*.sass"))

            if not cu_file.exists() and not has_valid_sass:
                continue

            hosts = {}
            if (test_dir / "host_cpu.cpp").exists():
                hosts["x86"] = str((test_dir / "host_cpu.cpp").relative_to(TEST_DIR))
            if (test_dir / "host_cuda.cu").exists():
                hosts["nvvm"] = str((test_dir / "host_cuda.cu").relative_to(TEST_DIR))

            base_key = f"{suite}/{testname}"
            by_base.setdefault(base_key, {})

            for sm in sms:
                sass_file = test_dir / f"{testname}_sm{sm}.sass"
                elf_file = sass_file.with_suffix(".elf")
                
                # Only record path if BOTH exist
                sass_rel = ""
                if sass_file.exists() and elf_file.exists():
                    sass_rel = str(sass_file.relative_to(TEST_DIR))

                by_base[base_key][sm] = {
                    "sass_rel": sass_rel,
                    "hosts": hosts
                }
    return by_base

def get_test_bases():
    return discover_bases()

def get_bases_for_sm_suite(bases: Dict, sm: str, suite: str) -> List[str]:
    return sorted([b for b, m in bases.items() if sm in m and b.startswith(f"{suite}/")])

def _compile_cuda_binary(test_dir: Path, testname: str, sm: str, out_dir: Path) -> Tuple[Path, Path]:
    tmp_exe = out_dir / "temp.out"
    
    if (test_dir / "Makefile").exists():
        build_root = out_dir / testname
        shutil.copytree(test_dir, build_root, dirs_exist_ok=True)
        
        run_cmd(["make", "clean"], cwd=build_root)
        run_cmd(["make", f"ARCH=sm_{sm}"], cwd=build_root)
        
        candidates = list(build_root.glob("*.out")) + list(build_root.glob("main")) + list(build_root.glob(testname))
        if not candidates:
            raise AssertionError("Makefile did not produce an executable")
        
        sass_out = run_cmd(["cuobjdump", "--dump-sass", str(candidates[0])], cwd=build_root)
        return sass_out, candidates[0]

    cu_files = list(test_dir.glob("**/*.cu"))
    c_files = list(test_dir.glob("**/*.c"))
    
    cmd = [
        "nvcc", f"-arch=sm_{sm}", "-I.", f"-I{test_dir.relative_to(TEST_DIR)}",
        "-Ddfloat=double", "-Ddlong=int", "-std=c++14", "-w"
    ]
    
    if len(cu_files) > 1 or c_files:
        cmd.extend([f"./{f.relative_to(TEST_DIR)}" for f in cu_files + c_files])
        cmd.extend(["-o", str(tmp_exe)])
        run_cmd(cmd, cwd=TEST_DIR)
    else:
        main_cu = test_dir / "main.cu" if (test_dir / "main.cu").exists() else test_dir / f"{testname}.cu"
        cmd = ["nvcc", f"-arch=sm_{sm}", "-cubin", f"./{main_cu.relative_to(TEST_DIR)}", "-o", str(tmp_exe)]
        run_cmd(cmd, cwd=TEST_DIR)
        
    sass_out = run_cmd(["cuobjdump", "--dump-sass", str(tmp_exe)], cwd=TEST_DIR)
    return sass_out, tmp_exe

def _ensure_sass(base: str, sm: str, entry: Dict) -> Path:
    suite, testname = base.split("/", 1)
    sass_rel = entry.get("sass_rel")
    
    # Use recorded SASS if both SASS and ELF exist
    if sass_rel:
        p = TEST_DIR / sass_rel
        if p.exists() and p.with_suffix(".elf").exists():
            return p

    test_dir = TEST_DIR / suite / (f"{testname}-cuda" if suite == "hecbench" else testname)
    target_sass = test_dir / f"{testname}_sm{sm}.sass"
    target_elf = target_sass.with_suffix(".elf")

    # If physical files exist (both), return early
    if target_sass.exists() and target_elf.exists():
        return target_sass

    if not list(test_dir.glob("*.cu")) and not (test_dir / "Makefile").exists():
        pytest.skip(f"No SASS or Source found for {base}")

    with tempfile.TemporaryDirectory(prefix=f"build_{testname}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            sass_content, exe_path = _compile_cuda_binary(test_dir, testname, sm, tmp_path)
            target_sass.write_text(sass_content)
            _dump_elf_from_exe(exe_path, target_elf, tmp_path)
        except AssertionError as e:
             pytest.skip(f"Build failed for {base}: {e}")

    return target_sass

def _lift_to_llvm(sass_path: Path, lifter: str, out_base_path: Path) -> Path:
    out_ll = out_base_path.with_suffix(".ll")
    elf_path = sass_path.with_suffix(".elf")
    
    cmd = [
        "python", "../main.py",
        "-i", 
        str(sass_path.relative_to(TEST_DIR)), 
        str(elf_path.relative_to(TEST_DIR)),
        "-o", str(out_ll.with_suffix('')),
        "--lifter", lifter
    ]
    run_cmd(cmd, cwd=TEST_DIR)
    return out_ll

def _run_x86_host(exe_path: Path, ll_path: Path, host_rel: str, sm: str):
    obj_path = exe_path.with_suffix(".o")
    
    run_cmd(["clang", "-O2", "-target", "x86_64", "-c", str(ll_path), "-o", str(obj_path)], cwd=TEST_DIR)
    
    wrapper_src = TEST_DIR / "common" / f"kernel_wrapper_{sm}.h"
    
    with tempfile.TemporaryDirectory(dir=TEST_DIR / "common") as tmp_inc:
        shutil.copy(wrapper_src, Path(tmp_inc) / "kernel_wrapper.h")
        
        link_cmd = [
            "clang++", "-std=c++20", "-O2", 
            f"-D__CUDA_ARCH__={sm}0",
            f"-I{Path(tmp_inc).relative_to(TEST_DIR)}",
            "-I./common",
            f"./{host_rel}",
            str(obj_path),
            "-o", str(exe_path)
        ]
        run_cmd(link_cmd, cwd=TEST_DIR)

    out = run_cmd([str(exe_path)], cwd=TEST_DIR)
    assert "TEST PASSED" in out

def _run_nvvm_host(exe_path: Path, ll_path: Path, host_rel: str, sm: str, out_base_ptx: Path):
    ptx_path = out_base_ptx.with_suffix(".ptx")
    cubin_path = out_base_ptx.with_suffix(".cubin")

    run_cmd([
        "llc", "-march=nvptx64", f"-mcpu=sm_{sm}", f"-mattr=+ptx{sm}",
        "-o", str(ptx_path), str(ll_path)
    ], cwd=TEST_DIR)

    device_img = ptx_path
    if os.environ.get("USE_PTXAS"):
        run_cmd(["ptxas", f"-arch=sm_{sm}", str(ptx_path), "-o", str(cubin_path)], cwd=TEST_DIR)
        device_img = cubin_path

    run_cmd([
        "nvcc", "-std=c++17", "-I./common", f"./{host_rel}", "-lcuda",
        "-o", str(exe_path)
    ], cwd=TEST_DIR)

    out = run_cmd([str(exe_path), str(device_img)], cwd=TEST_DIR)
    assert "TEST PASSED" in out

def execute_test_case(bases: Dict, base: str, sm: str, lifter: str):
    entry = bases.get(base, {}).get(sm)
    if not entry:
        pytest.skip(f"No configuration found for {base} sm{sm}")

    lifter = lifter.lower()
    host_rel = entry["hosts"].get(lifter)
    
    sass_path = _ensure_sass(base, sm, entry)
    
    test_subdir = sass_path.parent
    testname = base.split('/')[-1]
    out_base = test_subdir / f"{testname}_{lifter}_sm{sm}"
    
    out_ll = _lift_to_llvm(sass_path, lifter, out_base)

    suite = base.split('/')[0]
    if suite == "hecbench":
        return

    if not host_rel:
        if suite == "cuobjdump":
            pytest.skip("Skipping cuobjdump test: missing host implementation")
        return 

    exe_path = out_base.with_suffix(".out")
    
    try:
        if lifter == "x86":
            _run_x86_host(exe_path, out_ll, host_rel, sm)
        elif lifter == "nvvm":
            _run_nvvm_host(exe_path, out_ll, host_rel, sm, out_base)
        else:
            raise NotImplementedError(f"Unknown lifter: {lifter}")
    finally:
        for p in [exe_path, out_base.with_suffix(".o"), out_base.with_suffix(".cubin")]:
            p.unlink(missing_ok=True)