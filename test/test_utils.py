"""
Shared utilities for SLifter test suite.
Contains common test discovery and execution logic.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"


def run(cmd, cwd):
    """Execute a command and assert success with a fixed 10s timeout."""
    TIMEOUT_SECS = 10
    try:
        r = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=TIMEOUT_SECS,
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            "Timed out after "
            f"{TIMEOUT_SECS}s: {' '.join(map(str, cmd))}\n"
            f"partial stdout:\n{e.stdout or ''}\n"
            f"partial stderr:\n{e.stderr or ''}"
        )

    assert r.returncode == 0, (
        f"rc={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )
    return r.stdout


def ensure_arch_header(sass: str, sm: str) -> str:
    """Ensure the SASS dump contains an 'arch =' header for parser compatibility."""
    if "arch = sm_" in sass:
        return sass
    header = f"arch = sm_{sm}\n"
    # Keep existing leading newlines minimal
    return header + sass.lstrip("\n")


 


LIFTERS = ("x86", "nvvm")

def discover_bases() -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
    """Return { suite/testname: { sm: {sass_rel, hosts: {lifter: host_rel}} } }.
    Include a test if its dir has <name>.cu or any .sass. Use suffixed SASS if present.
    """
    by_base: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}
    suites = ["nvbit", "cuobjdump", "hecbench"]
    sms = ["75", "90"]
    for suite in suites:
        suite_dir = TEST_DIR / suite
        if not suite_dir.exists():
            continue
        for test_dir in suite_dir.iterdir():
            if not test_dir.is_dir():
                continue
            testname = test_dir.name
            
            # Handle HeCBench naming convention
            if suite == "hecbench":
                # HeCBench tests are in directories named <testname>-cuda
                if testname.endswith("-cuda"):
                    actual_testname = testname[:-5]  # Remove "-cuda" suffix
                    # Try multiple possible .cu file names
                    cu_file = test_dir / "main.cu"
                    if not cu_file.exists():
                        cu_file = test_dir / f"{actual_testname}.cu"
                else:
                    continue  # Skip non-cuda directories in hecbench
            else:
                actual_testname = testname
                cu_file = test_dir / f"{testname}.cu"
            
            if not cu_file.exists() and not list(test_dir.glob("*.sass")):
                continue
            
            host_cpu_rel = str((test_dir / "host_cpu.cpp").relative_to(TEST_DIR)) if (test_dir / "host_cpu.cpp").exists() else ""
            host_cuda_rel = str((test_dir / "host_cuda.cu").relative_to(TEST_DIR)) if (test_dir / "host_cuda.cu").exists() else ""
            base = f"{suite}/{actual_testname}"
            by_base.setdefault(base, {})
            for sm in sms:
                suffixed = test_dir / f"{actual_testname}_sm{sm}.sass"
                by_base[base][sm] = {
                    "sass_rel": str(suffixed.relative_to(TEST_DIR)) if suffixed.exists() else "",
                    "hosts": {
                        "x86": host_cpu_rel,
                        "nvvm": host_cuda_rel,
                    },
                }
    return by_base


def discover_bases_legacy() -> Dict[str, Dict[str, Dict[str, str]]]:
    """Deprecated: retained for compatibility, returns an empty mapping."""
    return {}


def get_test_bases():
    """Get test bases from available SASS in canonical naming only."""
    return discover_bases()


def ensure_cuobjdump_every_sm(bases: Dict[str, Dict[str, Dict[str, str]]], target_sm: str):
    """Deprecated: no-op."""
    return


def get_bases_for_sm_suite(bases: Dict[str, Dict[str, Dict[str, str]]], sm: str, suite: str) -> List[str]:
    """Get all test bases for a specific SM and suite"""
    return sorted([
        base for base, sm_map in bases.items()
        if sm in sm_map and base.startswith(f"{suite}/")
    ])


def execute_test_case(bases: Dict[str, Dict[str, Dict[str, Dict[str, str]]]], base: str, sm: str, lifter: str):
    """Execute a single test case following the new workflow.

    - Only SASS drives execution.
    - For cuobjdump suite: host_cpu.cpp must exist; otherwise skip as broken.
    - For others/nvbit: lifting without error is sufficient; host is optional.
    - When linking host, provide the correct kernel wrapper via a per-test include dir.
    """
    entry = bases.get(base, {}).get(sm)
    if not entry:
        pytest.skip(f"No SASS or CU for {base} sm{sm}")

    sass_rel = entry.get("sass_rel", "")
    hosts = entry.get("hosts", {})
    lifter = lifter.lower()
    host_rel = hosts.get(lifter, "")

    clean_paths: List[Path] = []
    temp_dirs: List[Path] = []

    # Ensure suffixed SASS exists; if not, try compiling from .cu or renaming legacy .sass
    suite_name, testname = base.split("/", 1) if "/" in base else (base, base)
    
    # Handle HeCBench directory naming
    if suite_name == "hecbench":
        test_dir = TEST_DIR / suite_name / f"{testname}-cuda"
    else:
        test_dir = TEST_DIR / suite_name / testname
    
    desired_sass_path = test_dir / f"{testname}_sm{sm}.sass"
    recorded_sass_path = TEST_DIR / sass_rel if sass_rel else None
    force_rebuild = suite_name == "cuobjdump"
    sass_path = None

    if not force_rebuild:
        if desired_sass_path.exists():
            sass_path = desired_sass_path
        elif recorded_sass_path and recorded_sass_path.exists():
            sass_path = recorded_sass_path

    if not sass_path:
        # Handle HeCBench main.cu naming
        if suite_name == "hecbench":
            cu_path = test_dir / "main.cu"
            if not cu_path.exists():
                cu_path = test_dir / f"{testname}.cu"
        else:
            cu_path = test_dir / f"{testname}.cu"
            
        # If still absent, try compiling from .cu (handle multi-file projects)
        should_compile = cu_path.exists() and (force_rebuild or not desired_sass_path.exists())
        if should_compile:
            build_tmp_root = test_dir / ".tmp_builds"
            build_tmp_root.mkdir(exist_ok=True)
            tmp_build_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"{testname}_sm{sm}_",
                    dir=str(build_tmp_root),
                )
            )
            temp_dirs.append(tmp_build_dir)

            if desired_sass_path.exists() and not force_rebuild:
                sass_path = desired_sass_path
            else:
                tmp_exe = tmp_build_dir / "temp.out"
                try:
                    if suite_name == "hecbench":
                        makefile_path = test_dir / "Makefile"
                        if makefile_path.exists():
                            build_dir = tmp_build_dir / testname
                            shutil.copytree(test_dir, build_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".tmp_builds"))
                            try:
                                run(["make", "clean"], cwd=build_dir)
                            except AssertionError:
                                pass

                            run(["make", f"ARCH=sm_{sm}"], cwd=build_dir)

                            possible_exes = list(build_dir.glob("*.out")) + list(build_dir.glob("main")) + list(build_dir.glob(f"{testname}"))
                            if possible_exes:
                                built_exe = possible_exes[0]
                                sass_out = run(["cuobjdump", "--dump-sass", str(built_exe)], cwd=build_dir)
                                desired_sass_path.write_text(ensure_arch_header(sass_out, sm))
                            else:
                                raise AssertionError("No executable found after make")
                        else:
                            cu_files = list(test_dir.glob("**/*.cu"))
                            c_files = list(test_dir.glob("**/*.c"))

                            compile_cmd = ["nvcc", f"-arch=sm_{sm}", "-I.", f"-I{test_dir.relative_to(TEST_DIR)}"]
                            compile_cmd.extend(["-Ddfloat=double", "-Ddlong=int", "-std=c++14", "-w"])

                            if len(cu_files) > 1 or len(c_files) > 0:
                                cu_file_paths = [f"./{f.relative_to(TEST_DIR)}" for f in cu_files]
                                c_file_paths = [f"./{f.relative_to(TEST_DIR)}" for f in c_files]
                                compile_cmd.extend(cu_file_paths + c_file_paths)
                            else:
                                compile_cmd.append(f"./{cu_path.relative_to(TEST_DIR)}")

                            compile_cmd.extend(["-o", str(tmp_exe)])
                            run(compile_cmd, cwd=TEST_DIR)
                            sass_out = run(["cuobjdump", "--dump-sass", str(tmp_exe)], cwd=TEST_DIR)
                            desired_sass_path.write_text(ensure_arch_header(sass_out, sm))
                    else:
                        run([
                            "nvcc",
                            f"-arch=sm_{sm}",
                            "-cubin",
                            f"./{cu_path.relative_to(TEST_DIR)}",
                            "-o",
                            str(tmp_exe),
                        ], cwd=TEST_DIR)
                        sass_out = run(["cuobjdump", "--dump-sass", str(tmp_exe)], cwd=TEST_DIR)
                        desired_sass_path.write_text(ensure_arch_header(sass_out, sm))
                except AssertionError:
                    pytest.skip(f"Cannot build SASS for {base} sm{sm}")
                finally:
                    tmp_exe.unlink(missing_ok=True)

        if desired_sass_path.exists():
            sass_path = desired_sass_path
        else:
            pytest.skip(f"No SASS or CU for {base} sm{sm}")

    if sass_path:
        sass_rel = str(sass_path.relative_to(TEST_DIR))
        entry["sass_rel"] = sass_rel


    # Extract testname from base for output naming
    testname = base.split('/')[-1]
    out_base = f"{testname}_{lifter}_sm{sm}"

    # Track generated build artifacts to remove even if the test fails
    kernel_wrapper_variant = TEST_DIR / "common" / f"kernel_wrapper_{sm}.h"
    wrapper_include_dir: Optional[Path] = None

    #clean_paths: List[Path] = []
    #temp_dirs: List[Path] = []

    #clean_paths: List[Path] = []

    try:
        # 1) lift to LLVM
        if sass_rel:
            sass_path = Path(sass_rel)
            test_subdir = sass_path.parent

            # Output SASS compilation in test subfolder with new naming
            out_ll_path = test_subdir / f"{out_base}.ll"
            out_o_path = test_subdir / f"{out_base}.o"
            # Keep generated .ll files after tests to aid debugging

            lifter_cmd = [
                "python",
                "../main.py",
                "-i",
                f"{sass_rel}",
                "-o",
                str(out_ll_path.with_suffix('')),
                "--lifter",
                lifter,
            ]
            run(lifter_cmd, cwd=TEST_DIR)

            # 2) link & execute when host present; otherwise lifting success is enough
            should_run_host = bool(host_rel)

            # Enforce host existence for cuobjdump suite
            if suite_name == "cuobjdump" and not host_rel:
                pytest.skip(f"Broken cuobjdump test (missing host for {lifter} lifter): {base} sm{sm}")

            # For HeCBench, only do lifting for now (no host execution until host wrappers are created)
            if suite_name == "hecbench":
                should_run_host = False

            if should_run_host:
                exe_path = test_subdir / f"{out_base}.out"

                if lifter == "x86":
                    if wrapper_include_dir is None and kernel_wrapper_variant.exists():
                        tmp_dir_path = Path(
                            tempfile.mkdtemp(
                                prefix="kernel_wrapper_",
                                dir=str(TEST_DIR / "common"),
                            )
                        )
                        shutil.copy(kernel_wrapper_variant, tmp_dir_path / "kernel_wrapper.h")
                        wrapper_include_dir = tmp_dir_path
                        temp_dirs.append(wrapper_include_dir)

                    run([
                        "clang",
                        "-O2",
                        "-target",
                        "x86_64",
                        "-c",
                        str(out_ll_path),
                        "-o",
                        str(out_o_path),
                    ], cwd=TEST_DIR)
                    clean_paths.append(TEST_DIR / out_o_path)

                    cu_flags: List[str] = [f"-D__CUDA_ARCH__={sm}0"]

                    include_flags = []
                    if wrapper_include_dir:
                        include_flags.append(f"-I{wrapper_include_dir.relative_to(TEST_DIR)}")
                    link_cmd = [
                        "clang++",
                        "-std=c++20",
                        "-O2",
                        *cu_flags,
                        *include_flags,
                        "-I./common",
                        f"./{host_rel}",
                        str(out_o_path),
                        "-o",
                        str(exe_path),
                    ]
                    run(link_cmd, cwd=TEST_DIR)

                    out = run([str(exe_path)], cwd=TEST_DIR)
                    assert "TEST PASSED" in out
                elif lifter == "nvvm":
                    ptx_path = test_subdir / f"{out_base}.ptx"
                    cubin_path = test_subdir / f"{out_base}.cubin"

                    run([
                        "llc",
                        "-march=nvptx64",
                        f"-mcpu=sm_{sm}",
                        f"-mattr=+ptx{sm}",
                        "-o",
                        str(ptx_path),
                        str(out_ll_path),
                    ], cwd=TEST_DIR)

                    device_image_path = ptx_path
                    if os.environ.get("USE_PTXAS"):
                        run([
                            "ptxas",
                            f"-arch=sm_{sm}",
                            str(ptx_path),
                            "-o",
                            str(cubin_path),
                        ], cwd=TEST_DIR)
                        clean_paths.append(TEST_DIR / cubin_path)
                        device_image_path = cubin_path

                    nvcc_cmd = [
                        "nvcc",
                        "-std=c++17",
                        "-I./common",
                        f"./{host_rel}",
                        "-lcuda",
                        "-o",
                        str(exe_path),
                    ]
                    run(nvcc_cmd, cwd=TEST_DIR)
                    out = run([str(exe_path), str(device_image_path)], cwd=TEST_DIR)
                    assert "TEST PASSED" in out
                else:
                    raise AssertionError(f"Unhandled lifter pipeline '{lifter}'")

            

        else:
            pytest.skip(f"Test {base} sm{sm} ({lifter}): no executable configuration")
    finally:
        # Always try to cleanup generated artifacts, even on failure
        for p in clean_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        # Always cleanup temporary wrapper include dirs
        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
