"""
Shared utilities for SLifter test suite.
Contains common test discovery and execution logic.
"""
import subprocess
from pathlib import Path
from typing import List, Dict

try:
    import pytest
except ImportError:
    pytest = None

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


 


def discover_bases() -> Dict[str, Dict[str, Dict[str, str]]]:
    """Return { suite/testname: { sm: {sass_rel, host_rel} } }.
    Include a test if its dir has <name>.cu or any .sass. Use suffixed SASS if present.
    """
    by_base: Dict[str, Dict[str, Dict[str, str]]] = {}
    suites = ["nvbit", "cuobjdump", "hecbench"]
    sms = ["75"]
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
            
            host_rel = str((test_dir / "host_cpu.cpp").relative_to(TEST_DIR)) if (test_dir / "host_cpu.cpp").exists() else ""
            base = f"{suite}/{actual_testname}"
            by_base.setdefault(base, {})
            for sm in sms:
                suffixed = test_dir / f"{actual_testname}_sm{sm}.sass"
                by_base[base][sm] = {"sass_rel": str(suffixed.relative_to(TEST_DIR)) if suffixed.exists() else "", "host_rel": host_rel}
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


def execute_test_case(bases: Dict[str, Dict[str, Dict[str, str]]], base: str, sm: str):
    """Execute a single test case following the new workflow.

    - Only SASS drives execution.
    - For cuobjdump suite: host_cpu.cpp must exist; otherwise skip as broken.
    - For others/nvbit: lifting without error is sufficient; host is optional.
    - When linking host, symlink the correct common/kernel_wrapper_{sm}.h.
    """
    entry = bases.get(base, {}).get(sm)
    if not entry:
        if pytest:
            pytest.skip(f"No SASS or CU for {base} sm{sm}")
        else:
            return  # Skip silently if pytest not available

    sass_rel = entry.get("sass_rel", "")
    host_rel = entry.get("host_rel", "")

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
        legacy = test_dir / f"{testname}.sass"
        # Try rename legacy first
        if legacy.exists() and not force_rebuild:
            try:
                legacy.rename(desired_sass_path)
            except Exception:
                legacy.unlink(missing_ok=True)
        # If still absent, try compiling from .cu (handle multi-file projects)
        should_compile = cu_path.exists() and (force_rebuild or not desired_sass_path.exists())
        if should_compile:
            tmp_exe = test_dir / f"{testname}_sm{sm}.tmp.out"
            try:
                # For HeCBench, use Makefile if available (much more robust!)
                if suite_name == "hecbench":
                    makefile_path = test_dir / "Makefile"
                    if makefile_path.exists():
                        # Use the project's Makefile with our SM architecture
                        try:
                            # Clean any previous builds
                            run(["make", "clean"], cwd=test_dir)
                        except AssertionError:
                            pass  # Clean might fail if no previous build
                        
                        # Build with specified architecture 
                        run(["make", f"ARCH=sm_{sm}"], cwd=test_dir)
                        
                        # Find the built executable (common patterns)
                        possible_exes = list(test_dir.glob("*.out")) + list(test_dir.glob("main")) + list(test_dir.glob(f"{testname}"))
                        if possible_exes:
                            built_exe = possible_exes[0]  # Take the first match
                            sass_out = run(["cuobjdump", "--dump-sass", str(built_exe)], cwd=TEST_DIR)
                            desired_sass_path.write_text(ensure_arch_header(sass_out, sm))
                            
                            # Clean up the built executable
                            try:
                                built_exe.unlink()
                                run(["make", "clean"], cwd=test_dir)
                            except:
                                pass  # Best effort cleanup
                        else:
                            raise AssertionError("No executable found after make")
                    else:
                        # Fallback to manual compilation for HeCBench without Makefile
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
                    # Non-HeCBench: single file compilation
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
                if pytest:
                    pytest.skip(f"Cannot build SASS for {base} sm{sm}")
                else:
                    return
            finally:
                tmp_exe.unlink(missing_ok=True)

        if desired_sass_path.exists():
            sass_path = desired_sass_path
        else:
            if pytest:
                pytest.skip(f"No SASS or CU for {base} sm{sm}")
            else:
                return

    if sass_path:
        sass_rel = str(sass_path.relative_to(TEST_DIR))
        entry["sass_rel"] = sass_rel


    # Extract testname from base for output naming
    testname = base.split('/')[-1]
    out_base = f"{testname}_sm{sm}"

    # Create symbolic link for the correct kernel_wrapper variant
    kernel_wrapper_link = TEST_DIR / "common" / "kernel_wrapper.h"
    kernel_wrapper_variant = TEST_DIR / "common" / f"kernel_wrapper_{sm}.h"

    link_created = False
    if kernel_wrapper_variant.exists():
        # Remove existing link or file if it exists
        if kernel_wrapper_link.exists() or kernel_wrapper_link.is_symlink():
            kernel_wrapper_link.unlink()
        # Create symbolic link using a relative target within common/
        kernel_wrapper_link.symlink_to(kernel_wrapper_variant.name)
        link_created = True

    # Track generated build artifacts to remove even if the test fails
    clean_paths: List[Path] = []

    try:
        # 1) lift to LLVM
        if sass_rel:
            sass_path = Path(sass_rel)
            test_subdir = sass_path.parent

            # Output SASS compilation in test subfolder with new naming
            out_ll_path = test_subdir / f"{out_base}.ll"
            out_o_path = test_subdir / f"{out_base}.o"
            # Keep generated .ll files after tests to aid debugging

            run(["python", "../main.py", "-i", f"{sass_rel}", "-o", str(out_ll_path.with_suffix(''))], cwd=TEST_DIR)

            # 2) link & execute when host present; otherwise lifting success is enough
            should_run_host = bool(host_rel)

            # Enforce host existence for cuobjdump suite
            if suite_name == "cuobjdump" and not host_rel:
                if pytest:
                    pytest.skip(f"Broken cuobjdump test (missing host): {base} sm{sm}")
                else:
                    return

            # For HeCBench, only do lifting for now (no host execution until host wrappers are created)
            if suite_name == "hecbench":
                should_run_host = False

            if should_run_host:
                # compile LLVM IR to object file
                run(["clang", "-O2", "-target", "x86_64", "-c", str(out_ll_path), "-o", str(out_o_path)], cwd=TEST_DIR)
                exe_path = test_subdir / f"{out_base}.out"
                # mark .o and .out for cleanup
                clean_paths.append(TEST_DIR / out_o_path)
                clean_paths.append(TEST_DIR / exe_path)

                # Add arch define for host
                cu_flags: List[str] = [f"-D__CUDA_ARCH__={sm}0"]

                run([
                    "clang++",
                    "-std=c++20",
                    "-O2",
                    *cu_flags,
                    f"./{host_rel}",
                    str(out_o_path),
                    "-I./common",
                    "-o",
                    str(exe_path),
                ], cwd=TEST_DIR)

                out = run([str(exe_path)], cwd=TEST_DIR)
                assert "TEST PASSED" in out

            

        else:
            if pytest:
                pytest.skip(f"Test {base} sm{sm}: no executable configuration")
    finally:
        # Always try to cleanup generated artifacts, even on failure
        for p in clean_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        # Always cleanup symbolic link if we created it
        if link_created and (kernel_wrapper_link.exists() or kernel_wrapper_link.is_symlink()):
            kernel_wrapper_link.unlink()
