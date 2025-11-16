#!/usr/bin/env python3
"""
Benchmark selected cuobjdump tests using the SLifter x86 and NVVM pipelines.

Usage:
    python scripts/benchmark_cuobjdump.py \
        --bench-dir benchmark_output \
        --sm 75 \
        --runs 3 \
        --csv benchmark_output/benchmark_results.csv

The script assumes:
  * benchmark_dir contains per-test folders copied from test/cuobjdump/<name>.
  * Each folder has <test>_sm<SM>.sass plus host_cpu.cpp and host_cuda.cu.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
TEST_COMMON_DIR = ROOT / "test" / "common"
MAIN_PY = ROOT / "main.py"


class CommandError(RuntimeError):
    """Raised when an external command fails."""

    def __init__(self, cmd: List[str], result: subprocess.CompletedProcess):
        joined = " ".join(str(p) for p in cmd)
        msg = (
            f"command failed: {joined}\n"
            f"retcode={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
        )
        super().__init__(msg)
        self.result = result


def run_cmd(cmd: List[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command and raise if it fails."""
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CommandError(cmd, result)
    return result


def discover_tests(bench_dir: Path, sm: str) -> List[Tuple[str, Path, Path]]:
    """Return (label, test_dir, sass_path) tuples for each benchmark folder."""
    tests: List[Tuple[str, Path, Path]] = []
    for entry in sorted(bench_dir.iterdir()):
        if not entry.is_dir():
            continue
        sass_candidates = sorted(entry.glob(f"*sm{sm}.sass"))
        if not sass_candidates:
            continue
        tests.append((entry.name, entry, sass_candidates[0]))
    return tests


def ensure_kernel_wrapper(sm: str, include_dir: Path) -> None:
    """Copy the SM-specific kernel wrapper to the include_dir as kernel_wrapper.h."""
    src = TEST_COMMON_DIR / f"kernel_wrapper_{sm}.h"
    if not src.exists():
        raise FileNotFoundError(f"Missing kernel wrapper header: {src}")
    include_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, include_dir / "kernel_wrapper.h")


def build_x86(test_dir: Path, sass_path: Path, sm: str) -> Dict[str, Path]:
    """Build the x86 pipeline artifacts and return paths."""
    host_cpu = test_dir / "host_cpu.cpp"
    if not host_cpu.exists():
        raise FileNotFoundError(f"{host_cpu} missing")

    build_dir = test_dir / "build" / "x86"
    build_dir.mkdir(parents=True, exist_ok=True)
    stem = sass_path.stem
    prefix = build_dir / f"{stem}_x86"
    ll_path = prefix.with_suffix(".ll")
    obj_path = prefix.with_suffix(".o")
    exe_path = prefix.with_suffix(".out")

    run_cmd(
        [
            sys.executable,
            str(MAIN_PY),
            "-i",
            str(sass_path),
            "-o",
            str(prefix),
            "--lifter",
            "x86",
        ],
        cwd=ROOT,
    )

    run_cmd(
        [
            "clang",
            "-O2",
            "-target",
            "x86_64",
            "-c",
            str(ll_path),
            "-o",
            str(obj_path),
        ],
        cwd=ROOT,
    )

    include_dir = build_dir / "include"
    ensure_kernel_wrapper(sm, include_dir)

    run_cmd(
        [
            "clang++",
            "-std=c++20",
            "-O2",
            f"-D__CUDA_ARCH__={sm}0",
            f"-I{include_dir}",
            f"-I{TEST_COMMON_DIR}",
            str(host_cpu),
            str(obj_path),
            "-o",
            str(exe_path),
        ],
        cwd=ROOT,
    )

    return {"exe": exe_path}


def build_nvvm(test_dir: Path, sass_path: Path, sm: str) -> Dict[str, Path]:
    """Build the NVVM pipeline artifacts and return paths."""
    host_cuda = test_dir / "host_cuda.cu"
    if not host_cuda.exists():
        raise FileNotFoundError(f"{host_cuda} missing")

    build_dir = test_dir / "build" / "nvvm"
    build_dir.mkdir(parents=True, exist_ok=True)
    stem = sass_path.stem
    prefix = build_dir / f"{stem}_nvvm"
    ll_path = prefix.with_suffix(".ll")
    ptx_path = prefix.with_suffix(".ptx")
    cubin_path = prefix.with_suffix(".cubin")
    exe_path = prefix.with_suffix(".out")

    run_cmd(
        [
            sys.executable,
            str(MAIN_PY),
            "-i",
            str(sass_path),
            "-o",
            str(prefix),
            "--lifter",
            "nvvm",
        ],
        cwd=ROOT,
    )

    run_cmd(
        [
            "llc",
            "-march=nvptx64",
            f"-mcpu=sm_{sm}",
            f"-mattr=+ptx{sm}",
            "-o",
            str(ptx_path),
            str(ll_path),
        ],
        cwd=ROOT,
    )

    # Produce a cubin via ptxas when available; otherwise fall back to PTX.
    device_image_path = ptx_path
    try:
        run_cmd(
            [
                "ptxas",
                f"-arch=sm_{sm}",
                str(ptx_path),
                "-o",
                str(cubin_path),
            ],
            cwd=ROOT,
        )
        device_image_path = cubin_path
    except CommandError as exc:
        print(
            f"[warn] ptxas failed for {test_dir.name}; using PTX directly\n{exc}",
            file=sys.stderr,
        )

    run_cmd(
        [
            "nvcc",
            "-std=c++17",
            f"-I{TEST_COMMON_DIR}",
            str(host_cuda),
            "-lcuda",
            "-o",
            str(exe_path),
        ],
        cwd=ROOT,
    )

    return {"exe": exe_path, "device_image": device_image_path}


def measure_execution(cmd: List[str], *, cwd: Path) -> Tuple[float, str]:
    """Run cmd and return (elapsed_seconds, stdout)."""
    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise CommandError(cmd, result)
    if "TEST PASSED" not in result.stdout:
        raise RuntimeError(
            f"benchmark run did not report success.\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return elapsed, result.stdout


def benchmark_tests(
    tests: Iterable[Tuple[str, Path, Path]],
    sm: str,
    runs: int,
) -> List[Dict[str, object]]:
    """Build and benchmark each test for both backends."""
    rows: List[Dict[str, object]] = []
    for label, test_dir, sass_path in tests:
        print(f"=== {label} ({sass_path.name}) ===")
        builders = {
            "x86": lambda: build_x86(test_dir, sass_path, sm),
            "nvvm": lambda: build_nvvm(test_dir, sass_path, sm),
        }

        artifacts: Dict[str, Dict[str, Path]] = {}
        for backend, builder in builders.items():
            print(f"  building {backend}…", end="", flush=True)
            artifacts[backend] = builder()
            print("done")

        for backend, info in artifacts.items():
            exe = info["exe"]
            cmd = [str(exe)]
            if backend == "nvvm":
                cmd.append(str(info["device_image"]))

            print(f"  running {backend} ({runs}x)…")
            for run_idx in range(1, runs + 1):
                elapsed, _ = measure_execution(cmd, cwd=ROOT)
                rows.append(
                    {
                        "test": label,
                        "backend": backend,
                        "run": run_idx,
                        "seconds": elapsed,
                    }
                )
                print(f"    run {run_idx}: {elapsed:.6f}s")
    return rows


def write_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test", "backend", "run", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")


def summarize(rows: List[Dict[str, object]]) -> None:
    summary: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        key = (row["test"], row["backend"])
        summary.setdefault(key, []).append(float(row["seconds"]))

    print("\nSummary (avg ± stddev over runs):")
    for (test, backend), samples in sorted(summary.items()):
        avg = sum(samples) / len(samples)
        variance = (
            sum((x - avg) ** 2 for x in samples) / len(samples) if len(samples) > 1 else 0.0
        )
        stddev = variance ** 0.5
        print(f"  {test:<20} {backend:>4}: {avg:.6f}s ± {stddev:.6f}s ({len(samples)} runs)")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cuobjdump tests.")
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=ROOT / "benchmark_output",
        help="Directory containing copied cuobjdump tests.",
    )
    parser.add_argument("--sm", default="75", help="SM version used for suffix lookup.")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs per backend.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "benchmark_output" / "benchmark_results.csv",
        help="Destination CSV path for benchmark timings.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    tests = discover_tests(args.bench_dir, args.sm)
    if not tests:
        print(f"No tests found in {args.bench_dir}", file=sys.stderr)
        return 1

    rows = benchmark_tests(tests, args.sm, args.runs)
    write_csv(rows, args.csv)
    summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
