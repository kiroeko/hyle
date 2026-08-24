#!/usr/bin/env python3
"""Certify mem's shipping same-link standard adapters in fresh processes.

Section L of hyle_mem_bench intentionally runs one variant per process.  This
driver alternates forward/reverse order, records every raw value, and prints the
paired medians used by docs/performance/benchmarks-mem.md.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


VARIANTS = (
    ("B", "backend_raw_same_link"),
    ("S", "system_allocator_direct"),
    ("A", "allocator_runtime"),
    ("P", "pmr_direct_backend"),
    ("M", "system_memory_resource"),
    ("R", "pmr_resource_runtime"),
    ("T", "system_std_allocator"),
    ("D", "std_allocator_runtime"),
    ("E", "backend_reallocate_same_link"),
    ("F", "system_allocator_reallocate"),
    ("G", "allocator_runtime_reallocate"),
)

COMPARISONS = (
    ("system_allocator_direct", "backend_raw_same_link"),
    ("system_std_allocator", "system_allocator_direct"),
    ("allocator_runtime", "backend_raw_same_link"),
    ("allocator_runtime", "pmr_direct_backend"),
    ("std_allocator_runtime", "allocator_runtime"),
    ("system_memory_resource", "pmr_direct_backend"),
    ("pmr_resource_runtime", "pmr_direct_backend"),
    ("system_allocator_reallocate", "backend_reallocate_same_link"),
    ("allocator_runtime_reallocate", "backend_reallocate_same_link"),
)

ROW_RE = re.compile(r"^\|\s+[0-9.]+%\s+\|\s+([0-9,.]+)\s+\|", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--linkage", choices=("shared", "static"), default="shared")
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--cpu", type=int, help="pin each child process where sched_setaffinity exists")
    parser.add_argument("--apple-qos", choices=("interactive", "utility"))
    parser.add_argument("--prefix", action="append", default=[], help="launcher argv item; repeat as needed")
    return parser.parse_args()


def git_value(source_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_dir), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def child_affinity(cpu: int | None):
    if cpu is None or not hasattr(os, "sched_setaffinity"):
        return None

    def apply() -> None:
        os.sched_setaffinity(0, {cpu})

    return apply


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    executable = args.executable.resolve()
    if not executable.is_file():
        raise SystemExit(f"benchmark executable not found: {executable}")

    commit = git_value(args.source_dir, "rev-parse", "HEAD")
    dirty = "1" if git_value(args.source_dir, "status", "--porcelain") else "0"
    values: dict[str, list[float]] = {name: [] for _, name in VARIANTS}
    rows: list[dict[str, str | int | float]] = []
    env = os.environ.copy()
    env["HYLE_BENCH_SECTIONS"] = "L"
    if args.apple_qos:
        env["HYLE_BENCH_APPLE_QOS"] = args.apple_qos

    for run in range(args.runs):
        ordered = VARIANTS if run % 2 == 0 else tuple(reversed(VARIANTS))
        for variant_id, variant_name in ordered:
            env["HYLE_MEM_LINK_VARIANT"] = variant_id
            command = [*args.prefix, str(executable)]
            result = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                preexec_fn=child_affinity(args.cpu),
            )
            output = result.stdout + result.stderr
            if result.returncode != 0:
                sys.stderr.write(output)
                raise SystemExit(f"run {run} variant {variant_id} failed: {result.returncode}")
            if f"HYLE_MEM_LINKAGE {args.linkage}" not in output:
                raise SystemExit(
                    f"certificate requires a {args.linkage} hyle_mem build"
                )
            if "HYLE_MEM_BUILD_MODE shipping" not in output:
                raise SystemExit("certificate requires HYLE_TESTING=OFF")
            if args.apple_qos and (
                f"HYLE_MEM_LINK_QOS requested={args.apple_qos} qos_ok=1" not in output
            ):
                sys.stderr.write(output)
                raise SystemExit("requested Apple QoS was not applied")
            matches = ROW_RE.findall(output)
            if len(matches) != 1:
                sys.stderr.write(output)
                raise SystemExit(
                    f"run {run} variant {variant_id}: expected one nanobench row, got {len(matches)}"
                )
            ns_per_op = float(matches[0].replace(",", ""))
            values[variant_name].append(ns_per_op)
            rows.append(
                {
                    "commit": commit,
                    "dirty": dirty,
                    "compiler": args.compiler,
                    "platform": args.platform,
                    "qos": args.apple_qos or "n/a",
                    "linkage": args.linkage,
                    "run": run,
                    "variant": variant_name,
                    "ns_per_op": ns_per_op,
                }
            )

    args.raw.parent.mkdir(parents=True, exist_ok=True)
    with args.raw.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"commit={commit} dirty={dirty} platform={args.platform} n={args.runs}")
    for name, samples in values.items():
        print(f"{name:30s} median={statistics.median(samples):.3f} ns/op")
    print("paired throughput ratios (reference_ns / candidate_ns):")
    for candidate, reference in COMPARISONS:
        ratios = [r / c for c, r in zip(values[candidate], values[reference])]
        wins = sum(ratio > 1.0 for ratio in ratios)
        print(
            f"{candidate:30s} vs {reference:28s} "
            f"median={statistics.median(ratios):.3f} sign={wins}/{args.runs}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
