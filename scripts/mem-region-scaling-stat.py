#!/usr/bin/env python3
"""Repeat the long-lived Region retained-block scaling certificate.

This is deliberately separate from short request latency: it keeps one Region
alive, repeatedly clear/reuses 32..2048 retained blocks, and aggregates across
processes. Linear total time / bounded ns-per-allocation is the release gate.
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys


ROW = re.compile(
    r"^HYLE_MEM_CERT region_scaling scenario=(\S+) "
    r"retained_blocks=(\d+) samples=(\d+) allocations_per_epoch=(\d+) "
    r"p50_ns=(\d+) p95_ns=(\d+) p99_ns=(\d+) "
    r"p50_ns_per_allocation=([0-9.]+)$"
)
META = re.compile(
    r"^HYLE_MEM_CERT meta format=(\d+) compiler=(\S+) platform=(\S+)$"
)
APPLE_RUNTIME = re.compile(
    r"^# HYLE_MEM_BENCH apple_main_qos=(\S+) qos_ok=(\d+) affinity=(\S+)$"
)


def parse(text):
    meta = None
    runtime = None
    rows = {}
    for raw in text.splitlines():
        line = raw.strip()
        match = META.match(line)
        if match:
            meta = {
                "format": int(match.group(1)),
                "compiler": match.group(2),
                "platform": match.group(3),
            }
            continue
        match = APPLE_RUNTIME.match(line)
        if match:
            runtime = {
                "qos": match.group(1),
                "qos_ok": int(match.group(2)),
                "affinity": match.group(3),
            }
            continue
        match = ROW.match(line)
        if match:
            blocks = int(match.group(2))
            rows[blocks] = {
                "scenario": match.group(1),
                "retained_blocks": blocks,
                "samples": int(match.group(3)),
                "allocations_per_epoch": int(match.group(4)),
                "p50_ns": int(match.group(5)),
                "p95_ns": int(match.group(6)),
                "p99_ns": int(match.group(7)),
                "p50_ns_per_allocation": float(match.group(8)),
            }
    return meta, runtime, rows


def git_metadata(repo):
    try:
        head = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", repo, "status", "--porcelain"], text=True
            ).strip()
        )
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def main():
    parser = argparse.ArgumentParser(
        description="aggregate long-lived HYLE Region scaling across processes"
    )
    parser.add_argument("exe")
    parser.add_argument("-n", "--runs", type=int, default=31)
    parser.add_argument("--samples", type=int, default=10001)
    parser.add_argument("--cpu", help="Linux taskset CPU/list")
    parser.add_argument("--raw", help="write parsed rows as CSV")
    args = parser.parse_args()
    if args.runs < 3 or args.samples < 1000:
        parser.error("runs must be >= 3 and samples must be >= 1000")

    exe = os.path.abspath(args.exe)
    env = os.environ.copy()
    env["HYLE_BENCH_SECTIONS"] = "J"
    env["HYLE_MEM_REGION_SAMPLES"] = str(args.samples)
    command = [exe]
    if args.cpu:
        command = ["taskset", "-c", args.cpu, exe]

    all_rows = []
    common_meta = None
    expected = None
    for index in range(args.runs):
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            sys.exit(
                f"run {index} exited {result.returncode}: "
                f"{(result.stderr or result.stdout)[:500]}"
            )
        meta, runtime, rows = parse(result.stdout + result.stderr)
        if meta is None or meta["format"] != 1 or not rows:
            sys.exit(f"run {index}: missing metadata or scaling rows")
        if meta["platform"] == "macos":
            if runtime is None or runtime["qos_ok"] != 1:
                sys.exit(f"run {index}: Apple QoS metadata missing/unsuccessful")
            meta.update(runtime)
        else:
            meta.update({"qos": "n/a", "qos_ok": 1, "affinity": "external"})
        if common_meta is not None and meta != common_meta:
            sys.exit(f"run {index}: metadata changed across runs")
        keys = set(rows)
        if expected is not None and keys != expected:
            sys.exit(f"run {index}: retained-block row set changed")
        common_meta = meta
        expected = keys
        all_rows.append(rows)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit, dirty = git_metadata(repo)
    print(
        f"commit={commit} dirty={int(dirty)} compiler={common_meta['compiler']} "
        f"platform={common_meta['platform']} cpu={args.cpu or 'unbound'} "
        f"qos={common_meta['qos']} affinity={common_meta['affinity']} "
        f"runs={args.runs} epochs_per_run={args.samples}"
    )

    medians = {}
    for blocks in sorted(expected):
        p50 = [run[blocks]["p50_ns"] for run in all_rows]
        p95 = [run[blocks]["p95_ns"] for run in all_rows]
        p99 = [run[blocks]["p99_ns"] for run in all_rows]
        per_alloc = [run[blocks]["p50_ns_per_allocation"] for run in all_rows]
        medians[blocks] = statistics.median(per_alloc)
        print(
            f"blocks={blocks:4d} p50_ns={statistics.median(p50):.0f}"
            f"[{min(p50)},{max(p50)}] "
            f"p95_ns={statistics.median(p95):.0f}[{min(p95)},{max(p95)}] "
            f"p99_ns={statistics.median(p99):.0f}[{min(p99)},{max(p99)}] "
            f"p50_ns_per_allocation={medians[blocks]:.3f}"
            f"[{min(per_alloc):.3f},{max(per_alloc):.3f}]"
        )
    smallest, largest = min(medians), max(medians)
    normalized_growth = medians[largest] / medians[smallest]
    print(
        f"normalized_cost_ratio_{largest}_over_{smallest}={normalized_growth:.3f} "
        "(near 1 is linear total work; quadratic traversal grows with blocks)"
    )

    if args.raw:
        path = os.path.abspath(args.raw)
        fields = [
            "commit", "dirty", "compiler", "platform", "cpu", "qos",
            "qos_ok", "affinity", "run", "scenario", "retained_blocks",
            "samples", "allocations_per_epoch", "p50_ns", "p95_ns",
            "p99_ns", "p50_ns_per_allocation",
        ]
        with open(path, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for index, run in enumerate(all_rows):
                for blocks in sorted(run):
                    writer.writerow(
                        {
                            "commit": commit,
                            "dirty": int(dirty),
                            "compiler": common_meta["compiler"],
                            "platform": common_meta["platform"],
                            "cpu": args.cpu or "unbound",
                            "qos": common_meta["qos"],
                            "qos_ok": common_meta["qos_ok"],
                            "affinity": common_meta["affinity"],
                            "run": index,
                            **run[blocks],
                        }
                    )
        print(f"raw={path}")


if __name__ == "__main__":
    main()
