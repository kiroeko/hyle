#!/usr/bin/env python3
"""Aggregate scheduler-owned FiberStackPool latency certificates."""

import argparse
import csv
import math
import os
import re
import statistics
import subprocess
import sys


META = re.compile(
    r"^HYLE_MEM_FIBER_CERT meta format=(\d+) compiler=(\S+) platform=(\S+)$"
)
ROW = re.compile(
    r"^HYLE_MEM_FIBER_CERT latency variant=(\S+) samples=(\d+) fibers=(\d+) "
    r"p50_batch_ns=(\d+) p95_batch_ns=(\d+) p99_batch_ns=(\d+)$"
)
APPLE = re.compile(
    r"^# HYLE_MEM_FIBER_BENCH apple_main_qos=(\S+) qos_ok=(\d+) "
    r"affinity=(\S+)$"
)
VARIANTS = {"scheduler_pool", "direct_vm"}
METRICS = ("p50_batch_ns", "p95_batch_ns", "p99_batch_ns")


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    return min(2.0 * sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n), 1.0)


def parse_output(text):
    metadata = None
    runtime = None
    rows = {}
    for line in text.splitlines():
        line = line.strip()
        if match := META.match(line):
            metadata = {
                "format": int(match.group(1)),
                "compiler": match.group(2),
                "platform": match.group(3),
            }
        elif match := APPLE.match(line):
            runtime = {
                "qos": match.group(1),
                "qos_ok": int(match.group(2)),
                "affinity": match.group(3),
            }
        elif match := ROW.match(line):
            rows[match.group(1)] = {
                "samples": int(match.group(2)),
                "fibers": int(match.group(3)),
                "p50_batch_ns": int(match.group(4)),
                "p95_batch_ns": int(match.group(5)),
                "p99_batch_ns": int(match.group(6)),
            }
    return metadata, runtime, rows


def git_metadata(repo):
    try:
        commit = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", repo, "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def main():
    parser = argparse.ArgumentParser(
        description="aggregate FiberStackPool scheduler workload latency"
    )
    parser.add_argument("exe")
    parser.add_argument("-n", "--runs", type=int, default=15)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--dll-dir", action="append", default=[])
    parser.add_argument("--cpu", help="Linux taskset CPU/list")
    parser.add_argument("--raw", help="write every process row as CSV")
    args = parser.parse_args()
    if args.runs < 1 or args.samples < 100:
        parser.error("runs must be positive and samples must be >= 100")

    executable = os.path.abspath(args.exe)
    environment = os.environ.copy()
    environment["HYLE_MEM_FIBER_SAMPLES"] = str(args.samples)
    if args.dll_dir:
        directories = [os.path.abspath(path) for path in args.dll_dir]
        environment["PATH"] = os.pathsep.join(
            directories + [environment.get("PATH", "")]
        )
    command = [executable]
    if args.cpu:
        command = ["taskset", "-c", args.cpu, executable]

    runs = []
    metadata = None
    for index in range(args.runs):
        result = subprocess.run(
            command, capture_output=True, text=True, env=environment
        )
        if result.returncode != 0:
            sys.exit(
                f"run {index} exited {result.returncode}: "
                f"{(result.stderr or result.stdout)[:500]}"
            )
        current, runtime, rows = parse_output(result.stdout + result.stderr)
        if current is None or current["format"] != 1:
            sys.exit(f"run {index}: missing or unsupported certificate metadata")
        if set(rows) != VARIANTS:
            sys.exit(f"run {index}: expected {sorted(VARIANTS)}, got {sorted(rows)}")
        if current["platform"] == "macos":
            if runtime is None or runtime["qos_ok"] != 1:
                sys.exit(f"run {index}: Apple QoS metadata missing or unsuccessful")
            current.update(runtime)
        else:
            current.update({"qos": "n/a", "qos_ok": 1, "affinity": "external"})
        if metadata is not None and current != metadata:
            sys.exit(f"run {index}: metadata changed across runs")
        metadata = current
        runs.append(rows)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit, dirty = git_metadata(repo)
    print(
        f"commit={commit} dirty={int(dirty)} compiler={metadata['compiler']} "
        f"platform={metadata['platform']} cpu={args.cpu or 'unbound'} "
        f"qos={metadata['qos']} qos_ok={metadata['qos_ok']} "
        f"affinity={metadata['affinity']} runs={args.runs} samples={args.samples}"
    )

    for variant in sorted(VARIANTS):
        summary = " ".join(
            f"{metric}={statistics.median(run[variant][metric] for run in runs):.0f}"
            f"[{min(run[variant][metric] for run in runs)},"
            f"{max(run[variant][metric] for run in runs)}]"
            for metric in METRICS
        )
        print(f"{variant:16s} {summary}")

    print("\npaired latency ratios (direct_vm/scheduler_pool; >1 favors pool)")
    for metric in METRICS:
        ratios = [
            run["direct_vm"][metric] / run["scheduler_pool"][metric]
            for run in runs
        ]
        above = sum(value > 1.0 for value in ratios)
        below = sum(value < 1.0 for value in ratios)
        effective = above + below
        p_value = binom_two_sided(max(above, below), effective)
        median = statistics.median(ratios)
        verdict = (
            "WIN"
            if effective >= 6 and p_value < 0.01 and median > 1.01
            else "TIE"
        )
        print(
            f"  {metric}: med={median:.3f} "
            f"range={min(ratios):.3f}-{max(ratios):.3f} "
            f"{above}/{effective}>1 p={p_value:.1e} {verdict}"
        )

    if args.raw:
        with open(args.raw, "w", newline="", encoding="utf-8") as output:
            fields = [
                "run", "commit", "dirty", "compiler", "platform", "cpu",
                "qos", "qos_ok", "affinity", "variant", "samples", "fibers",
                *METRICS,
            ]
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for index, run in enumerate(runs):
                for variant in sorted(VARIANTS):
                    writer.writerow(
                        {
                            "run": index,
                            "commit": commit,
                            "dirty": int(dirty),
                            "compiler": metadata["compiler"],
                            "platform": metadata["platform"],
                            "cpu": args.cpu or "unbound",
                            "qos": metadata["qos"],
                            "qos_ok": metadata["qos_ok"],
                            "affinity": metadata["affinity"],
                            "variant": variant,
                            **run[variant],
                        }
                    )


if __name__ == "__main__":
    main()
