#!/usr/bin/env python3
"""Aggregate request-level mem latency certificates across process runs.

The benchmark emits one p50/p95/p99 distribution per whole request. This tool
repeats the process, preserves paired HYLE/reference ratios, and can write the
raw rows used by docs/performance. Lower latency is better.
"""

import argparse
import csv
import math
import os
import re
import statistics
import subprocess
import sys


ROW = re.compile(
    r"^HYLE_MEM_CERT latency scenario=(\S+) variant=(\S+) "
    r"samples=(\d+) operations=(\d+) p50_ns=(\d+) p95_ns=(\d+) "
    r"p99_ns=(\d+)$"
)
META = re.compile(
    r"^HYLE_MEM_CERT meta format=(\d+) compiler=(\S+) platform=(\S+)$"
)
APPLE_RUNTIME = re.compile(
    r"^# HYLE_MEM_BENCH apple_main_qos=(\S+) qos_ok=(\d+) affinity=(\S+)$"
)

PAIRS = {
    "external_region": ("hyle_fixed_region", "pmr_reconstruct"),
    "owned_region": ("hyle_region", "pmr_release"),
    "fixed_pool_window": ("hyle_fixed_pool", "onetbb_fixed_pool"),
}


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return min(2.0 * tail / (2**n), 1.0)


def parse_rows(text):
    rows = {}
    for line in text.splitlines():
        match = ROW.match(line.strip())
        if not match:
            continue
        scenario, variant = match.group(1), match.group(2)
        rows[(scenario, variant)] = {
            "samples": int(match.group(3)),
            "operations": int(match.group(4)),
            "p50_ns": int(match.group(5)),
            "p95_ns": int(match.group(6)),
            "p99_ns": int(match.group(7)),
        }
    return rows


def parse_meta(text):
    for line in text.splitlines():
        match = META.match(line.strip())
        if match:
            return {
                "format": int(match.group(1)),
                "compiler": match.group(2),
                "platform": match.group(3),
            }
    return None


def parse_apple_runtime(text):
    for line in text.splitlines():
        match = APPLE_RUNTIME.match(line.strip())
        if match:
            return {
                "qos": match.group(1),
                "qos_ok": int(match.group(2)),
                "affinity": match.group(3),
            }
    return None


def git_metadata(repo):
    try:
        head = subprocess.check_output(
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
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def main():
    parser = argparse.ArgumentParser(
        description="aggregate HYLE mem request p50/p95/p99 across processes"
    )
    parser.add_argument("exe")
    parser.add_argument("-n", "--runs", type=int, default=15)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument(
        "--dll-dir", action="append", default=[], help="prepend a runtime DLL dir"
    )
    parser.add_argument("--cpu", help="Linux taskset CPU/list for every process")
    parser.add_argument("--raw", help="write every parsed row as CSV")
    args = parser.parse_args()
    if args.runs < 1 or args.samples < 100:
        parser.error("runs must be positive and samples must be >= 100")

    exe = os.path.abspath(args.exe)
    env = os.environ.copy()
    env["HYLE_BENCH_SECTIONS"] = "G"
    env["HYLE_MEM_CERT_SAMPLES"] = str(args.samples)
    if args.dll_dir:
        dirs = [os.path.abspath(path) for path in args.dll_dir]
        env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])

    command = [exe]
    if args.cpu:
        command = ["taskset", "-c", args.cpu, exe]

    runs = []
    metadata = None
    expected_keys = None
    for index in range(args.runs):
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            sys.exit(
                f"run {index} exited {result.returncode}: "
                f"{(result.stderr or result.stdout)[:500]}"
            )
        combined_output = result.stdout + result.stderr
        rows = parse_rows(combined_output)
        current_metadata = parse_meta(combined_output)
        if not rows:
            sys.exit(f"run {index}: no HYLE_MEM_CERT rows parsed")
        if current_metadata is None or current_metadata["format"] != 1:
            sys.exit(f"run {index}: missing or unsupported HYLE_MEM_CERT meta")
        if "HYLE_MEM_CERT build testing=0" not in combined_output:
            sys.exit(
                f"run {index}: benchmark is not a shipping build "
                "(HYLE_TESTING must be OFF)"
            )
        runtime = parse_apple_runtime(combined_output)
        if current_metadata["platform"] == "macos":
            if runtime is None or runtime["qos_ok"] != 1:
                sys.exit(f"run {index}: Apple QoS metadata missing or unsuccessful")
            current_metadata.update(runtime)
        else:
            current_metadata.update(
                {"qos": "n/a", "qos_ok": 1, "affinity": "external"}
            )
        if metadata is not None and current_metadata != metadata:
            sys.exit(f"run {index}: certificate metadata changed across runs")
        current_keys = set(rows)
        if expected_keys is not None and current_keys != expected_keys:
            missing = sorted(expected_keys - current_keys)
            extra = sorted(current_keys - expected_keys)
            sys.exit(
                f"run {index}: certificate row set changed; "
                f"missing={missing} extra={extra}"
            )
        metadata = current_metadata
        expected_keys = current_keys
        runs.append(rows)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit, dirty = git_metadata(repo)
    print(
        f"commit={commit} dirty={int(dirty)} compiler={metadata['compiler']} "
        f"platform={metadata['platform']} cpu={args.cpu or 'unbound'} "
        f"qos={metadata['qos']} qos_ok={metadata['qos_ok']} "
        f"affinity={metadata['affinity']} runs={args.runs} samples={args.samples}"
    )

    keys = sorted(expected_keys)
    metrics = ("p50_ns", "p95_ns", "p99_ns")
    for scenario, variant in keys:
        values = {
            metric: [run[(scenario, variant)][metric] for run in runs]
            for metric in metrics
        }
        summary = " ".join(
            f"{metric}={statistics.median(series):.0f}"
            f"[{min(series)},{max(series)}]"
            for metric, series in values.items()
        )
        print(f"{scenario:20s} {variant:20s} {summary}")

    print("\npaired latency ratios (reference/HYLE; >1 favors HYLE)")
    for scenario, (hyle, reference) in PAIRS.items():
        if (scenario, hyle) not in keys or (scenario, reference) not in keys:
            continue
        for metric in metrics:
            ratios = [
                run[(scenario, reference)][metric] / run[(scenario, hyle)][metric]
                for run in runs
            ]
            above = sum(value > 1.0 for value in ratios)
            below = sum(value < 1.0 for value in ratios)
            effective = above + below
            p_value = binom_two_sided(max(above, below), effective)
            median = statistics.median(ratios)
            gap = abs(median - 1.0)
            if effective >= 6 and p_value < 0.01 and gap > 0.01:
                verdict = "WIN" if above > below else "LOSE"
            else:
                verdict = "TIE"
            print(
                f"  {scenario:20s} {metric}: med={median:.3f} "
                f"range={min(ratios):.3f}-{max(ratios):.3f} "
                f"{above}/{effective}>1 p={p_value:.1e} {verdict}"
            )

    if args.raw:
        raw_path = os.path.abspath(args.raw)
        with open(raw_path, "w", newline="", encoding="utf-8") as output:
            fields = [
                "commit",
                "dirty",
                "compiler",
                "platform",
                "cpu",
                "qos",
                "qos_ok",
                "affinity",
                "run",
                "scenario",
                "variant",
                "samples",
                "operations",
                "p50_ns",
                "p95_ns",
                "p99_ns",
            ]
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for run_index, run in enumerate(runs):
                for (scenario, variant), row in sorted(run.items()):
                    writer.writerow(
                        {
                            "commit": commit,
                            "dirty": int(dirty),
                            "compiler": metadata["compiler"],
                            "platform": metadata["platform"],
                            "cpu": args.cpu or "unbound",
                            "qos": metadata["qos"],
                            "qos_ok": metadata["qos_ok"],
                            "affinity": metadata["affinity"],
                            "run": run_index,
                            "scenario": scenario,
                            "variant": variant,
                            **row,
                        }
                    )
        print(f"raw={raw_path}")


if __name__ == "__main__":
    main()
