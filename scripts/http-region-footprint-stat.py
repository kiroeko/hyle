#!/usr/bin/env python3
"""Measure request-Region retention in isolated real HTTP processes."""

import argparse
import csv
import os
import statistics
import subprocess
import sys


PREFIX = "HYLE_HTTP_MEM_CERT footprint "
VARIANTS = ("capped", "capped_collect", "unbounded")
NUMERIC_FIELDS = (
    "connections",
    "spike_bytes_per_connection",
    "max_block_bytes",
    "retain_limit_bytes",
    "baseline_rss_bytes",
    "warm_rss_bytes",
    "spike_rss_bytes",
    "reuse_rss_bytes",
    "release_rss_bytes",
    "process_peak_rss_bytes",
    "warm_backing_bytes",
    "spike_backing_bytes",
    "reuse_backing_bytes",
    "peak_backing_bytes",
    "release_backing_bytes",
    "collect_ns",
)


def parse_key_values(line):
    values = {}
    for item in line.split():
        if "=" in item:
            key, value = item.split("=", 1)
            values[key] = value
    return values


def parse_row(text):
    matches = [line for line in text.splitlines() if line.startswith(PREFIX)]
    if len(matches) != 1:
        return None
    values = parse_key_values(matches[0])
    if values.get("scenario") != "http_request_region":
        return None
    try:
        return {
            "scenario": values["scenario"],
            "variant": values["variant"],
            **{field: int(values[field]) for field in NUMERIC_FIELDS},
        }
    except (KeyError, ValueError):
        return None


def parse_meta(text):
    for line in text.splitlines():
        if line.startswith("HYLE_HTTP_MEM_CERT meta "):
            values = parse_key_values(line)
            try:
                return {
                    "format": int(values["format"]),
                    "compiler": values["compiler"],
                    "platform": values["platform"],
                }
            except (KeyError, ValueError):
                return None
    return None


def parse_apple_runtime(text):
    for line in text.splitlines():
        if line.startswith("# HYLE_HTTP_MEM_BENCH apple_main_qos="):
            values = parse_key_values(line)
            try:
                return {
                    "qos": values["apple_main_qos"],
                    "qos_ok": int(values["qos_ok"]),
                    "affinity": values["affinity"],
                }
            except (KeyError, ValueError):
                return None
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


def percentile(values, quantile):
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summary(values):
    return (
        f"p50={statistics.median(values):.0f} "
        f"p99={percentile(values, 0.99):.0f} "
        f"range=[{min(values)},{max(values)}]"
    )


def main():
    parser = argparse.ArgumentParser(
        description="measure capped vs unbounded request Region in real HTTP"
    )
    parser.add_argument("exe")
    parser.add_argument("-n", "--runs", type=int, default=15)
    parser.add_argument("--connections", type=int, default=128)
    parser.add_argument("--spike-kib", type=int, default=1024)
    parser.add_argument("--block-kib", type=int, default=64)
    parser.add_argument(
        "--dll-dir", action="append", default=[], help="prepend a runtime DLL dir"
    )
    parser.add_argument("--cpu", help="Linux taskset CPU/list for every process")
    parser.add_argument("--raw", help="write every parsed row as CSV")
    args = parser.parse_args()
    if (
        args.runs < 1
        or args.connections < 2
        or args.spike_kib < 256
        or args.block_kib < 64
    ):
        parser.error(
            "runs must be positive, connections >= 2, spike-kib >= 256, "
            "block-kib >= 64"
        )

    executable = os.path.abspath(args.exe)
    base_env = os.environ.copy()
    base_env["HYLE_HTTP_REGION_CONNECTIONS"] = str(args.connections)
    base_env["HYLE_HTTP_REGION_SPIKE_KIB"] = str(args.spike_kib)
    base_env["HYLE_HTTP_REGION_BLOCK_KIB"] = str(args.block_kib)
    if args.dll_dir:
        directories = [os.path.abspath(path) for path in args.dll_dir]
        base_env["PATH"] = os.pathsep.join(
            directories + [base_env.get("PATH", "")]
        )
    command = [executable]
    if args.cpu:
        command = ["taskset", "-c", args.cpu, executable]

    runs = []
    metadata = None
    for run_index in range(args.runs):
        run = {}
        order = VARIANTS if run_index % 2 == 0 else tuple(reversed(VARIANTS))
        for variant in order:
            environment = base_env.copy()
            environment["HYLE_HTTP_REGION_FOOTPRINT_VARIANT"] = variant
            result = subprocess.run(
                command, capture_output=True, text=True, env=environment
            )
            output = result.stdout + result.stderr
            if result.returncode != 0:
                sys.exit(
                    f"run {run_index} {variant} exited {result.returncode}: "
                    f"{(result.stderr or result.stdout)[:1000]}"
                )
            row = parse_row(output)
            current_metadata = parse_meta(output)
            if row is None or row["variant"] != variant:
                sys.exit(f"run {run_index} {variant}: invalid footprint row")
            if current_metadata is None or current_metadata["format"] != 1:
                sys.exit(f"run {run_index} {variant}: invalid certificate metadata")
            runtime = parse_apple_runtime(output)
            if current_metadata["platform"] == "macos":
                if runtime is None or runtime["qos_ok"] != 1:
                    sys.exit(f"run {run_index} {variant}: Apple QoS audit failed")
                current_metadata.update(runtime)
            else:
                current_metadata.update(
                    {"qos": "n/a", "qos_ok": 1, "affinity": "external"}
                )
            if metadata is not None and current_metadata != metadata:
                sys.exit("certificate metadata changed across child processes")
            if row["release_backing_bytes"] != 0:
                sys.exit(f"run {run_index} {variant}: server backing remains live")
            metadata = current_metadata
            run[variant] = row
        runs.append(run)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit, dirty = git_metadata(repo)
    print(
        f"commit={commit} dirty={int(dirty)} compiler={metadata['compiler']} "
        f"platform={metadata['platform']} cpu={args.cpu or 'unbound'} "
        f"qos={metadata['qos']} qos_ok={metadata['qos_ok']} "
        f"affinity={metadata['affinity']} runs={args.runs} "
        f"connections={args.connections} spike_kib={args.spike_kib} "
        f"block_kib={args.block_kib}"
    )
    for variant in VARIANTS:
        rows = [run[variant] for run in runs]
        print(variant)
        measurements = (
            (
                "rss_warm_minus_baseline",
                [row["warm_rss_bytes"] - row["baseline_rss_bytes"] for row in rows],
            ),
            (
                "rss_spike_minus_warm",
                [row["spike_rss_bytes"] - row["warm_rss_bytes"] for row in rows],
            ),
            (
                "rss_reuse_minus_warm",
                [row["reuse_rss_bytes"] - row["warm_rss_bytes"] for row in rows],
            ),
            (
                "rss_release_minus_baseline",
                [
                    row["release_rss_bytes"] - row["baseline_rss_bytes"]
                    for row in rows
                ],
            ),
            ("backing_warm", [row["warm_backing_bytes"] for row in rows]),
            ("backing_spike", [row["spike_backing_bytes"] for row in rows]),
            ("backing_reuse", [row["reuse_backing_bytes"] for row in rows]),
            ("backing_peak", [row["peak_backing_bytes"] for row in rows]),
            ("collect_ns", [row["collect_ns"] for row in rows]),
        )
        for name, values in measurements:
            print(f"  {name:28s} {summary(values)}")

    if args.raw:
        raw_path = os.path.abspath(args.raw)
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
            *NUMERIC_FIELDS,
        ]
        with open(raw_path, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for run_index, run in enumerate(runs):
                for variant in VARIANTS:
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
                            **run[variant],
                        }
                    )
        print(f"raw={raw_path}")


if __name__ == "__main__":
    main()
