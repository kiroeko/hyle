#!/usr/bin/env python3
"""Run isolated HYLE mem footprint variants and emit commit-bound evidence."""

import argparse
import csv
import os
import statistics
import subprocess
import sys


PREFIX = "HYLE_MEM_CERT footprint "
NUMERIC_FIELDS = (
    "requested_bytes",
    "reuse_requested_bytes",
    "region_max_block_bytes",
    "peak_reserved_bytes",
    "retained_after_clear_bytes",
    "reuse_reserved_bytes",
    "released_bytes",
    "baseline_rss_bytes",
    "live_rss_bytes",
    "clear_rss_bytes",
    "reuse_rss_bytes",
    "release_rss_bytes",
    "process_peak_rss_bytes",
)
VARIANTS = ("hyle_region", "pmr_monotonic")


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
    if values.get("scenario") != "request_scratch":
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
        if line.startswith("HYLE_MEM_CERT meta "):
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
        if line.startswith("# HYLE_MEM_BENCH apple_main_qos="):
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


def delta(row, field, baseline="baseline_rss_bytes"):
    return row[field] - row[baseline]


def median_range(values):
    return f"{statistics.median(values):.0f}[{min(values)},{max(values)}]"


def main():
    parser = argparse.ArgumentParser(
        description="measure request-scratch reservation and RSS in fresh processes"
    )
    parser.add_argument("exe")
    parser.add_argument("-n", "--runs", type=int, default=15)
    parser.add_argument("--mib", type=int, default=128, help="peak request MiB")
    parser.add_argument(
        "--block-mib", type=int, default=4, help="HYLE Region maximum block MiB"
    )
    parser.add_argument(
        "--dll-dir", action="append", default=[], help="prepend a runtime DLL dir"
    )
    parser.add_argument("--cpu", help="Linux taskset CPU/list for every process")
    parser.add_argument("--raw", help="write every parsed row as CSV")
    args = parser.parse_args()
    if args.runs < 1 or args.mib < 4 or args.block_mib < 1:
        parser.error("runs and block-mib must be positive; mib must be >= 4")

    exe = os.path.abspath(args.exe)
    base_env = os.environ.copy()
    base_env["HYLE_BENCH_SECTIONS"] = "H"
    base_env["HYLE_MEM_FOOTPRINT_MIB"] = str(args.mib)
    base_env["HYLE_MEM_FOOTPRINT_BLOCK_MIB"] = str(args.block_mib)
    if args.dll_dir:
        dirs = [os.path.abspath(path) for path in args.dll_dir]
        base_env["PATH"] = os.pathsep.join(dirs + [base_env.get("PATH", "")])
    command = [exe]
    if args.cpu:
        command = ["taskset", "-c", args.cpu, exe]

    runs = []
    metadata = None
    for run_index in range(args.runs):
        run = {}
        order = VARIANTS if run_index % 2 == 0 else tuple(reversed(VARIANTS))
        for variant in order:
            env = base_env.copy()
            env["HYLE_MEM_FOOTPRINT_VARIANT"] = variant
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            output = result.stdout + result.stderr
            if result.returncode != 0:
                sys.exit(
                    f"run {run_index} {variant} exited {result.returncode}: "
                    f"{(result.stderr or result.stdout)[:500]}"
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
                    sys.exit(
                        f"run {run_index} {variant}: Apple QoS metadata missing "
                        "or unsuccessful"
                    )
                current_metadata.update(runtime)
            else:
                current_metadata.update(
                    {"qos": "n/a", "qos_ok": 1, "affinity": "external"}
                )
            if metadata is not None and current_metadata != metadata:
                sys.exit("certificate metadata changed across child processes")
            if row["released_bytes"] != 0:
                sys.exit(f"run {run_index} {variant}: backing bytes remain released")
            metadata = current_metadata
            run[variant] = row
        runs.append(run)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit, dirty = git_metadata(repo)
    print(
        f"commit={commit} dirty={int(dirty)} compiler={metadata['compiler']} "
        f"platform={metadata['platform']} cpu={args.cpu or 'unbound'} "
        f"qos={metadata['qos']} qos_ok={metadata['qos_ok']} "
        f"affinity={metadata['affinity']} runs={args.runs} mib={args.mib} "
        f"block_mib={args.block_mib}"
    )

    for variant in VARIANTS:
        rows = [run[variant] for run in runs]
        reserved_ratio = [
            row["peak_reserved_bytes"] / row["requested_bytes"] for row in rows
        ]
        reuse_ratio = [
            row["reuse_reserved_bytes"] / row["reuse_requested_bytes"]
            for row in rows
        ]
        print(
            f"{variant:16s} reserved/payload="
            f"{statistics.median(reserved_ratio):.4f}"
            f"[{min(reserved_ratio):.4f},{max(reserved_ratio):.4f}] "
            f"reuse_reserved/payload={statistics.median(reuse_ratio):.4f} "
            f"retained={median_range([r['retained_after_clear_bytes'] for r in rows])}"
        )
        for phase, field in (
            ("live", "live_rss_bytes"),
            ("clear", "clear_rss_bytes"),
            ("reuse", "reuse_rss_bytes"),
            ("release", "release_rss_bytes"),
            ("process_peak", "process_peak_rss_bytes"),
        ):
            print(
                f"  rss_delta_{phase:12s} "
                f"{median_range([delta(row, field) for row in rows])}"
            )

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
            "reserved_per_payload",
            "reuse_reserved_per_payload",
            "live_rss_delta_bytes",
            "clear_rss_delta_bytes",
            "reuse_rss_delta_bytes",
            "release_rss_delta_bytes",
            "process_peak_rss_delta_bytes",
        ]
        with open(raw_path, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for run_index, run in enumerate(runs):
                for variant in VARIANTS:
                    row = run[variant]
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
                            **row,
                            "reserved_per_payload": (
                                row["peak_reserved_bytes"] / row["requested_bytes"]
                            ),
                            "reuse_reserved_per_payload": (
                                row["reuse_reserved_bytes"]
                                / row["reuse_requested_bytes"]
                            ),
                            **{
                                f"{phase}_rss_delta_bytes": delta(row, field)
                                for phase, field in (
                                    ("live", "live_rss_bytes"),
                                    ("clear", "clear_rss_bytes"),
                                    ("reuse", "reuse_rss_bytes"),
                                    ("release", "release_rss_bytes"),
                                    ("process_peak", "process_peak_rss_bytes"),
                                )
                            },
                        }
                    )
        print(f"raw={raw_path}")


if __name__ == "__main__":
    main()
