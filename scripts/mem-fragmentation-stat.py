#!/usr/bin/env python3
"""Aggregate isolated fixed-pool fragmentation certificates."""

import argparse
import csv
import os
import statistics
import subprocess
import sys


PREFIX = "HYLE_MEM_CERT fragmentation "
NUMERIC_FIELDS = (
    "payload_budget_bytes",
    "backing_storage_bytes",
    "allocations",
    "empty_largest_bytes",
    "fill_requested_bytes",
    "live_requested_bytes",
    "freed_requested_bytes",
    "fragmented_largest_bytes",
    "recovered_largest_bytes",
)
VARIANTS = ("hyle_fixed_pool", "onetbb_fixed_pool")


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
    if values.get("scenario") != "fixed_pool_mixed_holes":
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


def metrics(row):
    available = row["empty_largest_bytes"] - row["live_requested_bytes"]
    if available <= 0 or row["empty_largest_bytes"] == 0:
        raise ValueError("invalid normalized capacity")
    return {
        "fill_per_empty": row["fill_requested_bytes"]
        / row["empty_largest_bytes"],
        "largest_per_available": row["fragmented_largest_bytes"] / available,
        "recovered_per_empty": row["recovered_largest_bytes"]
        / row["empty_largest_bytes"],
    }


def median_range(values, digits=4):
    return (
        f"{statistics.median(values):.{digits}f}"
        f"[{min(values):.{digits}f},{max(values):.{digits}f}]"
    )


def main():
    parser = argparse.ArgumentParser(
        description="measure fixed-pool external fragmentation in fresh processes"
    )
    parser.add_argument("exe")
    parser.add_argument("-n", "--runs", type=int, default=15)
    parser.add_argument("--mib", type=int, default=8, help="HYLE payload MiB")
    parser.add_argument(
        "--dll-dir", action="append", default=[], help="prepend a runtime DLL dir"
    )
    parser.add_argument("--cpu", help="Linux taskset CPU/list for every process")
    parser.add_argument("--raw", help="write every parsed row as CSV")
    args = parser.parse_args()
    if args.runs < 1 or args.mib < 1:
        parser.error("runs and mib must be positive")

    exe = os.path.abspath(args.exe)
    base_env = os.environ.copy()
    base_env["HYLE_BENCH_SECTIONS"] = "I"
    base_env["HYLE_MEM_FRAGMENTATION_MIB"] = str(args.mib)
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
            env["HYLE_MEM_FRAGMENTATION_VARIANT"] = variant
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
                sys.exit(f"run {run_index} {variant}: invalid fragmentation row")
            if current_metadata is None or current_metadata["format"] != 1:
                sys.exit(f"run {run_index} {variant}: invalid metadata")
            if "HYLE_MEM_CERT build testing=0" not in output:
                sys.exit(
                    f"run {run_index} {variant}: benchmark is not a shipping "
                    "build (HYLE_TESTING must be OFF)"
                )
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
            try:
                metrics(row)
            except ValueError as error:
                sys.exit(f"run {run_index} {variant}: {error}")
            metadata = current_metadata
            run[variant] = row
        runs.append(run)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit, dirty = git_metadata(repo)
    print(
        f"commit={commit} dirty={int(dirty)} compiler={metadata['compiler']} "
        f"platform={metadata['platform']} cpu={args.cpu or 'unbound'} "
        f"qos={metadata['qos']} qos_ok={metadata['qos_ok']} "
        f"affinity={metadata['affinity']} runs={args.runs} mib={args.mib}"
    )

    summaries = {}
    for variant in VARIANTS:
        rows = [run[variant] for run in runs]
        calculated = [metrics(row) for row in rows]
        summaries[variant] = calculated
        print(
            f"{variant:18s} empty_largest="
            f"{median_range([r['empty_largest_bytes'] for r in rows], 0)} "
            f"fragmented_largest="
            f"{median_range([r['fragmented_largest_bytes'] for r in rows], 0)}"
        )
        for field in (
            "fill_per_empty",
            "largest_per_available",
            "recovered_per_empty",
        ):
            print(
                f"  {field:24s} "
                f"{median_range([value[field] for value in calculated])}"
            )

    print("normalized ratios (HYLE/oneTBB; >1 favors HYLE)")
    for field in ("largest_per_available", "recovered_per_empty"):
        ratios = [
            summaries["hyle_fixed_pool"][index][field]
            / summaries["onetbb_fixed_pool"][index][field]
            for index in range(args.runs)
        ]
        print(f"  {field:24s} {median_range(ratios)}")

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
            "fill_per_empty",
            "largest_per_available",
            "recovered_per_empty",
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
                            **metrics(row),
                        }
                    )
        print(f"raw={raw_path}")


if __name__ == "__main__":
    main()
