#!/usr/bin/env python3
"""Run the fixed Linux HTTP/2 paired suite; retain every cell and failure."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time


COUNTERS = ("voluntary_switches", "involuntary_switches", "minor_faults", "major_faults")
MODES = ("paired-self", "paired", "paired", "paired", "paired-self")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def identity(path):
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256") if sys.version_info >= (3, 11) else None
        if digest is None:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_text(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def machine_metadata(cpu, allowed):
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    cpuinfo = read_text(Path("/proc/cpuinfo")) or ""
    return {
        "uname": platform.uname()._asdict(), "allowed_affinity": sorted(allowed), "cpu": cpu,
        "cpu_model": next((line.partition(":")[2].strip() for line in cpuinfo.splitlines()
                           if line.startswith("model name")), None),
        "selected_cpu_topology": {name: read_text(topology / name) for name in
                                  ("core_id", "physical_package_id", "thread_siblings_list",
                                   "core_siblings_list", "core_type")},
        "environment": {key: value for key, value in os.environ.items()
                        if key.startswith("HYLE_") or key in
                        ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT")},
        "topology_limit": "Guest CPU numbering/topology does not prove a physical P-core or host isolation.",
    }


def summarize(stdout, mode, cpu):
    records = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    if len(records) != 49 or any(not isinstance(record, dict) for record in records) or \
            records[0].get("type") != "metadata":
        raise ValueError("Expected one metadata record followed by 48 cells")
    metadata, cells = records[0], records[1:]
    expected = {"mode": "self" if mode == "paired-self" else "paired",
                "blocks": 12, "episodes_per_cell": 64,
                "warmups_per_arm": 16, "streams_per_episode": 4,
                "application_bytes_per_episode": 2097152, "socket_requested_bytes": 4096,
                "caller_cpu": cpu}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Unexpected metadata {key}: {metadata.get(key)!r}")
    delta = metadata.get("clock_min_delta_ns", 0)
    if not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("Invalid clock_min_delta_ns")
    for key in ("socket_send_bytes", "socket_receive_bytes"):
        if not isinstance(metadata.get(key), list) or not metadata[key] or any(
                type(value) is not int or value <= 0 for value in metadata[key]):
            raise ValueError(f"Invalid {key}")
    totals = {arm: dict.fromkeys(COUNTERS, 0) for arm in ("hyle", "reference")}
    blocks = []
    for block in range(12):
        group = cells[block * 4:block * 4 + 4]
        arms = ("hyle", "reference", "reference", "hyle")
        if block % 2:
            arms = ("reference", "hyle", "hyle", "reference")
        logs = {"hyle": [], "reference": []}
        for position, (cell, arm) in enumerate(zip(group, arms)):
            if any(cell.get(key) != value for key, value in
                   {"type": "cell", "block": block, "position": position,
                    "arm": arm, "iterations": 64}.items()) or cell.get("valid") is not True:
                raise ValueError(f"Invalid cell at block {block}, position {position}")
            elapsed = cell.get("seconds_per_episode", 0)
            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Cell time must be positive and finite")
            if cell.get("cpu_before") != cpu or cell.get("cpu_after") != cpu:
                raise ValueError("Cell CPU does not match the pinned CPU")
            logs[arm].append(math.log(elapsed))
            for key in COUNTERS:
                if type(cell.get(key)) is not int or cell[key] < 0:
                    raise ValueError(f"Invalid cell counter {key}")
                totals[arm][key] += cell[key]
        difference = (sum(logs["hyle"]) - sum(logs["reference"])) / 2
        blocks.append({"block": block, "log_ratio": difference,
                       "ratio_percent": 100 * math.expm1(difference)})
    ratios = [block["ratio_percent"] for block in blocks]
    quartiles = statistics.quantiles(ratios, n=4, method="inclusive")
    return {"metadata": metadata, "blocks": blocks,
            "median_ratio_percent": statistics.median(ratios),
            "geomean_ratio_percent": 100 * math.expm1(statistics.mean(
                block["log_ratio"] for block in blocks)),
            "range_percent": [min(ratios), max(ratios)],
            "quartiles_percent": [quartiles[0], quartiles[2]],
            "descriptive_iqr_percent": quartiles[2] - quartiles[0],
            "counter_totals_by_arm": totals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="built hyle_io_http2_p1_bench executable")
    parser.add_argument("--output", type=Path, required=True, help="new result directory; never overwritten")
    parser.add_argument("--cpu", type=int, help="allowed Linux CPU (default: 2, otherwise lowest allowed)")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This AF_UNIX suite requires Linux")
    allowed = os.sched_getaffinity(0)
    cpu = args.cpu if args.cpu is not None else (2 if 2 in allowed else min(allowed))
    if cpu not in allowed:
        parser.error(f"CPU {cpu} is outside allowed affinity {sorted(allowed)}")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"Not an executable file: {binary}")
    taskset = shutil.which("taskset")
    if taskset is None:
        parser.error("taskset is required")
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"Refusing to overwrite output: {args.output}")
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Refusing to overwrite output: {output}")
    root = Path(__file__).resolve().parents[2]
    git = shutil.which("git")
    head = subprocess.run([git, "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False) if git else None
    metadata = machine_metadata(cpu, allowed)
    metadata.update({"binary": identity(binary),
                     "git_head": head.stdout.strip() if head and head.returncode == 0 else None,
                     "source_identity_limit": "These current source hashes do not prove how the binary was built.",
                     "sources": [identity(path) for path in
                                 (Path(__file__).resolve(), root / "bench/io_http2/http2_p1_bench.cc",
                                  root / "src/io_http2/http2.cc")]})
    write_json(output / "metadata.json", metadata)
    summary = {"status": "running", "runs": [],
               "method": "Each statistic uses 12 complete four-cell blocks; positive means Hyle slower. "
                         "Iterations are not independent samples. IQR uses inclusive quartiles. "
                         "No confidence interval, equivalence claim, outlier removal or result-based retries.",
               "counters": "rusage deltas cover the cell measurement call including framework overhead; "
                           "they are not hardware performance counters."}
    for index, mode in enumerate(MODES):
        name = f"{index + 1:02d}-{mode}"
        command = [taskset, "-c", str(cpu), str(binary), "--" + mode]
        run = {"name": name, "mode": mode, "command": command,
               "cwd": str(Path.cwd()), "timeout_seconds": 60, "status": "running"}
        summary["runs"].append(run)
        write_json(output / (name + ".command.json"), run)
        write_json(output / "summary.json", summary)
        start = time.monotonic()
        try:
            with (output / (name + ".stdout.jsonl")).open("wb") as stdout, \
                    (output / (name + ".stderr.txt")).open("wb") as stderr:
                process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=60, check=False)
            run["returncode"] = process.returncode
            if process.returncode:
                raise ValueError(f"Process exited {process.returncode}")
            result = summarize((output / (name + ".stdout.jsonl")).read_text(encoding="utf-8"), mode, cpu)
            run.update({"status": "completed", "result": result})
            print(f"{name}: median={result['median_ratio_percent']:+.4f}%", flush=True)
        except (OSError, ValueError, OverflowError, subprocess.TimeoutExpired) as error:
            run.update({"status": "failed", "error": str(error)})
            summary["status"] = "failed; stopped without retry"
            print(f"{name}: {error}; retained results in {output}", file=sys.stderr)
        run["wall_seconds"] = time.monotonic() - start
        write_json(output / "summary.json", summary)
        if run["status"] != "completed":
            return 1
    summary.update({"status": "completed",
                    "self_observations": [summary["runs"][i]["result"] for i in (0, 4)],
                    "paired_observations": [run["result"] for run in summary["runs"][1:4]]})
    write_json(output / "summary.json", summary)
    print(f"Complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
