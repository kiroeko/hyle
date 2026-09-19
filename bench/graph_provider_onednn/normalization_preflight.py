#!/usr/bin/env python3
"""Module workload A/A only; all product statistics stay in hyle_perf."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from hyle_perf import (evaluate_machine_profile, expected_child_policy,
                       get_machine_profile, load_registry, probe_machine)
from hyle_perf.nanobench import parse_nanobench, select_pair
from hyle_perf.runner import _apply_sampling_policy, _child_environment

PREFIX = "# HYLE_NORMALIZATION_RUN "
STATUS = "hyle-normalization-workload-aa.status"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_record(path: Path, value: dict) -> None:
    # Records are immutable, including failures. Never reuse a preflight name.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def create_record_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    directory = root / uuid.uuid4().hex
    directory.mkdir()
    return directory


def run_child(directory: Path, command: list[str], environment: dict[str, str],
              observed_environment: dict[str, str], *, timeout: float,
              policy: dict | None = None) -> dict:
    directory.mkdir()
    result = {"command": command, "cwd": str(directory),
              "environment": observed_environment, "timeout_seconds": timeout,
              "exit_code": None, "timed_out": False, "pid": None}
    started = time.monotonic()
    with (directory / "stdout.bin").open("xb") as stdout, \
            (directory / "stderr.bin").open("xb") as stderr:
        try:
            with _apply_sampling_policy(policy) if policy is not None else nullcontext():
                with subprocess.Popen(
                        command, cwd=directory, env=environment,
                        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                        creationflags=0x00008000 if os.name == "nt" else 0) as child:
                    result["pid"] = child.pid
                    try:
                        result["exit_code"] = child.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        result["exit_code"] = child.wait()
                        result["timed_out"] = True
        except OSError as error:
            result["launch_error"] = str(error)
    result["wall_seconds_not_performance"] = time.monotonic() - started
    result["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                       for p in sorted(directory.iterdir()) if p.is_file()}
    write_record(directory / "process.json", result)
    return result


def fields(output: str, prefix: str) -> list[dict[str, str]]:
    result = []
    for line in output.splitlines():
        if line.startswith(prefix):
            pairs = [token.split("=", 1) for token in line[len(prefix):].split()]
            value = dict(pairs)
            if len(value) != len(pairs):
                raise RuntimeError("duplicate observation field")
            result.append(value)
    return result


def validate_instances(output: str, workloads: list[str], route: str,
                       *, pid: int, affinity_mask: int,
                       require_intervals: bool = True) -> dict:
    instances = fields(output, "# HYLE_NORMALIZATION_INSTANCE ")
    processes = fields(output, "# HYLE_NORMALIZATION_PROCESS ")
    codes = fields(output, "# HYLE_NORMALIZATION_CODE ")
    intervals = fields(output, "# HYLE_NORMALIZATION_INTERVAL ")
    expected = {(name, role) for name in workloads for role in ("candidate", "peer")}
    if len(instances) != len(expected) or {(r["workload"], r["role"]) for r in instances} != expected:
        raise RuntimeError("missing or duplicate instance observations")
    if len(processes) != len(workloads) or {r["workload"] for r in processes} != set(workloads):
        raise RuntimeError("missing or duplicate process observations")
    if require_intervals:
        if len(intervals) != len(workloads) or {r["workload"] for r in intervals} != set(workloads):
            raise RuntimeError("missing or duplicate interval observations")
    elif intervals:
        raise RuntimeError("verify-only process unexpectedly entered timing")
    if any(r["workload"] not in workloads for r in codes):
        raise RuntimeError("unknown code observation workload")
    for name in workloads:
        pair = [r for r in instances if r["workload"] == name]
        provider = ("onednn-prefix" if name.startswith("graph-") else "onednn") if route == "selected" else "native"
        if any(r["provider"] != provider for r in pair):
            raise RuntimeError("A/A instance provider mismatch")
        if pair[0]["input_handle"] != pair[1]["input_handle"]:
            raise RuntimeError("A/A inputs do not share the same immutable owner")
        for key in ("object", "output_handle"):
            if pair[0][key] == pair[1][key] or any(int(r[key], 16) == 0 for r in pair):
                raise RuntimeError("A/A mutable instances are not independent")
        if any(int(r["input_handle"], 16) == 0 or r["input_handle"] == r["output_handle"] for r in pair):
            raise RuntimeError("A/A input/output ownership mismatch")
        for row in pair:
            if name.startswith("graph-"):
                if row["workspace_handle"] != "opaque" or row["workspace_bytes"] != "opaque":
                    raise RuntimeError("unexpected Graph private backing claim")
            elif int(row["workspace_bytes"]) > 0 and int(row["workspace_handle"], 16) == 0:
                raise RuntimeError("Direct workspace observation missing")
        process = next(r for r in processes if r["workload"] == name)
        code = [r for r in codes if r["workload"] == name]
        if int(process["pid"]) != pid or int(process["tid"]) <= 0 or not affinity_mask & (1 << int(process["cpu"])):
            raise RuntimeError("A/A observation process/thread/CPU mismatch")
        if int(process["executable_base"], 16) == 0 or int(process["private_executable_ranges"]) != len(code):
            raise RuntimeError("incomplete executable address observations")
        if any(int(r["base"], 16) == 0 or int(r["bytes"]) <= 0 for r in code):
            raise RuntimeError("invalid executable address range")
        if require_intervals:
            interval = next(r for r in intervals if r["workload"] == name)
            if int(interval["wall_ns"]) <= 0 or int(interval["thread_cpu_100ns"]) < 0 or any(
                    not affinity_mask & (1 << int(interval[key])) for key in ("cpu_before", "cpu_after")):
                raise RuntimeError("invalid coarse interval observations")
    return {"instances": instances, "processes": processes,
            "private_executable_ranges": codes, "coarse_intervals": intervals}


def evaluate_ratios(runs: list[dict], rows: list[dict]) -> dict:
    summaries = {}
    for route in ("selected", "peer"):
        group = [run for run in runs if run["route"] == route]
        for row in rows:
            name = row["id"]
            values = [run["ratios"][name] for run in group]
            if any(not 1 / 1.10 <= value <= 1.10 for value in values):
                raise RuntimeError(f"gross A/A instability: {route}/{name}: {values}")
            medians = {"all": statistics.median(values)}
            for factor in ("variant", "order", "allocation"):
                for level in sorted({run[factor] for run in group}):
                    medians[f"{factor}={level}"] = statistics.median(
                        run["ratios"][name] for run in group if run[factor] == level)
            if any(not 1 / 1.05 <= value <= 1.05 for value in medians.values()):
                raise RuntimeError(f"fixed A/A factor bias: {route}/{name}: {medians}")
            summaries[f"{route}:{name}"] = medians
    return summaries


def stage_runtime(binaries: dict[str, Path], build: Path) -> dict[str, str]:
    # The qualified runner invokes this validator before any benchmark process.
    # Stage the compiler's x64 CRT next to every measured executable.
    redist = Path("C:/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Redist/MSVC")
    versions = sorted(p for p in redist.iterdir()
                      if (p / "x64/Microsoft.VC145.CRT").is_dir())
    if not versions:
        raise RuntimeError("MSVC x64 CRT is missing")
    libraries = sorted((versions[-1] / "x64/Microsoft.VC145.CRT").glob("*.dll"))
    if not libraries:
        raise RuntimeError("MSVC x64 CRT contains no DLLs")
    directories = {path.parent for path in binaries.values()}
    directories.update(path.parent for path in build.rglob("hyle_perf_control_bench.exe"))
    receipt = {}
    for directory in sorted(directories):
        for source in libraries:
            destination = directory / source.name
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            if not destination.exists():
                shutil.copy2(source, destination)
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"runtime DLL differs: {destination}")
            receipt[str(destination.relative_to(build))] = expected
    return receipt


def text_size(path: Path) -> int:
    data = path.read_bytes()
    offset = struct.unpack_from("<I", data, 0x3c)[0]
    if data[offset:offset + 4] != b"PE\0\0":
        raise RuntimeError("expected a Windows PE artifact")
    count = struct.unpack_from("<H", data, offset + 6)[0]
    optional_size = struct.unpack_from("<H", data, offset + 20)[0]
    sections = offset + 24 + optional_size
    for i in range(count):
        section = sections + i * 40
        if data[section:section + 8].rstrip(b"\0") == b".text":
            return struct.unpack_from("<I", data, section + 8)[0]
    raise RuntimeError("missing .text section")


def qualify(build: Path, record_directory: Path) -> dict:
    if os.name != "nt":
        raise RuntimeError("normalization preflight is bound to W1-WIN")
    binaries = {}
    for variant, suffix in (("primary", ""), ("placement", "_placement")):
        matches = list(build.rglob(f"hyle_onednn_normalization_bench{suffix}.exe"))
        if len(matches) != 1:
            raise RuntimeError(f"ambiguous {variant} executable: {matches}")
        binaries[variant] = matches[0]
    ctypes.WinDLL("kernel32", use_last_error=True).SetErrorMode(0x8003)
    runtime = stage_runtime(binaries, build)
    receipts = {path.parent / STATUS for path in binaries.values()}
    for receipt in receipts:
        receipt.write_text("FAIL\n", encoding="ascii")
    sizes = {key: text_size(path) for key, path in binaries.items()}
    if abs(sizes["placement"] - sizes["primary"]) > 4096:
        raise RuntimeError(f"placement materially changed .text: {sizes}")

    profile = get_machine_profile(load_registry(), "W1-WIN")
    probe = probe_machine(paths=[("build", build)])
    gate = evaluate_machine_profile(profile, probe, role="engineering",
                                    sampling_topology="single-logical-cpu")
    if gate["gate"] != "PASS":
        raise RuntimeError(f"machine gate failed: {gate}")
    policy = expected_child_policy(profile, probe=probe,
                                   sampling_topology="single-logical-cpu")
    contract_path = Path(__file__).resolve().parents[1] / "contracts" / "onednn-normalization-p1-v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rows = contract["workloads"]
    titles = {item["id"]: item["title"] for item in
              contract["extensions"]["runner"]["executables"][0]["workloads"]}
    runs = []
    artifacts = {key: {"sha256": digest(path), "text_bytes": sizes[key]}
                 for key, path in binaries.items()}
    source = Path(__file__).resolve().parents[2]
    sources = {str(p.relative_to(source)): digest(p) for p in (
        Path(__file__).resolve(), Path(__file__).with_name("normalization_p1_bench.cc"),
        Path(__file__).with_name("normalization_placement.cc"),
        source / "bench/common/perf_process.h", contract_path)}
    write_record(record_directory / "identity.json", {
        "artifacts": artifacts, "sources": sources, "child_policy": policy,
        "machine_gate": gate, "probe": probe, "runtime_dlls": runtime})
    print(json.dumps({"child_policy": policy, "text_sizes": sizes,
                      "runtime_dlls": runtime}), flush=True)
    for route in ("selected", "peer"):
        for variant, binary in binaries.items():
            for order in ("candidate-first", "peer-first"):
                for allocation in ("candidate-first", "peer-first"):
                    name = f"{route}-{variant}-{order}-{allocation}"
                    nonce = record_directory.name + ":" + name
                    overrides = {"HYLE_BENCH_AA_ROUTE": route,
                                 "HYLE_BENCH_OBSERVE_INSTANCES": "1",
                                 "ONEDNN_JIT_DUMP": "1",
                                 "HYLE_PERF_RUN_NONCE": nonce}
                    if order == "peer-first":
                        overrides["HYLE_BENCH_RIVAL_FIRST"] = "1"
                    if allocation == "peer-first":
                        overrides["HYLE_BENCH_PEER_ALLOCATION_FIRST"] = "1"
                    environment = _child_environment(
                        contract["extensions"]["runner"]["controlled_environment"] +
                        ["ONEDNN_JIT_DUMP", "DNNL_JIT_DUMP"],
                        overrides, profile, policy)
                    directory = record_directory / name
                    observed_names = set(contract["extensions"]["runner"]["controlled_environment"]) | set(overrides) | {
                        "HYLE_PERF_PARENT_MASK", "HYLE_PERF_PRIORITY", "HYLE_PERF_SAMPLING_TOPOLOGY",
                        "HYLE_PERF_CALLER_CPU", "TRACY_ONLY_LOCALHOST"}
                    result = run_child(directory, [str(binary)], environment,
                                       {k: environment[k] for k in observed_names if k in environment},
                                       timeout=120, policy=policy)
                    output = "".join((directory / stream).read_bytes().decode("utf-8", errors="replace")
                                     for stream in ("stdout.bin", "stderr.bin"))
                    if result["exit_code"] != 0 or result["timed_out"]:
                        raise RuntimeError(f"A/A child failed; raw record: {directory}")
                    markers = [line for line in output.splitlines() if line.startswith(PREFIX)]
                    if len(markers) != 1:
                        raise RuntimeError("missing or duplicate run marker")
                    telemetry = fields(output, PREFIX)[0]
                    expected = {"execution_order": order, "allocation_order": allocation,
                                "aa_route": route, "paired_mode": "nanobench-compare",
                                # Single-CPU policy is imposed by the canonical
                                # parent launcher; there are no worker roles.
                                "affinity_topology": "external",
                                "caller_cpu": "external",
                                "affinity_ok": "not-applicable", "tasks": "1",
                                "run_nonce": nonce}
                    if any(telemetry.get(key) != value for key, value in expected.items()):
                        raise RuntimeError(f"unexpected telemetry: {telemetry}; expected {expected}")
                    observations = validate_instances(
                        output, [row["id"] for row in rows], route,
                        pid=result["pid"], affinity_mask=int(str(policy["affinity_mask"]), 0))
                    jit = {name: record for name, record in result["files"].items()
                           if name.startswith("dnnl_dump_") and name.endswith(".bin")}
                    if (route == "selected" and len(jit) != 32) or (route == "peer" and jit):
                        raise RuntimeError(f"unexpected vendor JIT dump count: {route}/{len(jit)}")
                    parsed = parse_nanobench(output)
                    ratios = {}
                    for row in rows:
                        candidate, peer = select_pair(parsed, row["pair"]["candidate"],
                                                       row["pair"]["peer"], title=titles[row["id"]])
                        ratios[row["id"]] = candidate.operations_per_second / peer.operations_per_second
                    run = {"route": route, "variant": variant, "order": order,
                           "allocation": allocation, "ratios": ratios, "output": output}
                    runs.append(run)
                    write_record(directory / "parsed.json", {"telemetry": telemetry, "observations": observations,
                                                               "ratios": ratios, "vendor_jit": jit})
                    print(json.dumps(run), flush=True)
    summaries = evaluate_ratios(runs, rows)
    if any(digest(path) != artifacts[key]["sha256"] for key, path in binaries.items()) or any(
            digest(source / name) != value for name, value in sources.items()):
        raise RuntimeError("preflight artifact/source/contract identity drift")
    result = {"schema": "hyle.normalization-workload-aa/v2", "gate": "PASS",
              "artifacts": artifacts, "summaries": summaries}
    for receipt in receipts:
        receipt.write_text("PASS\n", encoding="ascii")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    args = parser.parse_args()
    directory = create_record_directory(args.records.resolve())
    write_record(directory / "invocation.json", {"argv": sys.argv, "build": str(args.build.resolve()),
                                                 "nonce": directory.name, "started_unix_ns": time.time_ns()})
    print(json.dumps({"raw_records": str(directory)}), flush=True)
    try:
        result = qualify(args.build.resolve(strict=True), directory)
        write_record(directory / "summary.json", result)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        # Include status files created during this first invocation, too.
        for receipt in args.build.rglob(STATUS):
            receipt.write_text("FAIL\n", encoding="ascii")
        write_record(directory / "failure.json", {"gate": "FAIL", "error": str(error)})
        raise
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"normalization preflight: {error}", file=sys.stderr)
        raise SystemExit(2)
