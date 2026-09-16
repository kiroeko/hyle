#!/usr/bin/env python3
"""Finite, full-FFN CUDA A/A and dependency qualification, not a P1 runner."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from hyle_perf.environment import (  # noqa: E402
    PERF_ENVIRONMENT_BLOCKLIST, PERF_ENVIRONMENT_FIXED, PERF_ENVIRONMENT_PREFIXES)
from hyle_perf.nanobench import parse_nanobench, select_pair  # noqa: E402
from hyle_perf.machine import (  # noqa: E402
    evaluate_machine_profile, expected_child_policy, get_machine_profile,
    load_registry, probe_machine)

PREFIX = "# HYLE_CUDA_FFN_RUN "
GPU = "NVIDIA GeForce RTX 4070"
GPU_UUID = "GPU-e7376560-6f5d-ec19-6ee4-3dd3a2c81ef8"
TOPOLOGY = "native-runtime-pair"


def child_environment() -> dict[str, str]:
    result = {key: value for key, value in os.environ.items()
              if key not in PERF_ENVIRONMENT_BLOCKLIST and not any(
                  key.startswith(prefix) for prefix in PERF_ENVIRONMENT_PREFIXES)}
    result.update(PERF_ENVIRONMENT_FIXED)
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_facts() -> dict[str, str]:
    keys = ["name", "uuid", "driver_version", "pstate", "temperature.gpu",
            "power.draw", "power.limit", "clocks.sm", "clocks.mem",
            "utilization.gpu", "clocks_event_reasons.sw_thermal_slowdown",
            "clocks_event_reasons.hw_thermal_slowdown"]
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + ",".join(keys),
         "--format=csv,noheader,nounits"], check=True, capture_output=True,
        text=True, timeout=20)
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(keys):
        raise RuntimeError("expected one recorded W1 GPU")
    facts = dict(zip(keys, rows[0]))
    if (facts["name"], facts["uuid"], facts["driver_version"]) != (
            GPU, GPU_UUID, "610.62"):
        raise RuntimeError(f"GPU/driver is outside frozen qualification: {facts}")
    if any(facts[key] != "Not Active" for key in keys[-2:]):
        raise RuntimeError(f"GPU thermal slowdown active: {facts}")
    return facts


def stage_runtime(build: Path) -> dict[str, str]:
    """Copy actual DLLs before any process launch; never depend on PATH."""
    cuda = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
    sources = [cuda / "bin/x64" / name for name in (
        "cudart64_13.dll", "cublasLt64_13.dll")]
    sources += [Path("C:/Windows/System32") / name for name in (
        "msvcp140.dll", "msvcp140_atomic_wait.dll", "vcruntime140.dll",
        "vcruntime140_1.dll")]
    receipt = {}
    for source in sources:
        destination = build / "bin" / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
        if digest(destination) != digest(source):
            raise RuntimeError(f"runtime DLL changed: {destination}")
        receipt[str(destination)] = digest(destination)
    return receipt


def reference_identity(build: Path) -> dict[str, str]:
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    match = re.search(r"^HYLE_CUDA_SGLANG_SOURCE_DIR:[^=]+=(.+)$", cache, re.M)
    if not match:
        raise RuntimeError("missing pinned peer checkout in the build contract")
    peer = Path(match[1])
    provenance = json.loads((Path(__file__).parent / "reference/provenance.json")
                            .read_text(encoding="utf-8"))["peer"]
    content = {}
    for relative, expected in provenance["files"].items():
        raw = subprocess.check_output(
            ["git", "show", provenance["commit"] + ":" + relative], cwd=peer)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError(f"pinned reference differs: {relative}")
        content[relative] = raw.decode("utf-8")
    original = content["python/sglang/kernels/jit/csrc/elementwise/activation.cuh"]
    fragment = original[original.index("namespace sglang {"):
                        original.index("struct UnaryActivationParams {")]
    local = (Path(__file__).parent / "reference/sglang_activation.cuh")\
        .read_text(encoding="utf-8")
    if local[local.index("namespace sglang {"):] != fragment + "\n} // namespace sglang\n":
        raise RuntimeError("compiled reference arithmetic/indexing differs from pin")
    compiler = re.search(r"^CMAKE_CUDA_COMPILER:[^=]+=(.+)$", cache, re.M)
    if not compiler:
        raise RuntimeError("CUDA compiler identity missing")
    return {"peer": provenance["commit"], "nvcc_sha256": digest(Path(compiler[1]))}


def telemetry(text: str) -> dict[str, str]:
    lines = [line[len(PREFIX):] for line in text.splitlines()
             if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise RuntimeError("expected exactly one full-FFN telemetry marker")
    # A device name contains spaces. Values end at the next key, not next space.
    pairs = re.findall(r"(\w+)=(.*?)(?= \w+=|$)", lines[0])
    if len(dict(pairs)) != len(pairs):
        raise RuntimeError("duplicate full-FFN telemetry key")
    return dict(pairs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("compute", "graph"), required=True)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("this qualification is W1-WIN engineering only")
    build = args.build.resolve(strict=True)
    ready = build / "bin" / f"hyle_cuda_{args.mode}_aa.ready"
    ready.write_text("FAIL\n", encoding="ascii")
    profile = get_machine_profile(load_registry(), "W1-WIN")
    probe = probe_machine(paths=[("build", build)])
    gate = evaluate_machine_profile(
        profile, probe, role="engineering", sampling_topology=TOPOLOGY)
    if gate["gate"] != "PASS":
        raise RuntimeError(f"CUDA A/A machine gate failed: {gate['errors']}")
    policy = expected_child_policy(profile, probe=probe, sampling_topology=TOPOLOGY)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetErrorMode(0x8003)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.GetProcessAffinityMask.argtypes = [ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    process = kernel32.GetCurrentProcess()
    # Match the instrument's complete CPU budget; the executable separately
    # pins its caller and leaves opaque SDK helper placement to the runtime.
    if not kernel32.SetProcessAffinityMask(process, int(policy["affinity_mask"], 16)):
        raise ctypes.WinError(ctypes.get_last_error())
    observed_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
            process, ctypes.byref(observed_mask), ctypes.byref(system_mask)):
        raise ctypes.WinError(ctypes.get_last_error())
    if observed_mask.value != int(policy["affinity_mask"], 16):
        raise RuntimeError("A/A process affinity differs from W1 P1 policy")
    target = f"hyle_{args.mode}_route_cuda_bench"
    executables = {variant: build / "bin" / (target + suffix + ".exe")
                   for variant, suffix in (("primary", ""), ("placement", "_placement"))}
    artifacts = {name: digest(path) for name, path in executables.items()}
    runtime = stage_runtime(build)
    reference = reference_identity(build)
    # Reuse only this bundle's exact immutable A/A receipt at the post-sampling
    # call. Recheck GPU, executable, compiler and all side-by-side DLL identities.
    receipt_path = args.output / f"cuda-{args.mode}-aa.json"
    identity = {"artifacts": artifacts, "runtime": runtime, "reference": reference,
                "validator": digest(Path(__file__)), "child_policy": policy,
                "machine_identity": gate["physical_host_identity"],
                "process_mask": hex(observed_mask.value)}
    before = gpu_facts()
    if receipt_path.exists():
        retained = json.loads(receipt_path.read_text(encoding="utf-8"))
        if retained.get("identity") != identity or retained.get("gate") != "PASS":
            raise RuntimeError("A/A identity changed or previous preflight failed")
        for run in retained["runs"]:
            if digest(Path(run["log"])) != run["sha256"]:
                raise RuntimeError("retained A/A raw log changed")
        print(json.dumps({"gate": "PASS", "reused_receipt": str(receipt_path),
                          "receipt_sha256": digest(receipt_path), "gpu_after": before}))
        ready.write_text("PASS\n", encoding="ascii")
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / f"cuda-{args.mode}-aa-raw"
    raw_dir.mkdir()
    kind = "Direct FFN" if args.mode == "compute" else "Graph captured FFN"
    runs = []
    for route in ("native", "selected"):
        for variant, executable in executables.items():
            for peer_first in (False, True):
                for allocation_first in (False, True):
                    environment = child_environment()
                    environment.update(HYLE_BENCH_AA_ROUTE=route,
                                       HYLE_PERF_CALLER_CPU=str(policy["caller_logical_cpu"]),
                                       HYLE_PERF_PARENT_MASK=policy["affinity_mask"],
                                       HYLE_PERF_PRIORITY=policy["priority"],
                                       HYLE_PERF_SAMPLING_TOPOLOGY=TOPOLOGY,
                                       HYLE_PERF_RUN_NONCE=uuid.uuid4().hex)
                    if peer_first:
                        environment["HYLE_BENCH_RIVAL_FIRST"] = "1"
                    if allocation_first:
                        environment["HYLE_BENCH_PEER_ALLOCATION_FIRST"] = "1"
                    log = raw_dir / f"{route}-{variant}-{int(peer_first)}-{int(allocation_first)}.log"
                    try:
                        completed = subprocess.run(
                            [str(executable)], env=environment, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=20,
                            creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS)
                    except subprocess.TimeoutExpired as exc:
                        log.write_bytes((exc.stdout or b"") + (exc.stderr or b""))
                        raise RuntimeError(f"A/A progress timeout; see {log}") from exc
                    text = completed.stdout + completed.stderr
                    log.write_text(text, encoding="utf-8")
                    if completed.returncode:
                        raise RuntimeError(f"A/A executable failed; see {log}")
                    expected = {
                        "mode": "direct" if args.mode == "compute" else "captured",
                        "execution_order": "peer-first" if peer_first else "candidate-first",
                        "allocation_order": "peer-first" if allocation_first else "candidate-first",
                        "aa_route": route, "paired_mode": "nanobench-compare",
                        "gpu": GPU, "sm": "89", "driver_api": "13030", "runtime": "13030",
                        "affinity_topology": TOPOLOGY,
                        "caller_cpu": str(policy["caller_logical_cpu"]),
                        "parent_mask": policy["affinity_mask"], "priority": policy["priority"],
                        "affinity_ok": "1", "verify_only": "0",
                        "run_nonce": environment["HYLE_PERF_RUN_NONCE"]}
                    observed = telemetry(text)
                    if any(observed.get(key) != value for key, value in expected.items()):
                        raise RuntimeError(f"A/A telemetry mismatch: {observed}")
                    ratios = {}
                    parsed = parse_nanobench(text)
                    for rows in (1, 16):
                        title = f"{kind} M={rows} K=4096 H=11008 N=4096"
                        candidate, peer = select_pair(
                            parsed, "HYLE CUDA " + title, "SGLang precise " + title, title=title)
                        ratios[str(rows)] = candidate.operations_per_second / peer.operations_per_second
                    runs.append({"route": route, "variant": variant,
                                 "execution_order": expected["execution_order"],
                                 "allocation_order": expected["allocation_order"],
                                 "ratios": ratios, "log": str(log.resolve()),
                                 "sha256": digest(log), "gpu": gpu_facts()})
    summaries, errors = {}, []
    for route in ("native", "selected"):
        matching = [run for run in runs if run["route"] == route]
        for rows in ("1", "16"):
            values = [run["ratios"][rows] for run in matching]
            medians = {"all": statistics.median(values)}
            for factor in ("variant", "execution_order", "allocation_order"):
                for level in sorted({run[factor] for run in matching}):
                    medians[f"{factor}={level}"] = statistics.median(
                        run["ratios"][rows] for run in matching if run[factor] == level)
            if any(not 1 / 1.10 <= value <= 1.10 for value in values):
                errors.append(f"{route} M{rows}: gross process instability exceeds 10%")
            if any(not 1 / 1.05 <= value <= 1.05 for value in medians.values()):
                errors.append(f"{route} M{rows}: factor median bias exceeds 5%")
            summaries[f"{route}-M{rows}"] = {"ratios": values, "factor_medians": medians}
    receipt = {"schema": "hyle.cuda-ffn-aa/v1", "gate": "FAIL" if errors else "PASS",
               "identity": identity, "gpu_before": before, "runs": runs,
               "summaries": summaries, "errors": errors}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": receipt["gate"], "receipt": str(receipt_path),
                      "receipt_sha256": digest(receipt_path), "errors": errors}))
    if not errors:
        ready.write_text("PASS\n", encoding="ascii")
    return 2 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"CUDA FFN A/A: {exc}", file=sys.stderr)
        raise SystemExit(2)
