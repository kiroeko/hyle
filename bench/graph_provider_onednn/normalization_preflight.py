#!/usr/bin/env python3
"""Module workload A/A only; all product statistics stay in hyle_perf."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from hyle_perf import (evaluate_machine_profile, expected_child_policy,
                       get_machine_profile, load_registry, probe_machine)
from hyle_perf.nanobench import parse_nanobench, select_pair
from hyle_perf.runner import _apply_sampling_policy, _child_environment

PREFIX = "# HYLE_NORMALIZATION_RUN "
STATUS = "hyle-normalization-workload-aa.status"


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("normalization preflight is bound to W1-WIN")
    build = args.build.resolve(strict=True)
    binaries = {}
    for variant, suffix in (("primary", ""), ("placement", "_placement")):
        matches = list(build.rglob(f"hyle_onednn_normalization_bench{suffix}.exe"))
        if len(matches) != 1:
            raise RuntimeError(f"ambiguous {variant} executable: {matches}")
        binaries[variant] = matches[0]
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
    contract = json.loads((Path(__file__).resolve().parents[1] / "contracts" /
                           "onednn-normalization-p1-v1.json").read_text(encoding="utf-8"))
    rows = contract["workloads"]
    titles = {item["id"]: item["title"] for item in
              contract["extensions"]["runner"]["executables"][0]["workloads"]}
    runs = []
    print(json.dumps({"child_policy": policy, "text_sizes": sizes}), flush=True)
    for route in ("selected", "peer"):
        for variant, binary in binaries.items():
            for order in ("candidate-first", "peer-first"):
                for allocation in ("candidate-first", "peer-first"):
                    overrides = {"HYLE_BENCH_AA_ROUTE": route}
                    if order == "peer-first":
                        overrides["HYLE_BENCH_RIVAL_FIRST"] = "1"
                    if allocation == "peer-first":
                        overrides["HYLE_BENCH_PEER_ALLOCATION_FIRST"] = "1"
                    environment = _child_environment(
                        contract["extensions"]["runner"]["controlled_environment"],
                        overrides, profile, policy)
                    with _apply_sampling_policy(policy):
                        result = subprocess.run([str(binary)], capture_output=True,
                                                text=True, encoding="utf-8", errors="replace",
                                                env=environment, timeout=120,
                                                creationflags=0x00008000)
                    output = result.stdout + result.stderr
                    if result.returncode != 0:
                        raise RuntimeError(f"A/A failed: {output}")
                    markers = [line for line in output.splitlines() if line.startswith(PREFIX)]
                    if len(markers) != 1:
                        raise RuntimeError("missing or duplicate run marker")
                    telemetry = dict(token.split("=", 1) for token in markers[0][len(PREFIX):].split())
                    expected = {"execution_order": order, "allocation_order": allocation,
                                "aa_route": route, "paired_mode": "nanobench-compare",
                                # Single-CPU policy is imposed by the canonical
                                # parent launcher; there are no worker roles.
                                "affinity_topology": "external",
                                "caller_cpu": "external",
                                "affinity_ok": "not-applicable", "tasks": "1"}
                    if any(telemetry.get(key) != value for key, value in expected.items()):
                        raise RuntimeError(f"unexpected telemetry: {telemetry}; expected {expected}")
                    parsed = parse_nanobench(output)
                    ratios = {}
                    for row in rows:
                        candidate, peer = select_pair(parsed, row["pair"]["candidate"],
                                                       row["pair"]["peer"], title=titles[row["id"]])
                        ratios[row["id"]] = candidate.operations_per_second / peer.operations_per_second
                    run = {"route": route, "variant": variant, "order": order,
                           "allocation": allocation, "ratios": ratios, "output": output}
                    runs.append(run)
                    print(json.dumps(run), flush=True)
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
    print(json.dumps({"schema": "hyle.normalization-workload-aa/v1", "gate": "PASS",
                      "artifacts": {key: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                          "text_bytes": sizes[key]} for key, path in binaries.items()},
                      "summaries": summaries}), flush=True)
    for receipt in receipts:
        receipt.write_text("PASS\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"normalization preflight: {error}", file=sys.stderr)
        raise SystemExit(2)
