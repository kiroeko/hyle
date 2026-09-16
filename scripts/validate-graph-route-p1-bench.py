#!/usr/bin/env python3
"""Workload-matched A/A preflight for the Graph CPU route P1 adapter.

The common arithmetic controls cannot detect order, cache, allocation, or
placement bias in a complete Graph/Invocation/Sched journey. This
validator replaces both timed rows with the same selected product or complete peer
route while preserving the product benchmark topology. It is intentionally a
short preflight: the common runner still owns independent process sampling and
all product verdicts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys

from hyle_perf.nanobench import parse_nanobench, select_pair
from hyle_perf import (evaluate_machine_profile, expected_child_policy,
                       get_machine_profile, load_registry, probe_machine)
from hyle_perf.runner import _apply_sampling_policy, _child_environment


PREFIX = "# HYLE_GRAPH_ONEDNN_RUN "
AA_PREFIX = "# HYLE_GRAPH_CPU_AA "
STATUS_FILE = "hyle-graph-workload-aa.status"
TOPOLOGY = "caller-plus-n-worker-physical-cores"
LOWER = 1.0 / 1.05
UPPER = 1.05
WORKLOADS = tuple(
    (title + shape, "HYLE Graph oneDNN region " + shape, peer + shape)
    for shape in (
        "decode f32 M=1 K=256 H=512",
        "short-prefill f32 M=16 K=512 H=1024",
        "holdout f32 M=17 K=129 H=257",
        "two-worker f32 M=17 K=129 H=257",
    )
    for title, peer in (
        ("graph oneDNN MLP abstraction ", "direct fused oneDNN region "),
        ("graph oneDNN MLP region benefit ", "direct sequential oneDNN operations "),
        ("graph CPU default versus Native ", "HYLE Graph explicit Native "),
    )
)


def _find_executable(build: pathlib.Path, stem: str) -> pathlib.Path:
    names = {stem, f"{stem}.exe"}
    matches = sorted(
        path.resolve()
        for path in build.rglob("*")
        if path.is_file() and path.name in names
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {stem!r} executable, found "
            f"{[str(path) for path in matches]!r}"
        )
    return matches[0]


def _telemetry(output: str, prefix: str) -> dict[str, str]:
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError(
            f"expected one Graph route marker, found {len(lines)}"
        )
    result: dict[str, str] = {}
    for token in lines[0][len(prefix) :].split():
        if "=" not in token:
            raise RuntimeError(f"malformed telemetry token {token!r}")
        key, value = token.split("=", 1)
        if key in result:
            raise RuntimeError(f"duplicate telemetry key {key!r}")
        result[key] = value
    return result


def _environment(
    route: str, peer_first: bool, peer_allocation_first: bool,
    profile: dict, policy: dict,
) -> dict[str, str]:
    environment = _child_environment(
        [], {"HYLE_BENCH_AA_ROUTE": route}, profile, policy)
    if peer_first:
        environment["HYLE_BENCH_RIVAL_FIRST"] = "1"
    if peer_allocation_first:
        environment["HYLE_BENCH_PEER_ALLOCATION_FIRST"] = "1"
    return environment


def _run(
    executable: pathlib.Path,
    *,
    route: str,
    peer_first: bool,
    peer_allocation_first: bool,
    profile: dict,
    policy: dict,
) -> dict[str, object]:
    creationflags = 0x00008000 if os.name == "nt" else 0
    with _apply_sampling_policy(policy):
        completed = subprocess.run(
            [str(executable)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_environment(
                route, peer_first, peer_allocation_first, profile, policy
            ),
            timeout=120,
            creationflags=creationflags,
        )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"{executable.name} A/A exited {completed.returncode}:\n{output}"
        )
    telemetry = {**_telemetry(output, PREFIX), **_telemetry(output, AA_PREFIX)}
    expected = {
        "execution_order": "peer-first" if peer_first else "candidate-first",
        "allocation_order": (
            "peer-first" if peer_allocation_first else "candidate-first"
        ),
        "aa_route": route,
        "paired_mode": "nanobench-compare",
        "affinity_topology": TOPOLOGY,
        "worker_count": "2",
        "worker_cpus": ",".join(str(cpu) for cpu in policy["worker_logical_cpus"]),
        "parent_mask": policy["affinity_mask"],
        "priority": policy["priority"],
        "caller_cpu": str(policy["caller_logical_cpu"]),
        "affinity_ok": "1",
        "idle_policy": "low-latency",
    }
    for key, value in expected.items():
        if telemetry.get(key) != value:
            raise RuntimeError(
                f"{executable.name} telemetry {key!r} expected {value!r}, "
                f"found {telemetry.get(key)!r}"
            )

    parsed = parse_nanobench(output)
    ratios: dict[str, float] = {}
    for title, candidate_name, peer_name in WORKLOADS:
        candidate, peer = select_pair(
            parsed, candidate_name, peer_name, title=title
        )
        ratio = candidate.operations_per_second / peer.operations_per_second
        ratios[title] = ratio
    return {
        "route": route,
        "execution_order": expected["execution_order"],
        "allocation_order": expected["allocation_order"],
        "ratios": ratios,
        "output": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("this preflight is bound to W1-WIN")
    build = pathlib.Path(args.build).resolve(strict=True)
    executables = {
        "primary": _find_executable(build, "hyle_graph_onednn_region_bench"),
        "placement": _find_executable(
            build, "hyle_graph_onednn_region_bench_placement"
        ),
    }

    # The module benchmark refuses instrumented product sampling unless this
    # preflight passes. The shared runner may collect diagnostics after a failed
    # command validator; this cold module guard keeps that from timing products.
    status_paths = {path.parent / STATUS_FILE for path in executables.values()}
    for status in status_paths:
        status.write_text("FAIL\n", encoding="ascii")
    profile = get_machine_profile(load_registry(), "W1-WIN")
    probe = probe_machine(paths=[("build", build)])
    gate = evaluate_machine_profile(
        profile, probe, role="engineering", sampling_topology=TOPOLOGY,
        sampling_worker_count=2)
    if gate["gate"] != "PASS":
        raise RuntimeError(f"Graph A/A machine gate failed: {gate!r}")
    policy = expected_child_policy(
        profile, probe=probe, sampling_topology=TOPOLOGY, sampling_worker_count=2)
    print(json.dumps({"schema": "hyle.graph-route-workload-aa-policy/v1",
                      "child_policy": policy}, sort_keys=True), flush=True)
    runs = []
    for route in ("peer", "selected"):
        for variant, executable in executables.items():
            for peer_first in (False, True):
                for peer_allocation_first in (False, True):
                    result = _run(
                        executable,
                        route=route,
                        peer_first=peer_first,
                        peer_allocation_first=peer_allocation_first,
                        profile=profile,
                        policy=policy,
                    )
                    result["variant"] = variant
                    runs.append(result)
                    print(json.dumps({"schema": "hyle.graph-route-workload-aa-run/v1",
                                      "run": result}, sort_keys=True), flush=True)

    # A short preflight must reject a fixed execution/allocation/placement
    # effect, not promote one process point at the 5% boundary into a product
    # verdict. Check every factor-level median against the material band and
    # retain a separate 10% per-process gross-instability stop.
    summaries: dict[str, dict[str, object]] = {}
    gross_lower = 1.0 / 1.10
    gross_upper = 1.10
    for route in ("peer", "selected"):
        for title, _, _ in WORKLOADS:
            matching = [run for run in runs if run["route"] == route]
            values = [float(run["ratios"][title]) for run in matching]
            if any(value < gross_lower or value > gross_upper for value in values):
                raise RuntimeError(
                    f"{route} A/A for {title!r} has a gross single-process "
                    f"ratio outside [{gross_lower:.6f}, {gross_upper:.6f}]: "
                    f"{values!r}"
                )
            factor_medians: dict[str, float] = {
                "all": statistics.median(values)
            }
            for factor in ("variant", "execution_order", "allocation_order"):
                for factor_value in sorted({str(run[factor]) for run in matching}):
                    factor_medians[f"{factor}={factor_value}"] = statistics.median(
                        float(run["ratios"][title])
                        for run in matching
                        if run[factor] == factor_value
                    )
            outside = {
                key: value
                for key, value in factor_medians.items()
                if not LOWER <= value <= UPPER
            }
            if outside:
                raise RuntimeError(
                    f"{route} A/A for {title!r} has fixed factor medians "
                    f"outside [{LOWER:.6f}, {UPPER:.6f}]: {outside!r}"
                )
            summaries[f"{route}:{title}"] = {
                "minimum_process_ratio": min(values),
                "maximum_process_ratio": max(values),
                "factor_medians": factor_medians,
            }

    print(
        json.dumps(
            {
                "schema": "hyle.graph-route-workload-aa/v1",
                "gate": "PASS",
                "maximum_bias_percent": 5,
                "gross_instability_percent": 10,
                "artifacts": {
                    name: {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in executables.items()
                },
                "summaries": summaries,
                "runs": runs,
            },
            sort_keys=True,
        )
    )
    for status in status_paths:
        status.write_text("PASS\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"graph route P1 validator: {exc}", file=sys.stderr)
        raise SystemExit(2)
