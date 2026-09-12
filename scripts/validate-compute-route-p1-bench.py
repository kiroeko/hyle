#!/usr/bin/env python3
"""Workload-matched A/A preflight for the Compute CPU route P1 adapter.

The common arithmetic controls cannot detect order, cache, allocation, or
placement bias in a complete Tensor/PreparedOperation/Invocation journey. This
validator replaces both timed rows with the same selected or native product
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
import subprocess
import sys

from hyle_perf.nanobench import parse_nanobench, select_pair


PREFIX = "# HYLE_COMPUTE_ROUTE_RUN "
LOWER = 1.0 / 1.05
UPPER = 1.05
WORKLOADS = (
    (
        "compute route row-major MatMul M=1 K=1024 N=1024",
        "HYLE selected MatMul M=1 K=1024 N=1024",
        "HYLE native MatMul M=1 K=1024 N=1024",
    ),
    (
        "compute route row-major MatMul M=4 K=1024 N=1024",
        "HYLE selected MatMul M=4 K=1024 N=1024",
        "HYLE native MatMul M=4 K=1024 N=1024",
    ),
    (
        "compute route row-major MatMul M=32 K=512 N=1024",
        "HYLE selected MatMul M=32 K=512 N=1024",
        "HYLE native MatMul M=32 K=512 N=1024",
    ),
    (
        "compute route transposed-weight MatMul M=1 K=1024 N=1024",
        "HYLE selected transposed MatMul M=1 K=1024 N=1024",
        "HYLE native transposed MatMul M=1 K=1024 N=1024",
    ),
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


def _telemetry(output: str) -> dict[str, str]:
    lines = [line for line in output.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise RuntimeError(
            f"expected one Compute route marker, found {len(lines)}"
        )
    result: dict[str, str] = {}
    for token in lines[0][len(PREFIX) :].split():
        if "=" not in token:
            raise RuntimeError(f"malformed telemetry token {token!r}")
        key, value = token.split("=", 1)
        if key in result:
            raise RuntimeError(f"duplicate telemetry key {key!r}")
        result[key] = value
    return result


def _environment(
    route: str, peer_first: bool, peer_allocation_first: bool, caller_cpu: str
) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HYLE_BENCH_") and not key.startswith("HYLE_PERF_")
    }
    environment.update(
        {
            "HYLE_BENCH_AA_ROUTE": route,
            "HYLE_PERF_CALLER_CPU": caller_cpu,
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
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
    caller_cpu: str,
) -> dict[str, object]:
    creationflags = 0x00008000 if os.name == "nt" else 0
    completed = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_environment(route, peer_first, peer_allocation_first, caller_cpu),
        timeout=120,
        creationflags=creationflags,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"{executable.name} A/A exited {completed.returncode}:\n{output}"
        )
    telemetry = _telemetry(output)
    expected = {
        "execution_order": "peer-first" if peer_first else "candidate-first",
        "allocation_order": (
            "peer-first" if peer_allocation_first else "candidate-first"
        ),
        "aa_route": route,
        "paired_mode": "nanobench-compare",
        "affinity_topology": "instrument",
        "caller_cpu": caller_cpu,
        "affinity_ok": "1",
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
        if not LOWER <= ratio <= UPPER:
            raise RuntimeError(
                f"{executable.name} {route} A/A for {title!r} has "
                f"advantage ratio {ratio:.6f}, outside [{LOWER:.6f}, {UPPER:.6f}]"
            )
        ratios[title] = ratio
    return {
        "route": route,
        "execution_order": expected["execution_order"],
        "allocation_order": expected["allocation_order"],
        "ratios": ratios,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True)
    parser.add_argument("--caller-cpu", default="2")
    args = parser.parse_args()
    build = pathlib.Path(args.build).resolve(strict=True)
    executables = {
        "primary": _find_executable(build, "hyle_compute_route_bench"),
        "placement": _find_executable(
            build, "hyle_compute_route_bench_placement"
        ),
    }

    runs = []
    for route in ("native", "selected"):
        for variant, executable in executables.items():
            for peer_first in (False, True):
                for peer_allocation_first in (False, True):
                    result = _run(
                        executable,
                        route=route,
                        peer_first=peer_first,
                        peer_allocation_first=peer_allocation_first,
                        caller_cpu=args.caller_cpu,
                    )
                    result["variant"] = variant
                    runs.append(result)

    print(
        json.dumps(
            {
                "schema": "hyle.compute-route-workload-aa/v1",
                "gate": "PASS",
                "maximum_bias_percent": 5,
                "artifacts": {
                    name: {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in executables.items()
                },
                "runs": runs,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"compute route P1 validator: {exc}", file=sys.stderr)
        raise SystemExit(2)
