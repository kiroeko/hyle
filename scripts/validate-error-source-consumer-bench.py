#!/usr/bin/env python3
"""Validate the source-consumer error replay and its placement perturbation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


CONTRACT_PREFIX = "# HYLE_ERROR_SOURCE_CONSUMER_CONTRACT "
LAYOUT_PREFIX = "# HYLE_ERROR_SOURCE_CONSUMER_LAYOUT "
BENCHMARK_PATH = pathlib.PurePosixPath(
    "bench/error/source_consumer_bench.cc"
)
LAYOUT_KEYS = (
    "hyle_eval",
    "hyle_parallel",
    "hyle_join_producer",
    "hyle_join_consumer",
    "peer_eval",
    "peer_parallel",
    "peer_join_producer",
    "peer_join_consumer",
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_executable(build: pathlib.Path, stem: str) -> pathlib.Path:
    names = {stem, f"{stem}.exe"}
    candidates = sorted(
        path.resolve()
        for path in build.rglob("*")
        if path.is_file() and path.name in names
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one {stem!r} executable, found "
            f"{[str(path) for path in candidates]!r}"
        )
    return candidates[0]


def _fields(output: str, prefix: str) -> dict[str, str]:
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError(f"expected one marker {prefix!r}, found {len(lines)}")
    result: dict[str, str] = {}
    for token in lines[0][len(prefix) :].split():
        if "=" not in token:
            raise RuntimeError(f"malformed marker token {token!r}")
        key, value = token.split("=", 1)
        if key in result:
            raise RuntimeError(f"duplicate marker field {key!r}")
        result[key] = value
    return result


def _run(executable: pathlib.Path) -> dict[str, object]:
    completed = subprocess.run(
        [str(executable), "--validate-contract"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"{executable} validation exited {completed.returncode}:\n{output}"
        )
    contract = _fields(output, CONTRACT_PREFIX)
    expected_contract = {
        "schema": "1",
        "nodes": "64",
        "hyle_success_calls": "64",
        "peer_success_calls": "64",
        "hyle_failure_calls": "64",
        "peer_failure_calls": "64",
        "hyle_success_checksum": "2080",
        "peer_success_checksum": "2080",
        "hyle_failure_checksum": "2080",
        "peer_failure_checksum": "2080",
        "hyle_failure_index": "63",
        "peer_failure_index": "63",
        "graph": "PASS",
        "parallel_handoff": "PASS",
        "join_handoff": "PASS",
        "source_location": "PASS",
        "verdict": "PASS",
    }
    if contract != expected_contract:
        raise RuntimeError(
            "semantic/call oracle differs: "
            f"expected={expected_contract!r} actual={contract!r}"
        )
    layout = _fields(output, LAYOUT_PREFIX)
    if set(layout) != set(LAYOUT_KEYS):
        raise RuntimeError(f"layout marker fields differ: {sorted(layout)!r}")
    addresses = {key: int(layout[key], 16) for key in LAYOUT_KEYS}
    if any(value == 0 for value in addresses.values()):
        raise RuntimeError("layout marker contains a null function address")
    if len(set(addresses.values())) != len(LAYOUT_KEYS):
        raise RuntimeError("measured noinline functions were folded together")
    return {
        "executable": str(executable),
        "sha256": _sha256(executable),
        "addresses": addresses,
    }


def _relative_layout(record: dict[str, object]) -> list[int]:
    addresses = record["addresses"]
    assert isinstance(addresses, dict)
    base = int(addresses["hyle_eval"])
    return [int(addresses[key]) - base for key in LAYOUT_KEYS]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--expected-benchmark-sha256", required=True)
    args = parser.parse_args()
    source = pathlib.Path(args.source).resolve(strict=True)
    build = pathlib.Path(args.build).resolve(strict=True)
    benchmark = source.joinpath(*BENCHMARK_PATH.parts)
    if _sha256(benchmark) != args.expected_benchmark_sha256:
        raise RuntimeError(
            "source-consumer benchmark differs from the contract-frozen hash"
        )

    primary = _run(
        _find_executable(build, "hyle_error_source_consumer_bench")
    )
    placement = _run(
        _find_executable(
            build, "hyle_error_source_consumer_bench_placement"
        )
    )
    primary_layout = _relative_layout(primary)
    placement_layout = _relative_layout(placement)
    if primary["sha256"] == placement["sha256"]:
        raise RuntimeError("primary and placement binaries are byte-identical")
    if primary_layout == placement_layout:
        raise RuntimeError(
            "placement target did not change measured-function relative layout"
        )

    print(
        json.dumps(
            {
                "schema": "hyle.error-source-consumer-validator/v1",
                "benchmark_sha256": args.expected_benchmark_sha256,
                "semantic_oracle": "PASS",
                "actual_call_topology": "PASS",
                "shared_algorithm_source": "PASS",
                "primary_sha256": primary["sha256"],
                "placement_sha256": placement["sha256"],
                "primary_relative_layout": primary_layout,
                "placement_relative_layout": placement_layout,
                "placement_authenticity": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error source-consumer validator: {exc}", file=sys.stderr)
        raise SystemExit(2)
