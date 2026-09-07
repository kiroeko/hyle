#!/usr/bin/env python3
"""Validate the Container v6 single-thread consumer adapter.

The shared performance runner owns sampling and statistics. This validator
binds the semantic matrix and benchmark source, runs the exact correctness
oracle for primary and placement artifacts, and proves those artifacts differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


PREFIX = "# HYLE_CONTAINER_V6_CONTRACT "
EXPECTED = {
    "schema": "2",
    "lru": "PASS",
    "lru_oom": "PASS",
    "container_text": "PASS",
    "unicode_malformed": "PASS",
    "unicode_output_small": "PASS",
    "lru_capacity": "4096",
    "lru_operations": "4096",
    "unicode_records": "256",
    "unicode_input_bytes": "32768",
    "unicode_units": "4096",
    "unicode_unit_bytes": "8",
    "verdict": "PASS",
}


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


def _fields(output: str) -> dict[str, str]:
    lines = [line for line in output.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise RuntimeError(
            f"expected one contract marker {PREFIX!r}, found {len(lines)}"
        )
    result: dict[str, str] = {}
    for token in lines[0][len(PREFIX) :].split():
        if "=" not in token:
            raise RuntimeError(f"malformed contract token {token!r}")
        key, value = token.split("=", 1)
        if key in result:
            raise RuntimeError(f"duplicate contract field {key!r}")
        result[key] = value
    return result


def _run(executable: pathlib.Path) -> dict[str, str]:
    completed = subprocess.run(
        [str(executable), "--validate-contract"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"{executable} validation exited {completed.returncode}:\n{output}"
        )
    fields = _fields(output)
    if fields != EXPECTED:
        raise RuntimeError(
            "Container v6 marker differs from the frozen semantic oracle: "
            f"expected={EXPECTED!r} actual={fields!r}"
        )
    return {"path": str(executable), "sha256": _sha256(executable)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--expected-benchmark-sha256", required=True)
    parser.add_argument("--expected-shape-sha256", required=True)
    args = parser.parse_args()

    source = pathlib.Path(args.source).resolve(strict=True)
    build = pathlib.Path(args.build).resolve(strict=True)
    benchmark = source / "bench" / "container" / "product_v6_bench.cc"
    shape = source / "bench" / "container" / "consumer_shapes_v6.json"
    benchmark_hash = _sha256(benchmark)
    shape_hash = _sha256(shape)
    if benchmark_hash != args.expected_benchmark_sha256:
        raise RuntimeError(
            "Container v6 benchmark differs from the contract-frozen hash: "
            f"expected={args.expected_benchmark_sha256} actual={benchmark_hash}"
        )
    if shape_hash != args.expected_shape_sha256:
        raise RuntimeError(
            "Container v6 consumer shape differs from the contract-frozen hash: "
            f"expected={args.expected_shape_sha256} actual={shape_hash}"
        )

    primary = _run(_find_executable(build, "hyle_container_v6_p1_bench"))
    placement = _run(
        _find_executable(build, "hyle_container_v6_p1_bench_placement")
    )
    if primary["sha256"] == placement["sha256"]:
        raise RuntimeError("primary and placement artifacts are byte-identical")

    print(
        json.dumps(
            {
                "schema": "hyle.container-v6-validator/v1",
                "semantic_oracle": "PASS",
                "placement_artifact": "PASS",
                "benchmark_sha256": benchmark_hash,
                "shape_sha256": shape_hash,
                "primary": primary,
                "placement": placement,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"container-v6 validator: {exc}", file=sys.stderr)
        raise SystemExit(2)
