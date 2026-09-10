#!/usr/bin/env python3
"""Validate Container Text Base64 semantics and placement evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


CONTRACT_PREFIX = "# HYLE_CONTAINER_TEXT_BASE64_CONTRACT "
LAYOUT_PREFIX = "# HYLE_CONTAINER_TEXT_BASE64_LAYOUT "
BENCHMARK_PATH = pathlib.PurePosixPath(
    "bench/container_text/base64_product_bench.cc"
)
SHAPE_PATH = pathlib.PurePosixPath(
    "bench/container_text/base64_consumer_shapes_v1.json"
)
LAYOUT_KEYS = ("hyle_decode", "peer_decode", "hyle_encode", "peer_encode")


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
    expected = {
        "schema": "1",
        "decode": "PASS",
        "encode": "PASS",
        "failures": "PASS",
        "decode_bytes": "262144",
        "encode_records": "4096",
        "digest_bytes": "20",
        "verdict": "PASS",
    }
    if contract != expected:
        raise RuntimeError(
            "semantic/call oracle differs: "
            f"expected={expected!r} actual={contract!r}"
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
        "path": str(executable),
        "sha256": _sha256(executable),
        "addresses": addresses,
    }


def _relative_layout(record: dict[str, object]) -> list[int]:
    addresses = record["addresses"]
    assert isinstance(addresses, dict)
    base = int(addresses["hyle_decode"])
    return [int(addresses[key]) - base for key in LAYOUT_KEYS]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--expected-benchmark-sha256", required=True)
    parser.add_argument("--expected-shape-sha256", required=True)
    args = parser.parse_args()

    source = pathlib.Path(args.source).resolve(strict=True)
    build = pathlib.Path(args.build).resolve(strict=True)
    benchmark = source.joinpath(*BENCHMARK_PATH.parts)
    shape = source.joinpath(*SHAPE_PATH.parts)
    benchmark_hash = _sha256(benchmark)
    shape_hash = _sha256(shape)
    if benchmark_hash != args.expected_benchmark_sha256:
        raise RuntimeError(
            "Base64 benchmark differs from the contract-frozen hash: "
            f"expected={args.expected_benchmark_sha256} actual={benchmark_hash}"
        )
    if shape_hash != args.expected_shape_sha256:
        raise RuntimeError(
            "Base64 consumer shape differs from the contract-frozen hash: "
            f"expected={args.expected_shape_sha256} actual={shape_hash}"
        )

    primary = _run(
        _find_executable(build, "hyle_container_text_base64_p1_bench")
    )
    placement = _run(
        _find_executable(
            build, "hyle_container_text_base64_p1_bench_placement"
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
                "schema": "hyle.container-text-base64-validator/v1",
                "benchmark_sha256": benchmark_hash,
                "shape_sha256": shape_hash,
                "semantic_oracle": "PASS",
                "same_contract": "PASS",
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
        print(f"container-text Base64 validator: {exc}", file=sys.stderr)
        raise SystemExit(2)
