#!/usr/bin/env python3
"""Narrow semantic/DCE/placement validator for error-p1-v1.

The shared performance runner owns statistics and verdicts.  This module-only
validator executes the exact primary/placement replay functions in validation
mode, checks their observable 64-call topology and semantic oracles, then
proves that the placement target changed relative function layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


CONTRACT_PREFIX = "# HYLE_ERROR_P1_CONTRACT "
LAYOUT_PREFIX = "# HYLE_ERROR_P1_LAYOUT "
LAYOUT_KEYS = (
    "hyle_eval",
    "hyle_success_node",
    "hyle_error_node",
    "hyle_info_leaf",
    "hyle_info_chain",
    "peer_eval",
    "peer_success_node",
    "peer_error_node",
    "peer_info_leaf",
    "peer_info_chain",
)


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
    layout = _fields(output, LAYOUT_PREFIX)
    expected_contract = {
        "schema": "1",
        "nodes": "64",
        "hyle_success_calls": "64",
        "peer_success_calls": "64",
        "hyle_error_calls": "64",
        "peer_error_calls": "64",
        "hyle_success_checksum": "2080",
        "peer_success_checksum": "2080",
        "hyle_error_checksum": "2080",
        "peer_error_checksum": "2080",
        "hyle_failure_index": "63",
        "peer_failure_index": "63",
        "graph": "PASS",
        "message": "PASS",
        "context": "PASS",
        "result_external_fixture": "PASS",
        "format": "PASS",
        "update": "PASS",
        "source_location": "PASS",
        "verdict": "PASS",
    }
    if contract != expected_contract:
        raise RuntimeError(
            "contract marker differs from the frozen semantic/call oracle: "
            f"expected={expected_contract!r} actual={contract!r}"
        )
    if set(layout) != set(LAYOUT_KEYS):
        raise RuntimeError(f"layout marker fields differ: {sorted(layout)!r}")
    addresses = {key: int(layout[key], 16) for key in LAYOUT_KEYS}
    if any(value == 0 for value in addresses.values()):
        raise RuntimeError("layout marker contains a null function address")
    if len(set(addresses.values())) < 8:
        raise RuntimeError("too many measured noinline functions were folded together")
    return {
        "executable": str(executable),
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "contract": contract,
        "addresses": addresses,
    }


def _relative_layout(record: dict[str, object]) -> list[int]:
    addresses = record["addresses"]
    assert isinstance(addresses, dict)
    base = int(addresses["hyle_eval"])
    return [int(addresses[key]) - base for key in LAYOUT_KEYS]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True)
    args = parser.parse_args()
    build = pathlib.Path(args.build).resolve(strict=True)

    primary = _run(_find_executable(build, "hyle_error_p1_bench"))
    placement = _run(
        _find_executable(build, "hyle_error_p1_bench_placement")
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
                "schema": "hyle.error-p1-validator/v1",
                "semantic_oracle": "PASS",
                "actual_call_topology": "PASS",
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
        print(f"error-p1 validator: {exc}", file=sys.stderr)
        raise SystemExit(2)
