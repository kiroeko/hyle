#!/usr/bin/env python3
"""Narrow semantic/resource validator for container-p1-v1.

The shared performance runner owns sampling, statistics, and verdicts. This
validator only executes the module adapter's frozen correctness oracle for the
primary and placement artifacts and checks the resource facts that make the
five candidate/peer pairs comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


PREFIX = "# HYLE_CONTAINER_P1_CONTRACT "
EXPECTED = {
    "schema": "1",
    "sequence": "PASS",
    "map": "PASS",
    "queue": "PASS",
    "bitset": "PASS",
    "hash": "PASS",
    "map_candidate_capacity": "65536",
    "map_peer_capacity": "65535",
    "queue_candidate_storage": "65536",
    "queue_peer_storage": "65600",
    "verdict": "PASS",
}


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
            "container contract marker differs from the frozen semantic/resource "
            f"oracle: expected={EXPECTED!r} actual={fields!r}"
        )
    return {
        "path": str(executable),
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True)
    args = parser.parse_args()
    build = pathlib.Path(args.build).resolve(strict=True)

    primary = _run(_find_executable(build, "hyle_container_p1_bench"))
    placement = _run(
        _find_executable(build, "hyle_container_p1_bench_placement")
    )
    if primary["sha256"] == placement["sha256"]:
        raise RuntimeError("primary and placement artifacts are byte-identical")

    print(
        json.dumps(
            {
                "schema": "hyle.container-p1-validator/v1",
                "semantic_oracle": "PASS",
                "resource_oracle": "PASS",
                "placement_artifact": "PASS",
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
        print(f"container-p1 validator: {exc}", file=sys.stderr)
        raise SystemExit(2)
