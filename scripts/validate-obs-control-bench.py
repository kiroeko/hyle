#!/usr/bin/env python3
"""Validate the frozen Obs call-site/inference performance adapter.

The shared performance runner owns sampling and statistics. This validator
checks only the source identity, semantic emission counts, and distinct
primary/placement artifacts used by obs-control-p1-v1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


PREFIX = "# HYLE_OBS_CONTROL_CONTRACT "
EXPECTED = {
    "schema": "1",
    "ordinary_checks": "4096",
    "hyle_ordinary": "64",
    "absl_ordinary": "64",
    "inference_slots": "16",
    "inference_ticks": "256",
    "hyle_inference": "80",
    "manual_inference": "80",
    "hyle_gate": "80",
    "manual_gate": "80",
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
            "Obs control marker differs from the frozen semantic oracle: "
            f"expected={EXPECTED!r} actual={fields!r}"
        )
    return {"path": str(executable), "sha256": _sha256(executable)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--expected-benchmark-sha256", required=True)
    args = parser.parse_args()

    source = pathlib.Path(args.source).resolve(strict=True)
    build = pathlib.Path(args.build).resolve(strict=True)
    benchmark = source / "bench" / "obs" / "control_bench.cc"
    actual_hash = _sha256(benchmark)
    if actual_hash != args.expected_benchmark_sha256:
        raise RuntimeError(
            "Obs control benchmark differs from the contract-frozen hash: "
            f"expected={args.expected_benchmark_sha256} actual={actual_hash}"
        )

    primary = _run(_find_executable(build, "hyle_obs_control_bench"))
    placement = _run(
        _find_executable(build, "hyle_obs_control_bench_placement")
    )
    if primary["sha256"] == placement["sha256"]:
        raise RuntimeError("primary and placement artifacts are byte-identical")

    print(
        json.dumps(
            {
                "schema": "hyle.obs-control-validator/v1",
                "semantic_oracle": "PASS",
                "placement_artifact": "PASS",
                "benchmark_sha256": actual_hash,
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
        print(f"obs-control validator: {exc}", file=sys.stderr)
        raise SystemExit(2)
