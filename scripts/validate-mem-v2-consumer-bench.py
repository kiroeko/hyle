#!/usr/bin/env python3
"""Validate the frozen Mem v2 P1 source-consumer adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


PREFIX = "# HYLE_MEM_V2_CONTRACT "
TEXT_FIELDS = {
    "schema": "1",
    "heap": "PASS",
    "arena": "PASS",
    "view": "PASS",
    "pool": "PASS",
    "verdict": "PASS",
}
NUMERIC_FIELDS = {
    "batch",
    "provider_allocations",
    "cache_reuses",
    "cached_buffers",
    "live_buffers",
    "peer_provider_allocations",
    "peer_provider_deallocations",
    "peer_cache_reuses",
    "peer_cached_buffers",
    "peer_live_buffers",
}
WORKLOADS = {
    "isolated-heap-lifecycle-64",
    "external-scratch-epoch-64",
    "buffer-view-range-fill-64",
    "transient-buffer-lifecycle-64",
}
DIMENSIONS = {
    "work-result-granularity",
    "ownership-lifetime-teardown",
    "failure-safety-diagnostics",
    "thread-progress-synchronization",
    "setup-steady-teardown-charging",
    "resource-delivery-codegen",
}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(path: pathlib.Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} sha256 differs: expected={expected}, actual={actual}"
        )


def _load_shapes(source: pathlib.Path, expected_digest: str) -> dict:
    path = source / "bench/mem/consumer_shapes_v2.json"
    _require_digest(path, expected_digest, "Mem v2 consumer shapes")
    shapes = json.loads(path.read_text(encoding="utf-8"))
    if shapes.get("schema") != "hyle.mem.v2-consumer-shapes/v1":
        raise RuntimeError("unsupported Mem v2 consumer-shapes schema")
    if shapes.get("comparison_class") != "SAME_CONTRACT":
        raise RuntimeError("Mem v2 comparison class is not SAME_CONTRACT")
    if shapes.get("provenance_class") != "PROVISIONAL_ARCHETYPE":
        raise RuntimeError("Mem v2 provenance class differs")
    if shapes.get("material_margin_percent") != 5:
        raise RuntimeError("Mem v2 material margin differs")

    for record in shapes.get("source_evidence", []):
        _require_digest(
            source / record["path"], record["sha256"],
            f"Mem v2 source evidence {record['role']}",
        )

    journeys = {record.get("id") for record in shapes.get("journeys", [])}
    if journeys != WORKLOADS:
        raise RuntimeError(
            f"Mem v2 journeys differ: expected={sorted(WORKLOADS)!r} "
            f"actual={sorted(str(value) for value in journeys)!r}"
        )
    records = shapes.get("semantic_equivalence")
    if not isinstance(records, list):
        raise RuntimeError("semantic_equivalence must be an array")
    for workload in WORKLOADS:
        dimensions = {
            record.get("dimension")
            for record in records
            if record.get("workload") == workload
        }
        if dimensions != DIMENSIONS:
            raise RuntimeError(
                f"semantic dimensions differ for {workload}: "
                f"expected={sorted(DIMENSIONS)!r} "
                f"actual={sorted(str(value) for value in dimensions)!r}"
            )
    for record in records:
        if record.get("workload") not in WORKLOADS:
            raise RuntimeError("semantic matrix names an unknown workload")
        if record.get("verdict") not in {"EQUAL", "ADAPTER_EQUALIZED"}:
            raise RuntimeError(
                f"material semantic difference in {record.get('workload')}/"
                f"{record.get('dimension')}"
            )
    return shapes


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


def _parse_fields(output: str) -> dict[str, str]:
    lines = [line for line in output.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise RuntimeError(
            f"expected one contract marker {PREFIX!r}, found {len(lines)}"
        )
    fields: dict[str, str] = {}
    for token in lines[0][len(PREFIX):].split():
        if "=" not in token:
            raise RuntimeError(f"malformed contract token {token!r}")
        key, value = token.split("=", 1)
        if key in fields:
            raise RuntimeError(f"duplicate contract field {key!r}")
        fields[key] = value
    expected = set(TEXT_FIELDS) | NUMERIC_FIELDS
    if set(fields) != expected:
        raise RuntimeError(
            "contract fields differ: "
            f"missing={sorted(expected - set(fields))!r} "
            f"unknown={sorted(set(fields) - expected)!r}"
        )
    for key, value in TEXT_FIELDS.items():
        if fields[key] != value:
            raise RuntimeError(
                f"contract field {key} differs: expected={value!r} "
                f"actual={fields[key]!r}"
            )
    return fields


def _validate_numeric(fields: dict[str, str]) -> None:
    try:
        values = {key: int(fields[key]) for key in NUMERIC_FIELDS}
    except ValueError as error:
        raise RuntimeError("contract numeric field is not an integer") from error
    if values["batch"] != 64:
        raise RuntimeError("Mem v2 aggregate batch differs")
    if values["provider_allocations"] != 64 or \
            values["cache_reuses"] != 64 or \
            values["cached_buffers"] != 64 or \
            values["live_buffers"] != 0:
        raise RuntimeError("HYLE BufferPool did not preserve the warm working set")
    if values["peer_provider_allocations"] <= 0 or \
            values["peer_provider_deallocations"] > \
            values["peer_provider_allocations"] or \
            values["peer_cache_reuses"] != 64 or \
            values["peer_cached_buffers"] != 64 or \
            values["peer_live_buffers"] != 0:
        raise RuntimeError("adapted oneTBB pool did not preserve the warm contract")


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
    _validate_numeric(_parse_fields(output))
    return {"path": str(executable), "sha256": _sha256(executable)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--build", required=True, type=pathlib.Path)
    parser.add_argument("--expected-shapes-sha256", required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    build = args.build.resolve(strict=True)
    shapes = _load_shapes(source, args.expected_shapes_sha256)
    primary = _run(_find_executable(build, "hyle_mem_v2_consumer_bench"))
    placement = _run(
        _find_executable(build, "hyle_mem_v2_consumer_bench_placement")
    )
    if primary["sha256"] == placement["sha256"]:
        raise RuntimeError("primary and placement artifacts are byte-identical")

    print(json.dumps({
        "schema": "hyle.mem-v2-validator/v1",
        "comparison_class": shapes["comparison_class"],
        "provenance_class": shapes["provenance_class"],
        "semantic_oracle": "PASS",
        "resource_oracle": "PASS",
        "placement_artifact": "PASS",
        "primary": primary,
        "placement": placement,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError,
            subprocess.SubprocessError) as error:
        print(f"Mem v2 consumer validation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
