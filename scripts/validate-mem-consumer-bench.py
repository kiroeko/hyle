#!/usr/bin/env python3
"""Validate retained mem-v1 replay inputs and resource/tail telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys


OWNER_PREFIX = "# HYLE_MEM_OWNER_RESOURCE "
HANDOFF_PREFIX = "# HYLE_MEM_HANDOFF_RESOURCE "


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(path: pathlib.Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} sha256 differs: expected={expected}, actual={actual}")


def parse_fields(line: str, prefix: str) -> dict[str, int]:
    fields: dict[str, int] = {}
    for token in line[len(prefix):].strip().split():
        if "=" not in token:
            raise RuntimeError(f"malformed telemetry token: {token!r}")
        key, value = token.split("=", 1)
        if not re.fullmatch(r"[0-9]+", value):
            raise RuntimeError(f"non-integer telemetry field {key}={value!r}")
        fields[key] = int(value)
    return fields


def require_tail(fields: dict[str, int], stem: str) -> None:
    values = [fields[f"{stem}_{name}_ns"] for name in
              ("p50", "p95", "p99", "max")]
    if values[0] <= 0 or values != sorted(values):
        raise RuntimeError(f"invalid {stem} latency order: {values}")


def validate_owner(fields: dict[str, int]) -> None:
    expected = {
        "fiber_cached", "fiber_capacity", "region_reserved",
        "region_peak_used",
        "fiber_pooled_p50_ns", "fiber_pooled_p95_ns",
        "fiber_pooled_p99_ns", "fiber_pooled_max_ns",
        "fiber_direct_p50_ns", "fiber_direct_p95_ns",
        "fiber_direct_p99_ns", "fiber_direct_max_ns",
    }
    if set(fields) != expected:
        raise RuntimeError(
            f"owner telemetry fields differ: {sorted(set(fields) ^ expected)}")
    if not 16 <= fields["fiber_cached"] <= fields["fiber_capacity"]:
        raise RuntimeError("guarded-stack cache does not retain the warmed batch")
    if fields["region_reserved"] > 256 * 1024:
        raise RuntimeError("request Region exceeds its frozen 256 KiB retention cap")
    if fields["region_peak_used"] == 0:
        raise RuntimeError("request Region peak usage was not observed")
    require_tail(fields, "fiber_pooled")
    require_tail(fields, "fiber_direct")
    for percentile in ("p95", "p99"):
        pooled = fields[f"fiber_pooled_{percentile}_ns"]
        direct = fields[f"fiber_direct_{percentile}_ns"]
        if pooled * 100 > direct * 105:
            raise RuntimeError(
                f"pooled fiber {percentile} exceeds 5% direct-VM budget: "
                f"pooled={pooled}, direct={direct}")


def validate_handoff(fields: dict[str, int]) -> None:
    expected = {
        "capacity", "batch",
        "hyle_p50_ns", "hyle_p95_ns", "hyle_p99_ns", "hyle_max_ns",
        "marl_p50_ns", "marl_p95_ns", "marl_p99_ns", "marl_max_ns",
    }
    if set(fields) != expected:
        raise RuntimeError(
            f"handoff telemetry fields differ: {sorted(set(fields) ^ expected)}")
    if fields["capacity"] != 256 or fields["batch"] != 64:
        raise RuntimeError("handoff capacity or batch differs from the contract")
    require_tail(fields, "hyle")
    require_tail(fields, "marl")
    for percentile in ("p95", "p99"):
        candidate = fields[f"hyle_{percentile}_ns"]
        peer = fields[f"marl_{percentile}_ns"]
        if candidate * 100 > peer * 105:
            raise RuntimeError(
                f"HYLE handoff {percentile} exceeds 5% Marl budget: "
                f"hyle={candidate}, marl={peer}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("owner", "handoff"), required=True)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--build", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--expected-shapes-sha256", required=True)
    parser.add_argument("--expected-marl-source-sha256")
    parser.add_argument("--expected-marl-overlay-sha256")
    args = parser.parse_args()

    source = args.source.resolve()
    shapes_path = source / "bench/mem/consumer_shapes_v1.json"
    require_digest(shapes_path, args.expected_shapes_sha256, "consumer shapes")
    shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
    if shapes.get("schema") != "hyle.bench.consumer-shapes/v1":
        raise RuntimeError("unsupported mem consumer-shapes schema")
    for record in shapes.get("source_evidence", []):
        require_digest(source / record["path"], record["sha256"],
                       f"consumer source {record['role']}")

    if args.mode == "handoff":
        if not args.expected_marl_source_sha256 or \
                not args.expected_marl_overlay_sha256:
            raise RuntimeError("handoff mode requires both Marl hashes")
        cache = (args.build / "CMakeCache.txt").read_text(
            encoding="utf-8", errors="replace")
        match = re.search(r"^HYLE_MARL_DIR:PATH=(.+)$", cache, re.MULTILINE)
        if match is None:
            raise RuntimeError("configured HYLE_MARL_DIR is unavailable")
        marl_dir = pathlib.Path(match.group(1).strip())
        require_digest(marl_dir / "include/marl/pool.h",
                       args.expected_marl_source_sha256, "Marl pool source")
        overlays = list(args.build.rglob("marl-compat/include/marl/pool.h"))
        if len(overlays) != 1:
            raise RuntimeError(f"expected one generated Marl overlay, got {overlays}")
        require_digest(overlays[0], args.expected_marl_overlay_sha256,
                       "generated Marl pool overlay")

    prefix = OWNER_PREFIX if args.mode == "owner" else HANDOFF_PREFIX
    logs = sorted((args.output / "raw").rglob("product-*-run-*.log"))
    # Validators run once before sampling to establish delegated-control
    # identity, and again after sampling. The pre-sampling pass has no raw log.
    if not logs:
        print(f"mem {args.mode} inputs PASS (pre-sampling)")
        return 0

    checked = 0
    for path in logs:
        lines = [line for line in path.read_text(
            encoding="utf-8", errors="replace").splitlines()
                 if line.startswith(prefix)]
        if len(lines) != 1:
            raise RuntimeError(
                f"{path} contains {len(lines)} {prefix.strip()} records")
        fields = parse_fields(lines[0], prefix)
        if args.mode == "owner":
            validate_owner(fields)
        else:
            validate_handoff(fields)
        checked += 1
    print(f"mem {args.mode} resource/tail PASS logs={checked}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"mem consumer validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
