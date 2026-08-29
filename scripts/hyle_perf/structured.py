"""Structured paired-process measurement adapter for HYLE perf v1.

The protocol is intentionally small: a child emits one JSON object per
measured row, while the central runner owns process pairing, controls,
statistics, stopping and evidence.  Concurrent benchmarks must also report
enough work and affinity telemetry to prove that a process mask did not merely
allow, but actually placed, every worker on its assigned physical core.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

from .model import ValidationError, require_identifier


STRUCTURED_MEASUREMENT_SCHEMA = "hyle.perf.structured-measurement/v1"
STRUCTURED_MEASUREMENT_RECORD = "hyle_perf_structured_measurement"

_REQUIRED_FIELDS = {
    "record",
    "schema",
    "row_id",
    "value",
    "worker_count",
    "work_items_per_worker",
    "work_item_count",
    "fanout",
    "callback_count",
    "expected_callback_count",
    "affinity_ok",
    "requested_logical_cpus",
    "observed_logical_cpus",
    "worker_singleton_masks",
}
_OPTIONAL_FIELDS = {"diagnostics"}


def _positive_number(value: Any, field: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value)) or float(value) <= 0.0):
        raise ValidationError(f"{field} must be finite and positive")
    return float(value)


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{field} must be a positive integer")
    return value


def _logical_cpus(value: Any, field: str, worker_count: int) -> list[int]:
    if (not isinstance(value, list) or len(value) != worker_count or
            any(isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in value) or len(set(value)) != len(value)):
        raise ValidationError(
            f"{field} must contain one distinct non-negative CPU per worker")
    return list(value)


def _singleton_masks(value: Any, cpus: list[int], field: str) -> list[str]:
    expected = [hex(1 << cpu) for cpu in cpus]
    if not isinstance(value, list) or value != expected:
        raise ValidationError(
            f"{field} must equal the requested per-worker singleton masks")
    return list(value)


def _normalize_row(
        raw: Mapping[str, Any], *, child_policy: Mapping[str, Any] | None,
        field: str) -> dict[str, Any]:
    unknown = set(raw) - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
    missing = _REQUIRED_FIELDS - set(raw)
    if unknown or missing:
        raise ValidationError(
            f"{field} fields differ: missing={sorted(missing)!r}, "
            f"unknown={sorted(unknown)!r}")
    if (raw["record"] != STRUCTURED_MEASUREMENT_RECORD or
            raw["schema"] != STRUCTURED_MEASUREMENT_SCHEMA):
        raise ValidationError(f"{field} has an unsupported record/schema")
    row_id = require_identifier(raw["row_id"], field + ".row_id")
    worker_count = _positive_integer(
        raw["worker_count"], field + ".worker_count")
    per_worker = _positive_integer(
        raw["work_items_per_worker"], field + ".work_items_per_worker")
    work_items = _positive_integer(
        raw["work_item_count"], field + ".work_item_count")
    fanout = _positive_integer(raw["fanout"], field + ".fanout")
    callbacks = _positive_integer(
        raw["callback_count"], field + ".callback_count")
    expected_callbacks = _positive_integer(
        raw["expected_callback_count"], field + ".expected_callback_count")
    if work_items != worker_count * per_worker:
        raise ValidationError(f"{field} work-item accounting is inconsistent")
    if expected_callbacks != work_items * fanout or callbacks != expected_callbacks:
        raise ValidationError(f"{field} callback accounting is inconsistent")
    if raw["affinity_ok"] is not True:
        raise ValidationError(f"{field} did not prove worker affinity")
    requested = _logical_cpus(
        raw["requested_logical_cpus"],
        field + ".requested_logical_cpus", worker_count)
    observed = _logical_cpus(
        raw["observed_logical_cpus"],
        field + ".observed_logical_cpus", worker_count)
    if observed != requested:
        raise ValidationError(f"{field} observed worker CPUs differ from requested")
    masks = _singleton_masks(
        raw["worker_singleton_masks"], requested,
        field + ".worker_singleton_masks")
    if child_policy is not None:
        expected_count = child_policy.get("sampling_worker_count")
        expected_cpus = child_policy.get("worker_logical_cpus")
        expected_masks = child_policy.get("worker_singleton_masks")
        if expected_count != worker_count:
            raise ValidationError(
                f"{field} worker count differs from the child policy")
        if expected_cpus != requested or expected_masks != masks:
            raise ValidationError(
                f"{field} worker topology differs from the child policy")
    diagnostics = raw.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        raise ValidationError(f"{field}.diagnostics must be an object")
    try:
        json.dumps(diagnostics, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"{field}.diagnostics must contain finite JSON values") from error
    return {
        "record": STRUCTURED_MEASUREMENT_RECORD,
        "schema": STRUCTURED_MEASUREMENT_SCHEMA,
        "row_id": row_id,
        "value": _positive_number(raw["value"], field + ".value"),
        "worker_count": worker_count,
        "work_items_per_worker": per_worker,
        "work_item_count": work_items,
        "fanout": fanout,
        "callback_count": callbacks,
        "expected_callback_count": expected_callbacks,
        "affinity_ok": True,
        "requested_logical_cpus": requested,
        "observed_logical_cpus": observed,
        "worker_singleton_masks": masks,
        "diagnostics": dict(diagnostics),
    }


def parse_structured_measurements(
        text: str, *, child_policy: Mapping[str, Any] | None = None
        ) -> dict[str, dict[str, Any]]:
    """Parse and validate every structured measurement row in one child log."""
    rows: dict[str, dict[str, Any]] = {}
    candidates = [line for line in text.splitlines() if line.startswith("{")]
    if not candidates:
        raise ValidationError("structured benchmark emitted no measurement rows")
    for index, line in enumerate(candidates):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValidationError(
                f"structured benchmark row {index} is invalid JSON") from error
        if not isinstance(raw, Mapping):
            raise ValidationError(
                f"structured benchmark row {index} must be an object")
        row = _normalize_row(
            raw, child_policy=child_policy,
            field=f"structured benchmark row {index}")
        if row["row_id"] in rows:
            raise ValidationError(
                f"structured benchmark repeats row {row['row_id']!r}")
        rows[row["row_id"]] = row
    return rows


def require_consistent_structured_row_sets(
        parsed_runs: Iterable[Mapping[str, Any]]) -> None:
    expected: set[str] | None = None
    count = 0
    for run in parsed_runs:
        row_ids = set(run)
        if expected is None:
            expected = row_ids
        elif row_ids != expected:
            raise ValidationError(
                "structured benchmark row set changed between processes")
        count += 1
    if count == 0:
        raise ValidationError("structured benchmark has no process samples")
