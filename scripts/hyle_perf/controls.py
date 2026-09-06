"""Pure, replayable control evaluation for the CPU nanobench adapter.

The runner extracts structured values from raw child logs.  This module owns
their schema and verdicts so evidence verification can replay the same result
without trusting gate strings written by the runner.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

from .model import ExactVerdict, Observation, ValidationError
from .stats import exact_decision, process_median_estimate


CONTROL_OBSERVATIONS_SCHEMA = "hyle.perf.cpu-control-observations/v1"
CONTROL_ORDERS = ("candidate-first", "peer-first")
SUPPORTED_TIMER_BATCHES = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
AA_MAXIMUM_BIAS_PERCENT = 5.0
KNOWN_DELTA_TOLERANCE_PERCENT = 50.0
TIMER_FLOOR_MULTIPLE = 20.0
MINIMUM_CONTROL_RUNS = 15


def _positive(value: Any, field: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value)) or float(value) <= 0.0):
        raise ValidationError(f"{field} must be finite and positive")
    return float(value)


def _ratio_list(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{field} must be a non-empty array")
    return [_positive(item, f"{field}[]") for item in value]


def validate_cpu_control_observations(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize replayable raw values from common controls."""
    if not isinstance(record, Mapping):
        raise ValidationError("control observations must be an object")
    data = dict(record)
    if set(data) != {"record", "schema", "orders"}:
        raise ValidationError("control observations have missing or unknown fields")
    if (data["record"] != "hyle_perf_cpu_control_observations" or
            data["schema"] != CONTROL_OBSERVATIONS_SCHEMA):
        raise ValidationError("control observations record/schema is unsupported")
    if not isinstance(data["orders"], Mapping) or \
            set(data["orders"]) != set(CONTROL_ORDERS):
        raise ValidationError("control observations require both fixed orders")

    normalized_orders: dict[str, Any] = {}
    expected_runs: int | None = None
    expected_batches: set[str] | None = None
    for order in CONTROL_ORDERS:
        raw_order = data["orders"][order]
        if not isinstance(raw_order, Mapping):
            raise ValidationError(f"control order {order!r} must be an object")
        order_record = dict(raw_order)
        if set(order_record) != {"aa", "delta5", "delta10", "timer_floor"}:
            raise ValidationError(f"control order {order!r} has invalid fields")
        normalized = {
            key: _ratio_list(order_record[key], f"controls.{order}.{key}")
            for key in ("aa", "delta5", "delta10")
        }
        run_counts = {len(values) for values in normalized.values()}
        if len(run_counts) != 1:
            raise ValidationError(f"control order {order!r} has unequal run counts")
        order_runs = next(iter(run_counts))
        if expected_runs is None:
            expected_runs = order_runs
        elif expected_runs != order_runs:
            raise ValidationError("control fixed orders have unequal run counts")

        floors = order_record["timer_floor"]
        if not isinstance(floors, Mapping) or not floors:
            raise ValidationError(
                f"controls.{order}.timer_floor must be a non-empty object")
        normalized_floors: dict[str, list[dict[str, float]]] = {}
        for batch_text, rows in floors.items():
            if (not isinstance(batch_text, str) or not batch_text.isdigit() or
                    int(batch_text) not in SUPPORTED_TIMER_BATCHES):
                raise ValidationError(
                    f"controls.{order}.timer_floor has unsupported batch")
            if not isinstance(rows, list) or len(rows) != order_runs:
                raise ValidationError(
                    f"controls.{order}.timer_floor[{batch_text}] has wrong run count")
            normalized_rows = []
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping) or set(row) != {
                        "candidate_ns_per_op", "peer_ns_per_op"}:
                    raise ValidationError(
                        f"timer floor row {batch_text}/{index} has invalid fields")
                normalized_rows.append({
                    "candidate_ns_per_op": _positive(
                        row["candidate_ns_per_op"], "timer candidate ns/op"),
                    "peer_ns_per_op": _positive(
                        row["peer_ns_per_op"], "timer peer ns/op"),
                })
            normalized_floors[batch_text] = normalized_rows
        batch_set = set(normalized_floors)
        if expected_batches is None:
            expected_batches = batch_set
        elif expected_batches != batch_set:
            raise ValidationError("control fixed orders have different timer batches")
        normalized["timer_floor"] = normalized_floors
        normalized_orders[order] = normalized
    return {
        "record": "hyle_perf_cpu_control_observations",
        "schema": CONTROL_OBSERVATIONS_SCHEMA,
        "orders": normalized_orders,
    }


def _control_id(contract: Mapping[str, Any], kind: str, *,
                nominal: float | None = None) -> str | None:
    matches = []
    for control in contract["controls"]:
        if control["kind"] != kind:
            continue
        if nominal is not None and float(
                control["parameters"].get("nominal_percent", -1)) != nominal:
            continue
        matches.append(control["id"])
    if not matches:
        return None
    if len(matches) != 1:
        raise ValidationError(f"control family {kind!r} must occur exactly once")
    return matches[0]


def _estimate_details(values: Mapping[str, list[float]]) -> dict[str, Any]:
    return {
        order: process_median_estimate(values[order]).to_dict()
        for order in CONTROL_ORDERS
    }


def control_allows_sample_escalation(result: Mapping[str, Any]) -> bool:
    """Whether a required control is ready or only lacks n=15 precision.

    A/A is itself an estimator. If one fixed order's interval merely crosses
    the already-frozen equivalence boundary, the predeclared n=31 stage is the
    correct next observation. Actual A/A direction, known-delta, timer,
    placement, delegated or malformed failures remain hard stops.
    """
    if result.get("gate") == "PASS":
        return True
    if result.get("gate") != "FAIL":
        return False
    details = result.get("details")
    if not isinstance(details, Mapping) or details.get("kind") != "aa":
        return False
    orders = details.get("orders")
    if not isinstance(orders, Mapping) or set(orders) != set(CONTROL_ORDERS):
        return False

    needs_more_samples = False
    for order in CONTROL_ORDERS:
        decision = orders[order]
        if not isinstance(decision, Mapping):
            return False
        verdict = decision.get("verdict")
        if verdict == ExactVerdict.PARITY.value:
            continue
        if (verdict == ExactVerdict.INCONCLUSIVE.value and
                decision.get("reason") ==
                "median-ci-crosses-decision-boundary"):
            needs_more_samples = True
            continue
        return False
    return needs_more_samples


def evaluate_cpu_controls(
        contract: Mapping[str, Any], raw_record: Mapping[str, Any],
        observations: Iterable[Observation]) -> list[dict[str, Any]]:
    """Recompute every common CPU control verdict from structured raw values."""
    raw = validate_cpu_control_observations(raw_record)
    product = list(observations)
    results: list[dict[str, Any]] = []
    known_delta_medians: dict[float, dict[str, float]] = {}

    aa_id = _control_id(contract, "aa")
    if aa_id is not None:
        ratios = {order: raw["orders"][order]["aa"] for order in CONTROL_ORDERS}
        decisions = {
            order: exact_decision(
                process_median_estimate(values), AA_MAXIMUM_BIAS_PERCENT,
                minimum_independent_samples=MINIMUM_CONTROL_RUNS)
            for order, values in ratios.items()
        }
        results.append({
            "id": aa_id,
            "gate": ("PASS" if all(
                decision.verdict is ExactVerdict.PARITY
                for decision in decisions.values()) else "FAIL"),
            "details": {
                "kind": "aa",
                "maximum_bias_percent": AA_MAXIMUM_BIAS_PERCENT,
                "orders": {order: decision.to_dict()
                           for order, decision in decisions.items()},
            },
        })

    for nominal, key in ((5.0, "delta5"), (10.0, "delta10")):
        control_id = _control_id(contract, "known-delta", nominal=nominal)
        if control_id is None:
            continue
        ratios = {order: raw["orders"][order][key] for order in CONTROL_ORDERS}
        estimates = {order: process_median_estimate(values)
                     for order, values in ratios.items()}
        lower_expected = 1.0 + nominal / 100.0 * (
            1.0 - KNOWN_DELTA_TOLERANCE_PERCENT / 100.0)
        upper_expected = 1.0 + nominal / 100.0 * (
            1.0 + KNOWN_DELTA_TOLERANCE_PERCENT / 100.0)
        gate = "PASS"
        for order, estimate in estimates.items():
            if (len(ratios[order]) < MINIMUM_CONTROL_RUNS or
                    not estimate.has_interval or
                    estimate.confidence_lower is None or
                    estimate.confidence_lower <= 1.0 or
                    not lower_expected <= estimate.median <= upper_expected):
                gate = "FAIL"
        known_delta_medians[nominal] = {
            order: estimate.median for order, estimate in estimates.items()}
        results.append({
            "id": control_id,
            "gate": gate,
            "details": {
                "kind": "known-delta",
                "nominal_percent": nominal,
                "relative_tolerance_percent": KNOWN_DELTA_TOLERANCE_PERCENT,
                "accepted_median_band": [lower_expected, upper_expected],
                "orders": {order: estimate.to_dict()
                           for order, estimate in estimates.items()},
            },
        })

    if 5.0 in known_delta_medians and 10.0 in known_delta_medians:
        monotonic = all(
            known_delta_medians[10.0][order] > known_delta_medians[5.0][order]
            for order in CONTROL_ORDERS)
        for result in results:
            if result["details"].get("kind") == "known-delta":
                result["details"]["monotonic_5_to_10"] = monotonic
                if not monotonic:
                    result["gate"] = "FAIL"

    timer_id = _control_id(contract, "timer-floor")
    if timer_id is not None:
        workload_rows = []
        timer_gate = "PASS"
        workload_ids = [workload["id"] for workload in contract["workloads"]]
        for workload_id in workload_ids:
            group = [item for item in product if item.workload_id == workload_id]
            if not group:
                raise ValidationError(
                    f"timer-floor lacks product observations for {workload_id!r}")
            batches = {item.metadata.get("measurement_batch") for item in group}
            if len(batches) != 1:
                raise ValidationError(
                    f"workload {workload_id!r} has no unique measurement_batch")
            batch = next(iter(batches))
            if (isinstance(batch, bool) or not isinstance(batch, int) or
                    batch not in SUPPORTED_TIMER_BATCHES):
                raise ValidationError(
                    f"workload {workload_id!r} has unsupported measurement_batch")
            batch_key = str(batch)
            floor_sides = []
            floor_details = {}
            for order in CONTROL_ORDERS:
                floor_rows = raw["orders"][order]["timer_floor"].get(batch_key)
                if floor_rows is None:
                    raise ValidationError(
                        f"timer-floor has no batch={batch} data")
                candidate_median = statistics.median(
                    row["candidate_ns_per_op"] for row in floor_rows)
                peer_median = statistics.median(
                    row["peer_ns_per_op"] for row in floor_rows)
                floor_sides.extend((candidate_median, peer_median))
                floor_details[order] = {
                    "candidate_median_ns_per_op": candidate_median,
                    "peer_median_ns_per_op": peer_median,
                }
            conservative_floor = max(floor_sides)
            candidate_cost = statistics.median(
                1e9 / item.candidate_value for item in group)
            peer_cost = statistics.median(
                1e9 / item.peer_value for item in group)
            observed_multiple = min(candidate_cost, peer_cost) / conservative_floor
            gate = "PASS" if observed_multiple >= TIMER_FLOOR_MULTIPLE else "FAIL"
            if gate != "PASS":
                timer_gate = "FAIL"
            workload_rows.append({
                "id": workload_id,
                "gate": gate,
                "measurement_batch": batch,
                "candidate_median_ns_per_op": candidate_cost,
                "peer_median_ns_per_op": peer_cost,
                "conservative_floor_ns_per_op": conservative_floor,
                "observed_multiple": observed_multiple,
                "orders": floor_details,
            })
        results.append({
            "id": timer_id,
            "gate": timer_gate,
            "details": {
                "kind": "timer-floor",
                "minimum_workload_floor_multiple": TIMER_FLOOR_MULTIPLE,
                "workloads": workload_rows,
            },
        })

    for control in contract["controls"]:
        if control["kind"] == "placement":
            variants = {
                stratum.get("target_variant", "primary")
                for stratum in contract["strata"]}
            results.append({
                "id": control["id"],
                "gate": "PASS" if len(variants) >= 2 else "FAIL",
                "details": {"kind": "placement",
                            "target_variants": sorted(variants)},
            })
        elif control["kind"] == "host":
            results.append({
                "id": control["id"], "gate": "PASS",
                "details": {"kind": "host",
                            "delegated_to": "machine-profile-gate"},
            })
        elif control["kind"] in {"codegen", "resource", "phase-floor"}:
            results.append({
                "id": control["id"], "gate": "UNRESOLVED",
                "details": {
                    "kind": control["kind"],
                    "delegated_validator_id":
                        control["parameters"].get("validator_id"),
                },
            })

    declared_order = [control["id"] for control in contract["controls"]]
    by_id = {result["id"]: result for result in results}
    if set(by_id) != set(declared_order):
        raise ValidationError("control evaluator did not cover the contract exactly")
    return [by_id[control_id] for control_id in declared_order]
