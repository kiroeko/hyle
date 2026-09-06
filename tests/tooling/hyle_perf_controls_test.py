#!/usr/bin/env python3
"""Adversarial canaries for replayable common CPU control evidence."""

from __future__ import annotations

import copy
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from hyle_perf.controls import (  # noqa: E402
    CONTROL_OBSERVATIONS_SCHEMA,
    control_allows_sample_escalation,
    evaluate_cpu_controls,
    validate_cpu_control_observations,
)
from hyle_perf.model import (  # noqa: E402
    MetricDirection,
    MetricSpec,
    Observation,
    ValidationError,
)


def closure_contract():
    return {
        "controls": [
            {"id": "aa", "kind": "aa",
             "parameters": {"maximum_bias_percent": 5}},
            {"id": "delta5", "kind": "known-delta",
             "parameters": {"nominal_percent": 5}},
            {"id": "delta10", "kind": "known-delta",
             "parameters": {"nominal_percent": 10}},
            {"id": "timer", "kind": "timer-floor",
             "parameters": {"minimum_workload_floor_multiple": 20}},
            {"id": "placement", "kind": "placement", "parameters": {}},
        ],
        "strata": [
            {"id": "primary", "target_variant": "primary"},
            {"id": "placement", "target_variant": "placement"},
        ],
        "workloads": [{"id": "workload"}],
    }


def raw_controls():
    def order():
        return {
            "aa": [1.0] * 15,
            "delta5": [1.05] * 15,
            "delta10": [1.10] * 15,
            "timer_floor": {
                "1": [{
                    "candidate_ns_per_op": 0.2,
                    "peer_ns_per_op": 0.2,
                } for _ in range(15)],
            },
        }
    return {
        "record": "hyle_perf_cpu_control_observations",
        "schema": CONTROL_OBSERVATIONS_SCHEMA,
        "orders": {
            "candidate-first": order(),
            "peer-first": order(),
        },
    }


def product_observations(peer_value=100_000_000.0):
    metric = MetricSpec(
        "operations-per-second", "op/s", MetricDirection.HIGHER_IS_BETTER)
    return [
        Observation(
            "experiment", "workload", stratum, run,
            100_000_000.0, peer_value, metric,
            metadata={"measurement_batch": 1})
        for stratum in ("primary", "placement")
        for run in range(1, 16)
    ]


class CpuControlsTest(unittest.TestCase):
    def test_only_boundary_crossing_aa_can_escalate(self):
        inconclusive = {
            "id": "aa",
            "gate": "FAIL",
            "details": {
                "kind": "aa",
                "orders": {
                    "candidate-first": {
                        "verdict": "INCONCLUSIVE",
                        "reason": "median-ci-crosses-decision-boundary",
                    },
                    "peer-first": {
                        "verdict": "PARITY",
                        "reason": "median-ci-inside-equivalence-band",
                    },
                },
            },
        }
        self.assertTrue(control_allows_sample_escalation(inconclusive))

        directional = copy.deepcopy(inconclusive)
        directional["details"]["orders"]["candidate-first"].update(
            verdict="LOSE", reason="confidence-interval-below-equivalence-band")
        self.assertFalse(control_allows_sample_escalation(directional))

        known_delta = copy.deepcopy(inconclusive)
        known_delta["details"]["kind"] = "known-delta"
        self.assertFalse(control_allows_sample_escalation(known_delta))

    def test_replay_passes_all_fixed_controls(self):
        results = evaluate_cpu_controls(
            closure_contract(), raw_controls(), product_observations())
        self.assertEqual(
            {row["id"]: row["gate"] for row in results},
            {"aa": "PASS", "delta5": "PASS", "delta10": "PASS",
             "timer": "PASS", "placement": "PASS"})

    def test_timer_floor_checks_peer_as_well_as_candidate(self):
        results = evaluate_cpu_controls(
            closure_contract(), raw_controls(),
            product_observations(peer_value=1_000_000_000.0))
        gates = {row["id"]: row["gate"] for row in results}
        self.assertEqual(gates["timer"], "FAIL")

    def test_known_delta_must_be_monotonic(self):
        raw = raw_controls()
        for order in raw["orders"].values():
            order["delta10"] = [1.04] * 15
        results = evaluate_cpu_controls(
            closure_contract(), raw, product_observations())
        gates = {row["id"]: row["gate"] for row in results}
        self.assertEqual(gates["delta5"], "FAIL")
        self.assertEqual(gates["delta10"], "FAIL")

    def test_raw_schema_rejects_missing_order_and_unequal_runs(self):
        raw = raw_controls()
        del raw["orders"]["peer-first"]
        with self.assertRaises(ValidationError):
            validate_cpu_control_observations(raw)
        raw = raw_controls()
        raw["orders"]["peer-first"]["aa"].pop()
        with self.assertRaises(ValidationError):
            validate_cpu_control_observations(raw)

    def test_raw_schema_rejects_nonfinite_and_unsupported_batch(self):
        raw = raw_controls()
        raw["orders"]["candidate-first"]["aa"][0] = float("nan")
        with self.assertRaises(ValidationError):
            validate_cpu_control_observations(raw)
        raw = copy.deepcopy(raw_controls())
        floors = raw["orders"]["candidate-first"]["timer_floor"]
        floors["3"] = floors.pop("1")
        with self.assertRaises(ValidationError):
            validate_cpu_control_observations(raw)


if __name__ == "__main__":
    unittest.main()
