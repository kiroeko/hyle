#!/usr/bin/env python3
"""Regression tests for paired estimates and benchmark log statistics."""
import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from hyle_perf.model import (ExactVerdict, MetricDirection, MetricSpec,
                             NonInferiorityGate, Observation, ValidationError)
from hyle_perf.stats import (clustered_bootstrap_median_estimate,
    combine_exact_verdicts, combine_noninferiority_gates, exact_decision,
    from_legacy_rival_over_candidate_throughput, noninferiority_decision,
    normalize_exact_verdict, process_median_estimate)

class StatisticsTest(unittest.TestCase):
    def test_advantage_ratio_directions_and_four_states(self):
        higher = Observation(
            "exp", "work", "stratum", 1, 120.0, 100.0,
            MetricSpec("throughput", "op/s", MetricDirection.HIGHER_IS_BETTER))
        lower = Observation(
            "exp", "work", "stratum", 1, 10.0, 12.0,
            MetricSpec("latency", "ns", MetricDirection.LOWER_IS_BETTER))
        self.assertAlmostEqual(higher.advantage_ratio, 1.2)
        self.assertAlmostEqual(lower.advantage_ratio, 1.2)
        with self.assertRaisesRegex(ValidationError, "MetricDirection"):
            MetricSpec("latency", "ns", "lower-is-better")

        cases = (
            ([1.05] * 15, ExactVerdict.WIN),
            ([1.0] * 15, ExactVerdict.PARITY),
            ([0.95] * 15, ExactVerdict.LOSE),
            ([0.95] * 7 + [1.05] * 8, ExactVerdict.INCONCLUSIVE),
        )
        for ratios, verdict in cases:
            estimate = process_median_estimate(ratios)
            self.assertIs(exact_decision(estimate, 3.0).verdict, verdict)
        short = process_median_estimate([1.0] * 14)
        self.assertIs(
            exact_decision(short, 3.0).verdict, ExactVerdict.INCONCLUSIVE)

    def test_noninferiority_is_independent_and_composition_is_conservative(self):
        mixed_win_parity = process_median_estimate([0.99] * 8 + [1.05] * 7)
        self.assertIs(
            exact_decision(mixed_win_parity, 3.0).verdict,
            ExactVerdict.INCONCLUSIVE)
        self.assertIs(
            noninferiority_decision(mixed_win_parity, 3.0).gate,
            NonInferiorityGate.PASS)
        self.assertIs(
            combine_exact_verdicts(["WIN", "PARITY"]),
            ExactVerdict.INCONCLUSIVE)
        self.assertIs(
            combine_exact_verdicts(["EQUIVALENT", "TIE"]),
            ExactVerdict.PARITY)
        self.assertIs(
            combine_noninferiority_gates(["PASS", "FAIL"]),
            NonInferiorityGate.FAIL)

    def test_clustered_estimator_is_deterministic_and_counts_clusters(self):
        clusters = [[1.04, 1.05, 1.06, 1.05] for _ in range(15)]
        first = clustered_bootstrap_median_estimate(
            clusters, rounds=200, seed_key="stable")
        second = clustered_bootstrap_median_estimate(
            clusters, rounds=200, seed_key="stable")
        self.assertEqual(first, second)
        self.assertEqual(first.sample_count, 60)
        self.assertEqual(first.independent_clusters, 15)
        self.assertIs(exact_decision(first, 3.0).verdict, ExactVerdict.WIN)

    def test_existing_bench_stat_raw_ratios_have_equivalent_v1_decisions(self):
        path = ROOT / "scripts" / "bench-stat.py"
        spec = importlib.util.spec_from_file_location("legacy_bench_stat", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        cases = (
            ([0.95] * 15, 3.0),
            ([1.05] * 15, 3.0),
            ([1.005] * 15, 1.0),
            ([1.02] * 8 + [0.98] * 7, 1.0),
        )
        for raw_ratios, margin in cases:
            old = legacy.paired_verdict(raw_ratios, margin)["verdict"]
            advantage = [from_legacy_rival_over_candidate_throughput(value)
                         for value in raw_ratios]
            new = exact_decision(
                process_median_estimate(advantage), margin).verdict
            self.assertIs(new, normalize_exact_verdict(old))

if __name__ == "__main__":
    unittest.main()
