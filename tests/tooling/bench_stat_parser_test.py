#!/usr/bin/env python3
"""Canary for every nanobench table shape consumed by bench-stat.py."""

import importlib.util
import json
import math
import pathlib


def load_bench_stat():
    script = pathlib.Path(__file__).parents[2] / "scripts" / "bench-stat.py"
    spec = importlib.util.spec_from_file_location("hyle_bench_stat", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    bench_stat = load_bench_stat()
    output = """
| relative | ns/op | op/s | err% | total | relative table
|---------:|------:|-----:|-----:|------:|:---------------
| 100.0% | 2.000 | 500,000,000.000 | 1.0% | 0.010 | `relative row`

| ns/op | op/s | err% | total | absolute table
|------:|-----:|-----:|------:|:---------------
| 4.000 | 250,000,000.000 | 2.0% | 0.020 | `absolute row` | :wavy_dash: unstable
"""
    parsed = bench_stat.parse_run(output)
    expected = {
        ("relative table", "relative row"): 500_000_000.0,
        ("absolute table", "absolute row"): 250_000_000.0,
    }
    if parsed != expected:
        raise AssertionError(f"nanobench parser mismatch: {parsed!r} != {expected!r}")

    keys = bench_stat.require_consistent_rows([parsed, dict(parsed)])
    if keys != sorted(expected):
        raise AssertionError(f"unexpected stable key set: {keys!r}")

    partial = dict(parsed)
    partial.pop(("absolute table", "absolute row"))
    try:
        bench_stat.require_consistent_rows([parsed, partial])
    except ValueError as error:
        if "run 2" not in str(error) or "absolute row" not in str(error):
            raise AssertionError(f"unhelpful partial-run diagnostic: {error}")
    else:
        raise AssertionError("partial cross-process samples were accepted")

    if bench_stat.parse_peer_spec("highway=../highway") != (
            "highway", str((pathlib.Path.cwd() / "../highway").resolve())):
        raise AssertionError("NAME=PATH peer parsing changed")
    for invalid in ("highway", "=../highway", "highway="):
        try:
            bench_stat.parse_peer_spec(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid peer spec accepted: {invalid!r}")

    missing_peer = pathlib.Path.cwd() / "bench-stat-peer-that-does-not-exist"
    if bench_stat.git_identity(missing_peer) is not None:
        raise AssertionError("missing peer path acquired a Git identity")

    order_interval = bench_stat._median_confidence_interval(
        [float(value) for value in range(1, 16)])
    if order_interval != (3.0, 13.0, 3):
        raise AssertionError(f"n=15 exact order interval changed: {order_interval!r}")

    # Verdicts are from BASELINE's perspective while ratios are RIVAL/BASELINE.
    verdict_cases = (
        ([0.95] * 15, 3.0, "WIN"),
        ([1.05] * 15, 3.0, "LOSE"),
        ([1.005] * 15, 1.0, "EQUIVALENT"),
        ([1.02] * 8 + [0.98] * 7, 1.0, "INCONCLUSIVE"),
        ([1.0] * 15, 1.0, "EQUIVALENT"),
        ([0.95] * 12 + [1.0] * 3, 3.0, "INCONCLUSIVE"),
        ([0.95] * 13 + [1.0] * 2, 3.0, "WIN"),
    )
    for ratios, margin, expected_verdict in verdict_cases:
        decision = bench_stat.paired_verdict(ratios, margin)
        if decision["verdict"] != expected_verdict:
            raise AssertionError(
                f"paired verdict mismatch: {decision!r} != {expected_verdict!r}"
            )

    undeclared = bench_stat.paired_verdict([1.0] * 15)
    if (undeclared["verdict"], undeclared["reason"]) != (
            "INCONCLUSIVE", "margin-not-declared"):
        raise AssertionError(f"undeclared margin acquired a claim: {undeclared!r}")
    too_few = bench_stat.paired_verdict([1.0] * 14, 1.0)
    if (too_few["verdict"], too_few["reason"]) != (
            "INCONCLUSIVE", "insufficient-samples"):
        raise AssertionError(f"short sample acquired a claim: {too_few!r}")

    margin = 3.0
    lower_boundary = 1.0 / (1.0 + margin / 100.0)
    upper_boundary = 1.0 + margin / 100.0
    for boundary in (lower_boundary, upper_boundary):
        decision = bench_stat.paired_verdict([boundary] * 15, margin)
        if decision["verdict"] != "EQUIVALENT":
            raise AssertionError(f"equivalence boundary was made material: {decision!r}")
    below = math.nextafter(lower_boundary, 0.0)
    above = math.nextafter(upper_boundary, math.inf)
    if bench_stat.paired_verdict([below] * 15, margin)["verdict"] != "WIN":
        raise AssertionError("ratio just below the lower boundary did not WIN")
    if bench_stat.paired_verdict([above] * 15, margin)["verdict"] != "LOSE":
        raise AssertionError("ratio just above the upper boundary did not LOSE")

    forward = bench_stat.paired_verdict([0.95] * 15, margin)
    reciprocal = bench_stat.paired_verdict([1.0 / 0.95] * 15, margin)
    if (forward["verdict"], reciprocal["verdict"]) != ("WIN", "LOSE"):
        raise AssertionError(
            f"swapping candidate and peer was not reciprocal: {forward!r}/{reciprocal!r}")
    if not math.isclose(forward["band"][0], 1.0 / forward["band"][1]):
        raise AssertionError(f"equivalence band is not reciprocal: {forward['band']!r}")
    equivalent = bench_stat.paired_verdict([1.005] * 15, margin)
    inverse_equivalent = bench_stat.paired_verdict(
        [1.0 / 1.005] * 15, margin)
    if (equivalent["verdict"], inverse_equivalent["verdict"]) != (
            "EQUIVALENT", "EQUIVALENT"):
        raise AssertionError(
            "equivalence changed when candidate and peer were swapped")

    informational = bench_stat.paired_verdict(
        [0.1] * 15, margin, comparable=False)
    if (informational["verdict"], informational["reason"]) != (
            "NOT_COMPARABLE", "comparison-contract-not-equivalent"):
        raise AssertionError(
            f"unequal contract acquired a performance claim: {informational!r}")

    # The one-sided SOTA gate is independent from the exact five-state
    # verdict, but uses the same predeclared upper limit U=1+m.
    win_equivalent_crossing = [0.95] * 8 + [1.0] * 7
    exact = bench_stat.paired_verdict(win_equivalent_crossing, margin)
    sota = bench_stat.sota_noninferiority_gate(
        win_equivalent_crossing, margin)
    if exact["verdict"] != "INCONCLUSIVE" or sota["gate"] != "PASS":
        raise AssertionError(
            "an exact WIN/EQUIVALENT crossing did not retain INCONCLUSIVE "
            f"while passing non-inferiority: {exact!r}/{sota!r}")

    sota_cases = (
        ([0.95] * 15, margin, True, "PASS"),
        ([1.0] * 15, margin, True, "PASS"),
        ([upper_boundary] * 15, margin, True, "PASS"),
        ([1.05] * 15, margin, True, "FAIL"),
        ([1.0] * 8 + [1.05] * 7, margin, True, "UNRESOLVED"),
        ([1.0] * 14, margin, True, "UNRESOLVED"),
        ([1.0] * 15, None, True, "UNRESOLVED"),
        ([0.1] * 15, margin, False, "N/A"),
    )
    for ratios, declared_margin, comparable, expected_gate in sota_cases:
        gate = bench_stat.sota_noninferiority_gate(
            ratios, declared_margin, comparable=comparable)
        if gate["gate"] != expected_gate:
            raise AssertionError(
                f"SOTA gate mismatch: {gate!r} != {expected_gate!r}")

    if bench_stat.combine_sota_gates("PASS", "PASS") != "PASS":
        raise AssertionError("two passing orders did not pass SOTA gate")
    if bench_stat.combine_sota_gates(
            "PASS", "PASS", shape="PASS") != "PASS":
        raise AssertionError("passing required shape gate did not compose")
    for gates in (("PASS", "UNRESOLVED", None),
                  ("PASS", "PASS", "UNRESOLVED")):
        if bench_stat.combine_sota_gates(*gates[:2], shape=gates[2]) != "UNRESOLVED":
            raise AssertionError(f"unresolved SOTA gate acquired PASS: {gates!r}")
    if bench_stat.combine_sota_gates("PASS", "FAIL") != "FAIL":
        raise AssertionError("a failing execution order did not block SOTA")
    if bench_stat.combine_sota_gates(
            "PASS", "PASS", comparable=False) != "N/A":
        raise AssertionError("a non-comparable contract acquired a SOTA gate")
    if bench_stat.combine_sota_gates("N/A", "N/A") != "N/A":
        raise AssertionError("not-comparable SOTA gates lost N/A")

    pair_runs = [
        {("consumer N=32", "HYLE"): 100.0,
         ("consumer N=32", "peer"): 95.0},
        {("consumer N=32", "HYLE"): 120.0,
         ("consumer N=32", "peer"): 126.0},
    ]
    records = bench_stat.pair_process_records(
        pair_runs, "consumer N=32", "HYLE", "peer")
    if len(records) != 2:
        raise AssertionError(f"raw process records lost samples: {records!r}")
    expected_raw = (
        (1, 100.0, 95.0, 0.95),
        (2, 120.0, 126.0, 1.05),
    )
    for record, expected_values in zip(records, expected_raw):
        actual_values = (
            record["run"], record["baseline_ops_per_second"],
            record["rival_ops_per_second"],
            record["ratio_rival_over_baseline"],
        )
        if actual_values != expected_values:
            raise AssertionError(
                f"raw paired process values changed: {record!r}")
        encoded = bench_stat.format_pair_process_record(record)
        if not encoded.startswith(bench_stat.PAIR_RAW_PREFIX):
            raise AssertionError(f"raw JSON marker missing: {encoded!r}")
        decoded = json.loads(encoded[len(bench_stat.PAIR_RAW_PREFIX):])
        if decoded != record:
            raise AssertionError(
                f"raw JSON did not round-trip: {decoded!r} != {record!r}")

    for verdict in ("WIN", "EQUIVALENT", "LOSE"):
        if bench_stat.combine_order_verdicts(verdict, verdict) != verdict:
            raise AssertionError(f"stable order verdict lost: {verdict}")
    for first, second in (("WIN", "EQUIVALENT"), ("WIN", "LOSE"),
                          ("INCONCLUSIVE", "INCONCLUSIVE")):
        if bench_stat.combine_order_verdicts(first, second) != "INCONCLUSIVE":
            raise AssertionError(f"unstable orders acquired a claim: {first}/{second}")
    if bench_stat.combine_order_verdicts(
            "WIN", "WIN", shape_stable=False) != "INCONCLUSIVE":
        raise AssertionError("placement-sensitive orders acquired a claim")
    if bench_stat.combine_order_verdicts(
            "WIN", "WIN", comparable=False) != "NOT_COMPARABLE":
        raise AssertionError("unequal contract acquired a combined claim")
    if bench_stat.combine_order_verdicts(
            "NOT_COMPARABLE", "NOT_COMPARABLE") != "NOT_COMPARABLE":
        raise AssertionError("not-comparable orders lost their contract verdict")

    for invalid_ratios in ([0.0] * 15, [float("nan")] * 15,
                           [float("inf")] * 15):
        try:
            bench_stat.paired_verdict(invalid_ratios, 1.0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid ratios acquired a verdict: {invalid_ratios!r}")

    try:
        bench_stat.paired_verdict([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty paired sample acquired a verdict")


if __name__ == "__main__":
    main()
