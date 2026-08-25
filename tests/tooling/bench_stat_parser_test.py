#!/usr/bin/env python3
"""Canary for every nanobench table shape consumed by bench-stat.py."""

import importlib.util
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


if __name__ == "__main__":
    main()
