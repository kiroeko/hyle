#!/usr/bin/env python3
"""bench-stat.py -- run a nanobench exe N times and aggregate ACROSS runs.

WHY THIS EXISTS
---------------
nanobench prints one median + err% per *single* process run. For sub-nanosecond
kernels (simd dot/saxpy ~0.03-0.08 ns/op) that single err% is 1-5%, and two
variants within a few percent of each other are INDISTINGUISHABLE from noise. A
one-shot "HYLE 1.10x Highway" readout is therefore not a result -- it's a coin
flip dressed up as a number. (The SwissTable find-miss row learned this the hard
way: a one-shot "4% behind absl" turned into "statistical tie" once measured
across 12 runs.)

This script runs the exe N times, parses the op/s column per (title, variant),
and reports the MEDIAN ns/op plus the run-to-run SPREAD ((max-min)/median). For
any pair you ask about, it computes the per-run op/s RATIO and applies the
decision rule the perf doc uses:

    call a win/loss ONLY if |ratio_median - 1| > (spreadA + spreadB);
    otherwise it's a STATISTICAL TIE.

That is the honest bar: a gap smaller than the measurement spread is not a win.

USAGE
-----
    scripts/bench-stat.py <exe> [-n N] [--pair BASELINE RIVAL] ...

    # simd: is HYLE avx2 really beating Highway, or is it a tie?
    scripts/bench-stat.py build-wsl-bench/bin/hyle_simd_bench -n 15 \
        --pair "Highway avx2 (SOTA)" "HYLE avx2 (256b+FMA)"

Run under `setarch -R` on Linux if the exe is built with a sanitizer. Per the
9p rule, run from a native-ext4 checkout, never /mnt/c.
"""
import argparse
import math
import os
import statistics
import subprocess
import sys

BT = chr(96)  # backtick; nanobench wraps variant names in `...`


def _binom_two_sided(k, n):
    """Two-sided binomial tail P(|X - n/2| >= |k - n/2|) under H0 p=0.5, k=max(above,below).
    Small n so exact via math.comb. Used by the paired sign test: if the ratio lands the same
    side of 1.0 in k of n runs, this is the probability of that imbalance by chance."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    p = tail / (2 ** n) * 2.0  # symmetric two-sided
    return min(p, 1.0)


def parse_run(text):
    """One process's stdout -> {(title, variant): op/s}. Tracks the current
    table title and op/s column from each nanobench header row.  Relative
    tables prepend a `relative` column; absolute tables do not."""
    out = {}
    title = None
    ops_col = None
    for ln in text.splitlines():
        if "|" not in ln:
            continue
        cols = [c.strip() for c in ln.split("|")]
        if len(cols) < 6:
            continue
        last = cols[-1].strip()
        if cols[1] == "relative":
            title = last
            ops_col = 3
            continue
        if cols[1] == "ns/op" and len(cols) >= 6:
            title = last
            ops_col = 2
            continue
        # nanobench appends an extra warning cell after an unstable row:
        #   ... | :wavy_dash: `variant` | (Unstable with ...)
        # The old parser required the LAST cell to start with a backtick, silently
        # dropping exactly the noisy rows for which cross-process aggregation is
        # most valuable. Find the quoted variant cell instead.
        variant_cell = next((c for c in cols[5:] if BT in c), None)
        if title and ops_col is not None and variant_cell is not None:
            begin = variant_cell.find(BT)
            end = variant_cell.find(BT, begin + 1)
            if end < 0:
                continue
            variant = variant_cell[begin + 1:end]
            try:
                ops = float(cols[ops_col].replace(",", ""))
            except (IndexError, ValueError):
                raise ValueError(
                    f"nanobench row has no parseable op/s value: {ln!r}")
            key = (title, variant)
            if key in out:
                raise ValueError(f"duplicate nanobench row: {key!r}")
            out[key] = ops
    return out


def require_consistent_rows(runs):
    """Reject partial cross-process samples instead of silently lowering N."""
    if not runs or not runs[0]:
        raise ValueError("no nanobench rows parsed -- is this a nanobench exe?")
    expected = set(runs[0])
    for index, run in enumerate(runs[1:], start=2):
        actual = set(run)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"nanobench row set changed in run {index}: "
                f"missing={missing!r}, extra={extra!r}")
    return sorted(expected)


def parse_sleep_telemetry(text):
    """Parse optional time normal-bench distribution rows.

    The executable measures one warm sleep distribution per process and emits
    stable key=value telemetry outside the nanobench tables. Keeping the raw
    sampler in C++ avoids process-launch overhead in each observation; this
    aggregator then reports medians across independent processes.
    """
    out = {}
    for line in text.splitlines():
        if not line.startswith("# HYLE_SLEEP "):
            continue
        fields = {}
        for token in line.split()[2:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = value
        required = ("source", "request_ns", "n", "p50_over_ns",
                    "p95_over_ns", "p99_over_ns", "max_over_ns", "early")
        if any(key not in fields for key in required):
            continue
        key = (fields["source"], int(fields["request_ns"]))
        out[key] = {name: int(fields[name]) for name in required[2:]}
    return out


def main():
    ap = argparse.ArgumentParser(description="aggregate a nanobench exe across N runs")
    ap.add_argument("exe", help="path to the nanobench executable")
    ap.add_argument("-n", "--runs", type=int, default=15, help="process repetitions (default 15)")
    ap.add_argument("--setarch", action="store_true", help="wrap in `setarch -R` (sanitizer builds)")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("BASELINE", "RIVAL"),
                    default=[], help="report RIVAL/BASELINE op/s ratio; repeatable")
    args = ap.parse_args()

    # Normalize to an absolute path: on Windows, CreateProcess won't resolve a
    # forward-slash relative path like "build/bin/foo.exe" (the shell would, but
    # subprocess bypasses it). abspath fixes that and is a no-op on POSIX.
    exe = os.path.abspath(args.exe)
    cmd = (["setarch", "-R"] if args.setarch else []) + [exe]
    runs = []
    sleep_runs = []
    for run_index in range(1, args.runs + 1):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"bench exited {r.returncode}: {r.stderr[:400]}")
        output = r.stdout + r.stderr
        try:
            runs.append(parse_run(output))
        except ValueError as error:
            sys.exit(f"cannot parse nanobench run {run_index}: {error}")
        sleep_runs.append(parse_sleep_telemetry(output))

    try:
        keys = require_consistent_rows(runs)
    except ValueError as error:
        sys.exit(str(error))

    agg = {}
    print(f"N={args.runs} runs\n")
    for k in keys:
        vals = [r[k] for r in runs]
        med = statistics.median(vals)
        nsop = 1e9 / med
        spread = (max(vals) - min(vals)) / med * 100.0
        agg[k] = (med, spread)
        print(f"  {k[0][:30]:32s} {k[1]:34s} med {nsop:7.3f} ns/op  spread {spread:5.1f}%  ({med/1e9:7.2f} G)")

    # auto-detect titles for --pair matching (a pair is matched within each title)
    titles = []
    for t, _ in keys:
        if t not in titles:
            titles.append(t)

    if args.pair:
        print("\n=== pair ratios (RIVAL/BASELINE op/s) ===")
        print("    two verdicts: [spread] gap>combined-spread (absolute); [sign] PAIRED SIGN TEST on")
        print("    per-run ratios -- if k/N runs land the SAME side of 1.0, binomial p<0.01 => real")
        print("    win/loss EVEN when absolute spread is large (same-machine noise is CORRELATED, so")
        print("    the ratio is far tighter than either variant's absolute spread). Trust [sign].")
    for base, rival in args.pair:
        any_hit = False
        for t in titles:
            bk, rk = (t, base), (t, rival)
            ratios = [r[rk] / r[bk] for r in runs if bk in r and rk in r]
            if not ratios:
                continue
            any_hit = True
            rmed = statistics.median(ratios)
            combined = agg[bk][1] + agg[rk][1]
            gap = abs(rmed - 1.0) * 100.0
            # [spread] verdict: absolute (the old, conservative rule).
            spread_v = "TIE" if gap <= combined else ("WIN" if rmed > 1 else "LOSE")
            # [sign] verdict: paired sign test. Count runs with ratio>1 vs <1 (ignore exact ties).
            n = len(ratios)
            above = sum(1 for r in ratios if r > 1.0)
            below = sum(1 for r in ratios if r < 1.0)
            eff = above + below  # non-tied runs
            k = max(above, below)
            # two-sided binomial p under H0=0.5: P(X>=k) + P(X<=eff-k), doubled by symmetry.
            p = _binom_two_sided(k, eff) if eff > 0 else 1.0
            # sign is meaningful only if a CONSISTENT direction AND a non-trivial median gap (>1%)
            # (guards against a 0.2% "win" that is 15/15 but practically nil).
            if eff >= 6 and p < 0.01 and gap > 1.0:
                sign_v = "WIN" if above > below else "LOSE"
            else:
                sign_v = "TIE"
            # A row is a REAL non-tie if EITHER verdict is significant; flag disagreements loudly.
            flag = "  <<< SIGN-CAUGHT (spread hid it)" if sign_v != "TIE" and spread_v == "TIE" else ""
            print(f"  {t[:34]:36s} {rival} / {base}")
            print(f"    ratio med {rmed:5.3f}  min {min(ratios):5.3f} max {max(ratios):5.3f}"
                  f"  gap {gap:4.1f}% vs spread {combined:4.1f}%"
                  f"  [spread]{spread_v} [sign]{sign_v} ({above}/{eff}>1, p={p:.1e}){flag}")
        if not any_hit:
            print(f"  (no rows matched pair: {base!r} / {rival!r})")

    sleep_keys = sorted({key for run in sleep_runs for key in run})
    if sleep_keys:
        print("\n=== sleep overshoot distributions (median across processes) ===")
    for key in sleep_keys:
        samples = [run[key] for run in sleep_runs if key in run]
        source, request_ns = key
        med = lambda field: statistics.median(sample[field] for sample in samples)
        early = sum(sample["early"] for sample in samples)
        observations = sum(sample["n"] for sample in samples)
        print(f"  {source:5s} request {request_ns:9d} ns  "
              f"p50_over {med('p50_over_ns'):9.0f}  "
              f"p95_over {med('p95_over_ns'):9.0f}  "
              f"p99_over {med('p99_over_ns'):9.0f}  "
              f"max_over {med('max_over_ns'):9.0f}  "
              f"early {early}/{observations}")


if __name__ == "__main__":
    main()
