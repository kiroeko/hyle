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
import os
import statistics
import subprocess
import sys

BT = chr(96)  # backtick; nanobench wraps variant names in `...`


def parse_run(text):
    """One process's stdout -> {(title, variant): op/s}. Tracks the current
    table title from each header row (the `relative` line carries it last)."""
    out = {}
    title = None
    for ln in text.splitlines():
        if "|" not in ln:
            continue
        cols = [c.strip() for c in ln.split("|")]
        if len(cols) < 6:
            continue
        last = cols[-1]
        if cols[1] == "relative":          # header row: title is in the last col
            title = last
            continue
        if title and last.startswith(BT):  # data row: `variant name`
            variant = last.strip(BT)
            try:
                ops = float(cols[3].replace(",", ""))
            except ValueError:
                continue
            out[(title, variant)] = ops
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
    for _ in range(args.runs):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"bench exited {r.returncode}: {r.stderr[:400]}")
        runs.append(parse_run(r.stdout))

    keys = sorted({k for r in runs for k in r})
    if not keys:
        sys.exit("no nanobench rows parsed -- is this a nanobench exe?")

    agg = {}
    print(f"N={args.runs} runs\n")
    for k in keys:
        vals = [r[k] for r in runs if k in r]
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
        print("\n=== pair ratios (RIVAL/BASELINE op/s; WIN/LOSE only if gap > combined spread) ===")
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
            if gap <= combined:
                verdict = "TIE(stat)"
            elif rmed > 1:
                verdict = "WIN"
            else:
                verdict = "LOSE"
            print(f"  {t[:34]:36s} {rival} / {base}")
            print(f"    ratio med {rmed:5.3f}  min {min(ratios):5.3f} max {max(ratios):5.3f}"
                  f"  gap {gap:4.1f}% vs spread {combined:4.1f}%  -> {verdict}")
        if not any_hit:
            print(f"  (no rows matched pair: {base!r} / {rival!r})")


if __name__ == "__main__":
    main()
