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
any requested pair it computes a same-process op/s RATIO and applies the
canonical product verdict:

    WIN/LOSE only when the paired two-sided sign test has p < 0.01 and the
    median gap is material (>1%); otherwise TIE.

Absolute per-variant spread is printed as a diagnostic, not as an alternative
verdict: machine noise is often correlated and cancels in the paired ratio.
Formal evidence also repeats the experiment with variant order swapped; a
direction that reverses under order swap cannot be signed as product truth.

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
import datetime
import hashlib
import json
import math
import os
import pathlib
import platform
import statistics
import subprocess
import sys

BT = chr(96)  # backtick; nanobench wraps variant names in `...`


def _run_text(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(path):
    """Return reproducibility metadata for one Git checkout, or None."""
    root = _run_text(["git", "rev-parse", "--show-toplevel"], cwd=path)
    if not root:
        return None
    head = _run_text(["git", "rev-parse", "HEAD"], cwd=root)
    tree = _run_text(["git", "rev-parse", "HEAD^{tree}"], cwd=root)
    status = _run_text(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root)
    if not head or not tree or status is None:
        return None
    remote = _run_text(["git", "remote", "get-url", "origin"], cwd=root)
    return {
        "root": os.path.abspath(root),
        "head": head,
        "tree": tree,
        "dirty": bool(status),
        "remote": remote or "none",
    }


def parse_peer_spec(spec):
    """Parse NAME=PATH used by --peer without guessing path delimiters."""
    if "=" not in spec:
        raise ValueError("peer must use NAME=PATH")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise ValueError("peer must use non-empty NAME=PATH")
    return name, os.path.abspath(path)


def _cmake_contract(exe):
    """Find the owning build cache and return certificate-relevant entries."""
    cache = None
    for parent in pathlib.Path(exe).resolve().parents:
        candidate = parent / "CMakeCache.txt"
        if candidate.is_file():
            cache = candidate
            break
    if cache is None:
        return None, {}
    wanted = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CXX_FLAGS_RELEASE",
        "CMAKE_GENERATOR",
        "CMAKE_GENERATOR_PLATFORM",
        "CMAKE_GENERATOR_TOOLSET",
        "HYLE_BUILD_BENCH",
        "HYLE_BUILD_TESTS",
        "HYLE_COMPONENTS",
        "HYLE_STATIC",
        "HYLE_SANITIZER",
        "HYLE_TRACE",
        "HYLE_TRACE_BACKEND",
        "HYLE_TEST_INSTRUMENTATION",
        "HYLE_SIMD_EXACT_TARGET",
        "HYLE_WERROR",
        "HYLE_BENCH_VENDOR_DIR",
    }
    values = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        if key in wanted or key.startswith("HYLE_BENCH_"):
            values[key] = value
    return str(cache), values


def _nanobench_identity(repo_root):
    lock_path = pathlib.Path(repo_root) / "cmake" / "HyleDependencyLock.json"
    header_path = pathlib.Path(repo_root) / "third_party" / "nanobench" / "nanobench.h"
    if not lock_path.is_file() or not header_path.is_file():
        return None
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dependency = next(
        (item for item in lock.get("dependencies", [])
         if item.get("name") == "nanobench"), None)
    if dependency is None:
        return None
    return {
        "version": dependency.get("version", "unknown"),
        "revision": dependency.get("revision", "unknown"),
        "header_sha256": _sha256(header_path),
    }


def print_provenance_metadata(args, exe):
    """Emit a runtime provenance snapshot and optionally require clean inputs.

    This deliberately does not claim that the executable was built from the
    observed checkouts.  That binding comes from the fresh configure/build log;
    a future embedded build manifest may make it machine-verifiable.
    """
    script = pathlib.Path(__file__).resolve()
    source = git_identity(script.parent.parent)
    issues = []
    if source is None:
        issues.append("HYLE source is not an identifiable Git checkout")
    elif source["dirty"]:
        issues.append("HYLE source checkout is dirty")

    peers = []
    for spec in args.peer:
        try:
            name, path = parse_peer_spec(spec)
        except ValueError as error:
            issues.append(f"invalid --peer {spec!r}: {error}")
            continue
        identity = git_identity(path)
        if identity is None:
            issues.append(f"peer {name!r} is not an identifiable Git checkout: {path}")
            continue
        if identity["dirty"]:
            issues.append(f"peer {name!r} checkout is dirty")
        peers.append((name, identity))

    cache_path, cache = _cmake_contract(exe)
    if cache_path is None:
        issues.append("benchmark executable has no discoverable CMakeCache.txt")
    nanobench = _nanobench_identity(source["root"] if source else script.parent.parent)
    if nanobench is None:
        issues.append("nanobench lock/header identity is unavailable")
    if args.require_clean and issues:
        sys.exit("clean-input gate rejected: " + "; ".join(issues))

    print("# HYLE_BENCH_PROVENANCE schema=1 binding=runtime-snapshot "
          f"clean={int(not issues)} require_clean={int(args.require_clean)}")
    print("# observed_at utc=" + json.dumps(
        datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(f"# runner path={json.dumps(str(script))} sha256={_sha256(script)}")
    print(f"# executable path={json.dumps(exe)} sha256={_sha256(exe)}")
    print("# host "
          f"system={json.dumps(platform.system())} "
          f"release={json.dumps(platform.release())} "
          f"machine={json.dumps(platform.machine())} "
          f"processor={json.dumps(platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER', ''))}")
    if source:
        print("# source " + " ".join(
            f"{key}={json.dumps(str(value))}" for key, value in source.items()))
    if nanobench:
        print("# nanobench " + " ".join(
            f"{key}={json.dumps(str(value))}" for key, value in nanobench.items()))
    if cache_path:
        print(f"# cmake_cache path={json.dumps(cache_path)}")
        for key in sorted(cache):
            print(f"# cmake {key}={json.dumps(cache[key])}")
    for name, identity in peers:
        print(f"# peer name={json.dumps(name)} " + " ".join(
            f"{key}={json.dumps(str(value))}" for key, value in identity.items()))
    for name in sorted(os.environ):
        if name.startswith("HYLE_") and ("BENCH" in name or "QOS" in name):
            print(f"# env {name}={json.dumps(os.environ[name])}")
    for issue in issues:
        print(f"# unclean reason={json.dumps(issue)}")
    print("# build_binding verified=0 "
          "requirement=\"attach the fresh configure/build log to formal evidence\"")
    print()


def _binom_two_sided(k, n):
    """Two-sided binomial tail P(|X - n/2| >= |k - n/2|) under H0 p=0.5, k=max(above,below).
    Small n so exact via math.comb. Used by the paired sign test: if the ratio lands the same
    side of 1.0 in k of n runs, this is the probability of that imbalance by chance."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    p = tail / (2 ** n) * 2.0  # symmetric two-sided
    return min(p, 1.0)


def paired_verdict(ratios):
    """Return the canonical same-process ratio verdict and its evidence."""
    if not ratios:
        raise ValueError("paired verdict needs at least one ratio")
    median = statistics.median(ratios)
    gap_percent = abs(median - 1.0) * 100.0
    above = sum(1 for ratio in ratios if ratio > 1.0)
    below = sum(1 for ratio in ratios if ratio < 1.0)
    effective = above + below
    p_value = (
        _binom_two_sided(max(above, below), effective)
        if effective > 0 else 1.0
    )
    if effective >= 6 and p_value < 0.01 and gap_percent > 1.0:
        verdict = "WIN" if above > below else "LOSE"
    else:
        verdict = "TIE"
    return {
        "median": median,
        "gap_percent": gap_percent,
        "above": above,
        "effective": effective,
        "p_value": p_value,
        "verdict": verdict,
    }


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
    ap.add_argument("--require-clean", action="store_true",
                    help="reject dirty/unidentified current HYLE or peer inputs; "
                         "does not bind an existing executable to those checkouts")
    ap.add_argument("--peer", action="append", default=[], metavar="NAME=PATH",
                    help="record and cleanliness-check a Git benchmark peer; repeatable")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("BASELINE", "RIVAL"),
                    default=[], help="report RIVAL/BASELINE op/s ratio; repeatable")
    args = ap.parse_args()

    if args.runs <= 0:
        ap.error("--runs must be positive")
    if args.require_clean and args.runs < 15:
        ap.error("--require-clean formal evidence requires at least 15 runs")

    # Normalize to an absolute path: on Windows, CreateProcess won't resolve a
    # forward-slash relative path like "build/bin/foo.exe" (the shell would, but
    # subprocess bypasses it). abspath fixes that and is a no-op on POSIX.
    exe = os.path.abspath(args.exe)
    if not os.path.isfile(exe):
        sys.exit(f"benchmark executable not found: {exe}")
    print_provenance_metadata(args, exe)
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
        print("    verdict: paired sign test on same-process ratios, p<0.01 and median gap>1%")
        print("    absolute combined spread is diagnostic only; formal evidence must also agree")
        print("    after swapping variant order")
    for base, rival in args.pair:
        any_hit = False
        for t in titles:
            bk, rk = (t, base), (t, rival)
            ratios = [r[rk] / r[bk] for r in runs if bk in r and rk in r]
            if not ratios:
                continue
            any_hit = True
            decision = paired_verdict(ratios)
            rmed = decision["median"]
            combined = agg[bk][1] + agg[rk][1]
            gap = decision["gap_percent"]
            # Absolute spread is a stability diagnostic, not a product verdict.
            absolute_hint = (
                "inside" if gap <= combined else
                ("outside+" if rmed > 1 else "outside-"))
            print(f"  {t[:34]:36s} {rival} / {base}")
            print(f"    ratio med {rmed:5.3f}  min {min(ratios):5.3f} max {max(ratios):5.3f}"
                  f"  gap {gap:4.1f}% vs spread {combined:4.1f}%"
                  f"  [absolute]{absolute_hint} [verdict]{decision['verdict']}"
                  f" ({decision['above']}/{decision['effective']}>1,"
                  f" p={decision['p_value']:.1e})")
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
