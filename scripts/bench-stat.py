#!/usr/bin/env python3
"""bench-stat.py -- run a nanobench exe N times and aggregate ACROSS runs.

WHY THIS EXISTS
---------------
nanobench prints one median + err% per *single* process run. For sub-nanosecond
kernels (simd dot/saxpy ~0.03-0.08 ns/op) that single err% is 1-5%, and two
variants within a few percent of each other are INDISTINGUISHABLE from noise. A
one-shot "HYLE 1.10x Highway" readout is therefore not a result -- it's a coin
flip dressed up as a number. (The SwissTable find-miss row learned this the hard
way: a one-shot "4% behind absl" became unresolved once measured across
independent processes.)

This script runs the exe N times, parses the op/s column per (title, variant),
and reports the MEDIAN ns/op plus the run-to-run SPREAD ((max-min)/median). For
any requested pair it computes a same-process op/s RATIO and a provisional,
single-order statistical classification. A formal classification requires a
workload-specific equivalence margin declared before sampling:

    WIN / EQUIVALENT / LOSE / INCONCLUSIVE

The classification uses a distribution-free confidence interval for the
population median paired ratio. WIN and LOSE are from BASELINE's perspective;
BASELINE is the HYLE candidate and RIVAL is its peer. It never turns "not
significant" into equivalence. A deliberately informational, unequal-contract
pair is classified NOT_COMPARABLE before statistical evidence is considered.

The same predeclared margin also drives a separate, one-sided SOTA
non-inferiority gate.  That gate answers only whether the interval rules out
BASELINE being materially slower than RIVAL; it does not replace or rewrite
the exact five-state verdict.  Consequently an exact INCONCLUSIVE interval
that crosses only the WIN/EQUIVALENT boundary can still PASS non-inferiority.

Absolute per-variant spread is printed as a diagnostic, not as an alternative
verdict: machine noise is often correlated and cancels in the paired ratio.
Formal evidence also repeats the experiment with variant order swapped and,
for short aggregates, an equivalent code-shape/placement perturbation. This
script labels a single order as provisional; the review certificate combines
those external gates and decides whether the comparison contract is valid.

USAGE
-----
    scripts/bench-stat.py <exe> [-n N] [--pair BASELINE RIVAL] ...

    # container_intrusive: classify one execution order of HYLE vs Boost.
    scripts/bench-stat.py build-bench/bin/hyle_container_intrusive_bench -n 15 \
        --equivalence-margin-pct 3 \
        --pair "HYLE IntrusiveList" "boost::intrusive auto_unlink"

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
PAIR_RAW_PREFIX = "# HYLE_BENCH_PAIR_RAW "
CHILD_TELEMETRY_PREFIX = "# HYLE_BENCH_CHILD_TELEMETRY "


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


def _median_confidence_interval(ratios, tail_alpha=0.01):
    """Return a distribution-free interval for the population median.

    Select the narrowest order-statistic interval whose two binomial tails are
    each no larger than tail_alpha. With n=15 and alpha=.01 this is [x3,x13]
    (1-indexed), with actual coverage about 99.26% (at least 98%).
    """
    ordered = sorted(ratios)
    n = len(ordered)
    selected = None
    for k in range(1, (n + 1) // 2 + 1):
        lower_tail = sum(math.comb(n, i) for i in range(k)) / (2 ** n)
        if lower_tail <= tail_alpha:
            selected = k
        else:
            break
    if selected is None:
        return None
    return ordered[selected - 1], ordered[n - selected], selected


def paired_verdict(ratios, equivalence_margin_percent=None,
                   tail_alpha=0.01, minimum_runs=15, comparable=True):
    """Classify one execution order from BASELINE's perspective.

    Ratios are RIVAL/BASELINE op/s. Contract comparability, order swap,
    code-placement stability, and platform aggregation are external gates.
    """
    if not ratios:
        raise ValueError("paired verdict needs at least one ratio")
    if any(not math.isfinite(ratio) or ratio <= 0.0 for ratio in ratios):
        raise ValueError("paired ratios must be finite and positive")
    if not 0.0 < tail_alpha < 0.5:
        raise ValueError("tail alpha must be between zero and 0.5")
    if not isinstance(minimum_runs, int) or minimum_runs <= 0:
        raise ValueError("minimum runs must be a positive integer")
    if equivalence_margin_percent is not None and not (
            math.isfinite(equivalence_margin_percent) and
            equivalence_margin_percent > 0.0):
        raise ValueError("equivalence margin must be finite and positive")

    median = statistics.median(ratios)
    gap_percent = abs(median - 1.0) * 100.0
    above = sum(1 for ratio in ratios if ratio > 1.0)
    below = sum(1 for ratio in ratios if ratio < 1.0)
    effective = above + below
    p_value = (
        _binom_two_sided(max(above, below), effective)
        if effective > 0 else 1.0
    )
    interval = _median_confidence_interval(ratios, tail_alpha)
    verdict = "INCONCLUSIVE"
    band = None
    if equivalence_margin_percent is not None:
        margin = equivalence_margin_percent / 100.0
        band = (1.0 / (1.0 + margin), 1.0 + margin)

    if not comparable:
        verdict = "NOT_COMPARABLE"
        reason = "comparison-contract-not-equivalent"
    elif len(ratios) < minimum_runs:
        reason = "insufficient-samples"
    elif equivalence_margin_percent is None:
        reason = "margin-not-declared"
    elif interval is None:
        reason = "confidence-interval-unavailable"
    else:
        lower, upper, _ = interval
        if upper < band[0]:
            verdict = "WIN"
            reason = "baseline-faster-outside-margin"
        elif lower > band[1]:
            verdict = "LOSE"
            reason = "rival-faster-outside-margin"
        elif lower >= band[0] and upper <= band[1]:
            verdict = "EQUIVALENT"
            reason = "median-ci-inside-equivalence-band"
        else:
            reason = "median-ci-crosses-decision-boundary"
    return {
        "median": median,
        "gap_percent": gap_percent,
        "above": above,
        "effective": effective,
        "p_value": p_value,
        "verdict": verdict,
        "reason": reason,
        "interval": interval,
        "band": band,
        "tail_alpha": tail_alpha,
    }


def sota_noninferiority_gate(ratios, equivalence_margin_percent=None,
                             tail_alpha=0.01, minimum_runs=15,
                             comparable=True):
    """Evaluate the independent one-sided SOTA gate for one execution order.

    Ratios and the predeclared margin have exactly the same meaning as in
    paired_verdict().  If U=1+m, PASS requires the median CI upper bound to be
    at most U; FAIL requires the lower bound to be greater than U.  An interval
    that still reaches beyond U is UNRESOLVED.  This deliberately leaves the
    exact five-state verdict untouched.
    """
    decision = paired_verdict(
        ratios,
        equivalence_margin_percent,
        tail_alpha=tail_alpha,
        minimum_runs=minimum_runs,
        comparable=comparable,
    )
    interval = decision["interval"]
    band = decision["band"]
    limit = None if band is None else band[1]

    if decision["verdict"] == "NOT_COMPARABLE":
        gate = "N/A"
        reason = "comparison-contract-not-equivalent"
    elif len(ratios) < minimum_runs:
        gate = "UNRESOLVED"
        reason = "insufficient-samples"
    elif limit is None:
        gate = "UNRESOLVED"
        reason = "sota-limit-not-declared"
    elif interval is None:
        gate = "UNRESOLVED"
        reason = "confidence-interval-unavailable"
    else:
        lower, upper, _ = interval
        if upper <= limit:
            gate = "PASS"
            reason = "median-ci-upper-within-sota-limit"
        elif lower > limit:
            gate = "FAIL"
            reason = "median-ci-lower-beyond-sota-limit"
        else:
            gate = "UNRESOLVED"
            reason = "median-ci-crosses-sota-limit"
    return {
        "gate": gate,
        "reason": reason,
        "interval": interval,
        "limit": limit,
        "tail_alpha": tail_alpha,
    }


def combine_order_verdicts(first, second, comparable=True, shape_stable=True):
    """Combine two independently sampled orders after external review gates."""
    valid = {"WIN", "EQUIVALENT", "LOSE", "INCONCLUSIVE",
             "NOT_COMPARABLE"}
    if first not in valid or second not in valid:
        raise ValueError("unknown execution-order verdict")
    if (not comparable or first == "NOT_COMPARABLE" or
            second == "NOT_COMPARABLE"):
        return "NOT_COMPARABLE"
    if not shape_stable:
        return "INCONCLUSIVE"
    stable = {"WIN", "EQUIVALENT", "LOSE"}
    if first in stable and first == second:
        return first
    return "INCONCLUSIVE"


def combine_sota_gates(default_order, peer_first, shape=None,
                       comparable=True):
    """Combine independently sampled SOTA gates.

    PASS requires every applicable gate to PASS.  Any valid FAIL blocks the
    claim; otherwise an incomplete or unstable gate remains UNRESOLVED.  Shape
    is optional only for workloads whose predeclared contract says the extra
    placement gate is not applicable.
    """
    valid = {"PASS", "FAIL", "UNRESOLVED", "N/A"}
    gates = [default_order, peer_first]
    if shape is not None:
        gates.append(shape)
    if any(gate not in valid for gate in gates):
        raise ValueError("unknown SOTA non-inferiority gate")
    if not comparable or any(gate == "N/A" for gate in gates):
        return "N/A"
    if any(gate == "FAIL" for gate in gates):
        return "FAIL"
    if all(gate == "PASS" for gate in gates):
        return "PASS"
    return "UNRESOLVED"


def pair_process_records(runs, title, baseline, rival, comparable=True):
    """Return one machine-readable paired measurement record per process."""
    baseline_key = (title, baseline)
    rival_key = (title, rival)
    records = []
    for run_index, run in enumerate(runs, start=1):
        if baseline_key not in run or rival_key not in run:
            continue
        baseline_ops = run[baseline_key]
        rival_ops = run[rival_key]
        if (not math.isfinite(baseline_ops) or baseline_ops <= 0.0 or
                not math.isfinite(rival_ops) or rival_ops <= 0.0):
            raise ValueError("paired op/s samples must be finite and positive")
        records.append({
            "record": "hyle_bench_pair_process",
            "schema": 1,
            "run": run_index,
            "title": title,
            "baseline": baseline,
            "rival": rival,
            "baseline_ops_per_second": baseline_ops,
            "rival_ops_per_second": rival_ops,
            "ratio_rival_over_baseline": rival_ops / baseline_ops,
            "comparable": comparable,
        })
    return records


def format_pair_process_record(record):
    """Encode one paired process record as a grep-friendly JSON payload."""
    return PAIR_RAW_PREFIX + json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False)


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


def extract_required_child_telemetry(text, prefix):
    """Return the unique child line for a caller-declared telemetry prefix."""
    matches = [line for line in text.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(
            f"required child telemetry {prefix!r} occurred "
            f"{len(matches)} times")
    return matches[0]


def format_child_telemetry_record(run, prefix, line):
    return CHILD_TELEMETRY_PREFIX + json.dumps({
        "record": "hyle_bench_child_telemetry",
        "schema": 1,
        "run": run,
        "prefix": prefix,
        "line": line,
    }, sort_keys=True, separators=(",", ":"))


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
                    default=[], help="classify candidate BASELINE from the "
                                     "RIVAL/BASELINE op/s ratio; repeatable")
    ap.add_argument("--not-comparable-pair", nargs=2, action="append",
                    metavar=("BASELINE", "RIVAL"), default=[],
                    help="report an informational RIVAL/BASELINE ratio without "
                         "a performance claim; repeatable")
    ap.add_argument("--equivalence-margin-pct", type=float,
                    help="predeclared practical equivalence margin in percent; "
                         "required with --pair for formal --require-clean evidence")
    ap.add_argument(
        "--require-telemetry-prefix", action="append", default=[],
        metavar="PREFIX",
        help="require exactly one matching child line per process and emit "
             "a structured run-tagged record; repeatable")
    args = ap.parse_args()

    if args.runs <= 0:
        ap.error("--runs must be positive")
    if args.require_clean and args.runs < 15:
        ap.error("--require-clean formal evidence requires at least 15 runs")
    if args.require_clean and args.pair and args.equivalence_margin_pct is None:
        ap.error("formal paired evidence requires --equivalence-margin-pct")
    if (args.equivalence_margin_pct is not None and
            (not math.isfinite(args.equivalence_margin_pct) or
             args.equivalence_margin_pct <= 0.0)):
        ap.error("--equivalence-margin-pct must be finite and positive")
    duplicate_pairs = set(map(tuple, args.pair)) & set(
        map(tuple, args.not_comparable_pair))
    if duplicate_pairs:
        ap.error("the same pair cannot be both comparable and not comparable")
    if len(set(args.require_telemetry_prefix)) != len(
            args.require_telemetry_prefix):
        ap.error("required telemetry prefixes must be unique")
    if any(not prefix or "\n" in prefix or "\r" in prefix
           for prefix in args.require_telemetry_prefix):
        ap.error("required telemetry prefixes must be non-empty single lines")

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
    child_telemetry = []
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
        for prefix in args.require_telemetry_prefix:
            try:
                line = extract_required_child_telemetry(output, prefix)
            except ValueError as error:
                sys.exit(f"run {run_index}: {error}")
            child_telemetry.append(
                format_child_telemetry_record(run_index, prefix, line))

    try:
        keys = require_consistent_rows(runs)
    except ValueError as error:
        sys.exit(str(error))

    for record in child_telemetry:
        print(record)

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

    requested_pairs = [
        (base, rival, True) for base, rival in args.pair
    ] + [
        (base, rival, False) for base, rival in args.not_comparable_pair
    ]
    if requested_pairs:
        print("\n=== pair ratios (RIVAL/BASELINE op/s) ===")
        print("    subject=BASELINE(candidate); single-order classification is provisional")
        print("    equivalence uses a distribution-free median interval with tail alpha=0.01")
        print("    SOTA gate independently tests non-inferiority at the same predeclared upper limit")
        print("    formal evidence must also agree after order swap and applicable code-shape gates")
    for base, rival, comparable in requested_pairs:
        any_hit = False
        for t in titles:
            records = pair_process_records(
                runs, t, base, rival, comparable=comparable)
            if not records:
                continue
            any_hit = True
            ratios = [record["ratio_rival_over_baseline"]
                      for record in records]
            decision = paired_verdict(
                ratios, args.equivalence_margin_pct, comparable=comparable)
            sota = sota_noninferiority_gate(
                ratios, args.equivalence_margin_pct, comparable=comparable)
            bk, rk = (t, base), (t, rival)
            rmed = decision["median"]
            combined = agg[bk][1] + agg[rk][1]
            gap = decision["gap_percent"]
            # Absolute spread is a stability diagnostic, not a product verdict.
            absolute_hint = (
                "inside" if gap <= combined else
                ("outside+" if rmed > 1 else "outside-"))
            interval = decision["interval"]
            ci_text = ("unavailable" if interval is None else
                       f"[{interval[0]:.3f},{interval[1]:.3f}]")
            band = decision["band"]
            band_text = ("undeclared" if band is None else
                         f"[{band[0]:.3f},{band[1]:.3f}]")
            print(f"  {t[:34]:36s} RIVAL={rival} BASELINE={base}")
            for record in records:
                print(format_pair_process_record(record))
            print(f"    ratio med {rmed:5.3f}  min {min(ratios):5.3f} max {max(ratios):5.3f}"
                  f"  gap {gap:4.1f}% vs spread {combined:4.1f}%"
                  f"  [absolute]{absolute_hint}")
            print(f"    median_ci98+ {ci_text} equivalence_band {band_text}"
                  f"  [order_verdict]{decision['verdict']}"
                  f" reason={decision['reason']}"
                  f" sign_diag={decision['above']}/{decision['effective']}>1,"
                  f"p={decision['p_value']:.1e}")
            limit_text = ("undeclared" if sota["limit"] is None else
                          f"{sota['limit']:.3f}")
            print(f"    sota_noninferiority_limit {limit_text}"
                  f"  [order_sota_gate]{sota['gate']}"
                  f" reason={sota['reason']}")
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
