#!/usr/bin/env python3
"""Aggregate op_parity_* bench output into trustworthy per-op ratios.

The benches (via bench/simd/op_parity_engine.h) emit 4 code replicas per side per op, labeled
HYLE#0..3 / HWY#0..3. A single replica's time carries a code-placement bias of +-10..15% (tail
~30%) that is deterministic per binary and NOT cancelled by order swapping -- see
docs/debug/op-parity-harness-artifacts-hunt.md. This script therefore:

  * aggregates each side over its replicas: best (max op/s ~= placement-ideal) and median;
  * ratio = HYLE/Highway for both estimators (>1 = HYLE faster);
  * gates everything on calibration anchors -- ops whose HYLE and Highway implementations are the
    same single instruction (verified at source/disasm level), whose true ratio is 1.00 by
    construction. If any anchor's best-ratio leaves [0.96, 1.04] the run is declared UNCALIBRATED
    and verdicts must not be read.
  * verdict per op: LOSE if best-ratio < 0.97 AND median-ratio < 0.97 (both estimators agree),
    WIN if both > 1.03, else TIE. With multiple input files (e.g. both orders via
    HYLE_BENCH_HWYFIRST, or repeat runs) the verdict must hold in every file, else UNSTABLE.

Usage: op-parity-report.py out1.txt [out2.txt ...]
Output: TSV to stdout (op, ratio_best, ratio_med, spreads, verdict), calibration banner first.
"""
import re
import statistics
import sys

TITLE_RE = re.compile(r'\|\s*relative\s*\|.*\|\s*([^|]+?)\s*$')
# tolerate nanobench's :wavy_dash: unstable-row marker before the label (dropping those rows
# silently halves replica counts on noisy hosts -- it bit the first Mac analysis)
ROW_RE = re.compile(
    r'\|\s*[\d.,]+%\s*\|\s*[\d.,]+\s*\|\s*([\d,]+\.\d+)\s*\|\s*[\d.,]+%\s*\|\s*[\d.,]+\s*\|\s*(?::wavy_dash:\s*)?`([^`]+)`')

# True-ratio == 1.00 by construction: source-verified same intrinsic both sides (Neg additionally
# verified instruction-identical in disassembly). NOT eligible: TableLookupBytes (HYLE adds an
# index-masking vpand by design -- a real, intended semantic difference).
ANCHORS = {
    'Neg i32', 'Sub i32', 'Mul i32', 'Max i32', 'ShiftLeft<3> i32',            # op_parity_full
    'Add f32 (4096)', 'MulAdd f32 (4096)', 'Min i32 (4096)',                   # op_parity
    'SaturatedAdd i16 (4096)',
}
# default +-4%; noisy hosts without CPU affinity (macOS) pass --gate 0.06 and the column is
# documented at that resolution.
GATE_DEFAULT = 0.04


def parse(path):
    """-> {title: {'H': [op/s...], 'G': [...], 'third': {label: op/s}}}"""
    res = {}
    cur = None
    for ln in open(path, encoding='utf-8', errors='replace'):
        mt = TITLE_RE.match(ln)
        if mt:
            cur = mt.group(1).strip()
            res.setdefault(cur, {'H': [], 'G': [], 'Hi': {}, 'Gi': {}, 'third': {}})
            continue
        m = ROW_RE.search(ln)
        if m and cur:
            ops = float(m.group(1).replace(',', ''))
            label = m.group(2)
            if label.startswith('HYLE#'):
                res[cur]['H'].append(ops)
                res[cur]['Hi'][int(label[5:])] = ops
            elif label.startswith('HWY#'):
                res[cur]['G'].append(ops)
                res[cur]['Gi'][int(label[4:])] = ops
            else:
                res[cur]['third'][label] = ops
    return res


def aggregate(rec):
    h, g = rec['H'], rec['G']
    if not h or not g:
        return None
    # paired: same-round H/G ran adjacent in time (engine interleaves H0 G0 H1 G1 ...), so their
    # ratio cancels slow frequency/scheduler drift -- the primary estimator on hosts without CPU
    # affinity (macOS). On pinned hosts it agrees with best/med.
    common = sorted(set(rec['Hi']) & set(rec['Gi']))
    paired = statistics.median(rec['Hi'][k] / rec['Gi'][k] for k in common) if common else None
    return {
        'best': max(h) / max(g),
        'med': statistics.median(h) / statistics.median(g),
        'paired': paired if paired is not None else max(h) / max(g),
        'h_spread': max(h) / min(h),
        'g_spread': max(g) / min(g),
    }


def verdict_one(a):
    # spread-aware: the within-side replica spread is a direct measurement of this op's placement
    # sensitivity in this binary; a verdict needs the gap to clear half of it (best-of-replicas
    # already absorbs most of the wobble, so half-spread is the conservative residual).
    u = max(0.03, (max(a['h_spread'], a['g_spread']) - 1) / 2)
    if a['best'] < 1 - u and a['med'] < 1 - u and a['paired'] < 1 - u:
        return 'LOSE'
    if a['best'] > 1 + u and a['med'] > 1 + u and a['paired'] > 1 + u:
        return 'WIN'
    return 'TIE'


def main():
    args = sys.argv[1:]
    gate = GATE_DEFAULT
    if args and args[0] == '--gate':
        gate = float(args[1])
        args = args[2:]
    files = args
    if not files:
        sys.exit(__doc__)
    gate_lo, gate_hi = 1 - gate, 1 + gate
    runs = [parse(p) for p in files]
    ops = sorted(set().union(*[set(r) for r in runs]))

    # calibration gate: judged on the MERGED estimate (median across input files) per anchor --
    # single-file readings wobble a few %, which is exactly why >=2 runs (both orders) are fed in.
    cal_fail = []
    n_anchor = 0
    print('== calibration anchors (true ratio 1.00 by construction) ==')
    for op in ops:
        if op not in ANCHORS:
            continue
        aggs = [aggregate(r[op]) for r in runs if op in r]
        aggs = [a for a in aggs if a]
        if not aggs:
            continue
        n_anchor += 1
        best = statistics.median(a['best'] for a in aggs)
        med = statistics.median(a['med'] for a in aggs)
        paired = statistics.median(a['paired'] for a in aggs)
        sp = max(max(a['h_spread'], a['g_spread']) for a in aggs)
        ok = gate_lo <= best <= gate_hi
        if not ok:
            cal_fail.append((op, best))
        print(f'  {op:24s} paired={paired:.3f} best={best:.3f} med={med:.3f} maxspread={sp:.3f} '
              f'({len(aggs)} runs) {"ok" if ok else "OUT-OF-GATE"}')
    if cal_fail:
        print(f'!! CALIBRATION FAIL ({len(cal_fail)} anchors outside '
              f'[{gate_lo:.2f},{gate_hi:.2f}]) -- verdicts below are NOT trustworthy')
    elif n_anchor:
        print('CALIBRATION PASS')
    else:
        print('(no anchors in these files -- gate not applicable; '
              'trust only if the sibling benches passed on this machine+build)')
    print()

    print('op\tratio_best\tratio_med\tspreadH\tspreadG\tverdict')
    rows = []
    for op in ops:
        aggs = [aggregate(r[op]) for r in runs if op in r]
        aggs = [a for a in aggs if a]
        if not aggs:
            continue
        vs = {verdict_one(a) for a in aggs}
        v = vs.pop() if len(vs) == 1 else 'UNSTABLE'
        best = statistics.median(a['paired'] for a in aggs)
        med = statistics.median(a['med'] for a in aggs)
        hs = max(a['h_spread'] for a in aggs)
        gs = max(a['g_spread'] for a in aggs)
        rows.append((v, best, med, hs, gs, op))
    order = {'LOSE': 0, 'UNSTABLE': 1, 'WIN': 2, 'TIE': 3}
    rows.sort(key=lambda r: (order[r[0]], r[1]))
    for v, best, med, hs, gs, op in rows:
        print(f'{op}\t{best:.3f}\t{med:.3f}\t{hs:.3f}\t{gs:.3f}\t{v}')
    n = len(rows)
    for k in ('LOSE', 'UNSTABLE', 'WIN', 'TIE'):
        c = sum(1 for r in rows if r[0] == k)
        print(f'# {k}: {c}/{n}', file=sys.stderr)

    # informational third rows (xsimd / store+reload baseline): vs HYLE best
    thirds = []
    for op in ops:
        for r in runs:
            if op in r and r[op]['third'] and r[op]['H']:
                for label, v in r[op]['third'].items():
                    thirds.append((op, label, v / max(r[op]['H'])))
    if thirds:
        print('\n# third / HYLE_best)')
        seen = set()
        for op, label, ratio in thirds:
            if (op, label) in seen:
                continue
            seen.add((op, label))
            print(f'# {op}\t{label}\t{ratio:.3f}')


if __name__ == '__main__':
    main()
