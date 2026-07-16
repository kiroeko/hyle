#!/usr/bin/env python3
"""Aggregate op_parity_* output and separate real candidates from measurement ambiguity.

The benchmark emits four code replicas per side. HYLE_BENCH_HWYFIRST also selects an alternate
RepTag/layout bank, so a certification run is two repetitions of each order: four independent run
clusters, each containing four correlated code-placement samples per op. This report:

  * refuses verdicts when source-verified 1.00 anchors miss the calibration gate;
  * normalizes each file by its median anchor ratio (removes residual run-wide skew);
  * uses a deterministic hierarchical (run cluster, then replica) bootstrap 95% CI;
  * calls LOSE/WIN only when the CI and both order medians clear both the 0.97/1.03 practical band
    and the operation's observed code-placement uncertainty;
  * distinguishes INCONCLUSIVE from TIE instead of hiding a 0.93 point estimate behind spread.

Usage:
  op-parity-report.py [--gate 0.06] [--certify] out1.txt out2.txt ...

--certify requires metadata from the current harness and >=2 complete files per order. Without it,
old two-file reports remain readable but are explicitly marked PROVISIONAL.
"""

import argparse
import hashlib
import random
import re
import statistics
import sys


TITLE_RE = re.compile(r'\|\s*relative\s*\|.*\|\s*([^|]+?)\s*$')
ROW_RE = re.compile(
    r'\|\s*[\d.,]+%\s*\|\s*[\d.,]+\s*\|\s*([\d,]+\.\d+)\s*\|\s*[\d.,]+%\s*\|'
    r'\s*[\d.,]+\s*\|\s*(?::wavy_dash:\s*)?`([^`]+)`')
META_RE = re.compile(r'^# HYLE_OP_PARITY\s+order=(HYLE_FIRST|HWY_FIRST)\s+replicas=(\d+)\s+'
                     r'layout_bank=(\d+)(?:\s+pin_ok=(\d+)\s+priority_ok=(\d+)\s+'
                     r'cpu=(-?\d+)\s+core_class=(-?\d+))?')

# True-ratio == 1.00 by construction, verified at source/disassembly level. TableLookupBytes is not
# eligible: HYLE intentionally adds an index-masking instruction for its stronger default contract.
ANCHORS = {
    'Neg i32', 'Sub i32', 'Mul i32', 'Max i32', 'ShiftLeft<3> i32',
    'Add f32 (4096)', 'MulAdd f32 (4096)', 'Min i32 (4096)',
    'SaturatedAdd i16 (4096)',
}
GATE_DEFAULT = 0.04
PRACTICAL_LO = 0.97
PRACTICAL_HI = 1.03


def read_lines(path):
    """Read benchmark output from shells that may redirect as UTF-8 or UTF-16 (Windows PS 5.1)."""
    with open(path, 'rb') as src:
        raw = src.read()
    encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8'
    return raw.decode(encoding, errors='replace').splitlines()


def parse(path):
    """Return (records, metadata)."""
    res = {}
    meta = {'path': path, 'order': None, 'replicas': None, 'layout_bank': None,
            'pin_ok': None, 'priority_ok': None, 'cpu': None, 'core_class': None}
    cur = None
    for ln in read_lines(path):
        mm = META_RE.match(ln)
        if mm:
            meta.update(order=mm.group(1), replicas=int(mm.group(2)),
                        layout_bank=int(mm.group(3)),
                        pin_ok=int(mm.group(4)) if mm.group(4) is not None else None,
                        priority_ok=int(mm.group(5)) if mm.group(5) is not None else None,
                        cpu=int(mm.group(6)) if mm.group(6) is not None else None,
                        core_class=int(mm.group(7)) if mm.group(7) is not None else None)
            continue
        mt = TITLE_RE.match(ln)
        if mt:
            cur = mt.group(1).strip()
            res.setdefault(cur, {'H': [], 'G': [], 'Hi': {}, 'Gi': {},
                                 'A': [], 'B': [], 'Ai': {}, 'Bi': {}, 'third': {}})
            continue
        m = ROW_RE.search(ln)
        if not (m and cur):
            continue
        ops = float(m.group(1).replace(',', ''))
        label = m.group(2)
        if label.startswith('HYLE#'):
            res[cur]['H'].append(ops)
            res[cur]['Hi'][int(label[5:])] = ops
        elif label.startswith('HWY#'):
            res[cur]['G'].append(ops)
            res[cur]['Gi'][int(label[4:])] = ops
        elif label.startswith('INFO_A#'):
            res[cur]['A'].append(ops)
            res[cur]['Ai'][int(label[7:])] = ops
        elif label.startswith('INFO_B#'):
            res[cur]['B'].append(ops)
            res[cur]['Bi'][int(label[7:])] = ops
        else:
            res[cur]['third'][label] = ops
    return res, meta


def paired_samples(rec):
    common = sorted(set(rec['Hi']) & set(rec['Gi']))
    return [rec['Hi'][k] / rec['Gi'][k] for k in common]


def aggregate(rec):
    h, g = rec['H'], rec['G']
    if not h or not g:
        return None
    paired = paired_samples(rec)
    return {
        'best': max(h) / max(g),
        'med': statistics.median(h) / statistics.median(g),
        'paired': statistics.median(paired) if paired else max(h) / max(g),
        'h_spread': max(h) / min(h),
        'g_spread': max(g) / min(g),
    }


def verdict_one(a):
    """Legacy/provisional single-file verdict, kept for old files without v3 metadata."""
    uncertainty = max(0.03, (max(a['h_spread'], a['g_spread']) - 1) / 2)
    if all(a[k] < 1 - uncertainty for k in ('best', 'med', 'paired')):
        return 'LOSE'
    if all(a[k] > 1 + uncertainty for k in ('best', 'med', 'paired')):
        return 'WIN'
    return 'TIE'


def percentile(sorted_values, p):
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = p * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_cluster_median_ci(clusters, key, rounds=5000):
    """Hierarchical bootstrap: complete runs are independent; replicas within one binary are not."""
    clusters = [list(cluster) for cluster in clusters if cluster]
    if not clusters:
        return float('nan'), float('nan')
    if len(clusters) == 1 and len(clusters[0]) == 1:
        return clusters[0][0], clusters[0][0]
    seed = int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'little')
    rng = random.Random(seed)
    n_clusters = len(clusters)
    estimates = []
    for _ in range(rounds):
        run_medians = []
        for _ in range(n_clusters):
            cluster = clusters[rng.randrange(n_clusters)]
            # Preserve the two-level sampling structure instead of pretending that all replicas
            # from one fixed binary are independent observations.
            run_medians.append(statistics.median(
                cluster[rng.randrange(len(cluster))] for _ in range(len(cluster))))
        estimates.append(statistics.median(run_medians))
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gate', type=float, default=GATE_DEFAULT)
    ap.add_argument('--certify', action='store_true',
                    help='require >=2 current-harness files for each execution/layout order')
    ap.add_argument('files', nargs='+')
    args = ap.parse_args()

    parsed = [parse(path) for path in args.files]
    runs = [item[0] for item in parsed]
    metas = [item[1] for item in parsed]
    ops = sorted(set().union(*(set(r) for r in runs)))
    gate_lo, gate_hi = 1 - args.gate, 1 + args.gate

    # Per-file normalization factor: median of several known-1.00 anchors is a robust estimate of
    # residual run-wide H/G skew. The raw paired estimate must still pass the gate before
    # normalization. Do not gate on best-of-replicas: v4 treats fixed code placement as a nuisance
    # parameter, and `best` is itself placement-sensitive (a source-identical Mac anchor reached
    # 1.075 while paired=1.028 and the single-side replica spread was 1.267). spread/best remain in
    # the diagnostic line, and placement uncertainty still participates in every op verdict.
    cal_factors = []
    for run in runs:
        vals = [aggregate(run[op])['paired'] for op in ANCHORS
                if op in run and aggregate(run[op])]
        cal_factors.append(statistics.median(vals) if vals else 1.0)

    cal_fail = []
    n_anchor = 0
    print('== calibration anchors (true ratio 1.00 by construction) ==')
    for op in ops:
        if op not in ANCHORS:
            continue
        aggs = [aggregate(run[op]) for run in runs if op in run]
        aggs = [a for a in aggs if a]
        if not aggs:
            continue
        n_anchor += 1
        best = statistics.median(a['best'] for a in aggs)
        med = statistics.median(a['med'] for a in aggs)
        paired = statistics.median(a['paired'] for a in aggs)
        spread = max(max(a['h_spread'], a['g_spread']) for a in aggs)
        ok = gate_lo <= paired <= gate_hi
        if not ok:
            cal_fail.append((op, best, paired))
        print(f'  {op:24s} paired={paired:.3f} best={best:.3f} med={med:.3f} '
              f'maxspread={spread:.3f} ({len(aggs)} runs) '
              f'{"ok" if ok else "OUT-OF-GATE"}')

    calibration_ok = n_anchor > 0 and not cal_fail
    if cal_fail:
        print(f'!! CALIBRATION FAIL ({len(cal_fail)} anchors outside '
              f'[{gate_lo:.2f},{gate_hi:.2f}]); all verdicts suppressed')
    elif n_anchor:
        print('CALIBRATION PASS')
    else:
        print('!! NO CALIBRATION ANCHORS; all verdicts suppressed')

    order_counts = {name: sum(m['order'] == name for m in metas)
                    for name in ('HYLE_FIRST', 'HWY_FIRST')}
    metadata_complete = all(m['order'] is not None and m['replicas'] == 4 for m in metas)
    layout_ok = all((m['order'] == 'HYLE_FIRST' and m['layout_bank'] == 0) or
                    (m['order'] == 'HWY_FIRST' and m['layout_bank'] == 1) for m in metas)
    distinct_paths = len({str(m['path']) for m in metas}) == len(metas)
    input_digests = []
    for meta in metas:
        with open(meta['path'], 'rb') as src:
            input_digests.append(hashlib.sha256(src.read()).digest())
    distinct_inputs = distinct_paths and len(set(input_digests)) == len(input_digests)
    certification_ok = (metadata_complete and order_counts['HYLE_FIRST'] >= 2 and
                        order_counts['HWY_FIRST'] >= 2 and layout_ok and distinct_inputs)
    print(f'run set: HYLE_FIRST={order_counts["HYLE_FIRST"]}, '
          f'HWY_FIRST={order_counts["HWY_FIRST"]}, metadata={"ok" if metadata_complete else "legacy"}, '
          f'layout={"ok" if layout_ok else "MISMATCH"}, '
          f'inputs={"distinct" if distinct_inputs else "DUPLICATED"}')
    audit_complete = all(m['pin_ok'] is not None for m in metas)
    if audit_complete:
        pinned = [m for m in metas if m['pin_ok']]
        cpu_consistent = not pinned or len({m['cpu'] for m in pinned}) == 1
        class_values = {m['core_class'] for m in pinned if m['core_class'] != -1}
        class_consistent = len(class_values) <= 1
        priority_ok = all(m['priority_ok'] for m in metas)
        pin_supported_ok = all(m['pin_ok'] or m['cpu'] == -1 for m in metas)
        execution_ok = priority_ok and pin_supported_ok and cpu_consistent and class_consistent
        certification_ok = certification_ok and execution_ok
        cpu_text = ','.join(str(value) for value in sorted({m['cpu'] for m in pinned})) or 'n/a'
        class_text = ','.join(str(value) for value in sorted(class_values)) or 'unknown'
        pin_text = 'n/a' if all(m['cpu'] == -1 for m in metas) else ('ok' if pin_supported_ok else 'FAIL')
        print(f'execution audit: priority={"ok" if priority_ok else "FAIL"}, '
              f'pin={pin_text}, '
              f'pinned_cpu={"consistent" if cpu_consistent else "MIXED"}({cpu_text}), '
              f'core_class={"consistent" if class_consistent else "MIXED"}({class_text})')
    else:
        print('execution audit: legacy metadata (pin/core class not recorded)')
        certification_ok = False
    if args.certify:
        print('CERTIFICATION PASS' if certification_ok else
              '!! CERTIFICATION INSUFFICIENT (need >=2 current-harness files per order and a '
              'consistent execution audit)')
    else:
        print('PROVISIONAL REPORT (pass --certify with >=2 files/order for authoritative verdicts)')
    print()

    suppress = not calibration_ok or (args.certify and not certification_ok)
    print('op\tratio_paired\tci95_lo\tci95_hi\tratio_med\tspreadH\tspreadG\tplace_u\tverdict')
    rows = []
    for op in ops:
        indexed = [(i, aggregate(run[op])) for i, run in enumerate(runs) if op in run]
        indexed = [(i, a) for i, a in indexed if a]
        if not indexed:
            continue

        clusters = []
        by_order = {'HYLE_FIRST': [], 'HWY_FIRST': []}
        for i, _ in indexed:
            normalized = [v / cal_factors[i] for v in paired_samples(runs[i][op])]
            if normalized:
                clusters.append(normalized)
            order = metas[i]['order']
            if order in by_order:
                by_order[order].extend(normalized)

        samples = [value for cluster in clusters for value in cluster]
        ratio = statistics.median(statistics.median(cluster) for cluster in clusters) if clusters else statistics.median(
            a['paired'] / cal_factors[i] for i, a in indexed)
        ci_lo, ci_hi = bootstrap_cluster_median_ci(clusters, op)
        med = statistics.median(a['med'] / cal_factors[i] for i, a in indexed)
        hs = max(a['h_spread'] for _, a in indexed)
        gs = max(a['g_spread'] for _, a in indexed)
        # Front-end/code-placement effects are a nuisance parameter, not evidence that either
        # implementation is slower. Half the worst single-side replica range is a conservative
        # estimate of that uncertainty. Never narrow the practical 3% equivalence band.
        placement_uncertainty = max(1.0 - PRACTICAL_LO, PRACTICAL_HI - 1.0,
                                    (max(hs, gs) - 1.0) / 2.0)
        decision_lo = 1.0 - placement_uncertainty
        decision_hi = 1.0 + placement_uncertainty

        if suppress:
            verdict = 'UNCALIBRATED' if not calibration_ok else 'INSUFFICIENT'
        elif metadata_complete and all(by_order.values()):
            order_medians = [statistics.median(by_order[name]) for name in by_order]
            if ci_hi < decision_lo and all(v < decision_lo for v in order_medians):
                verdict = 'LOSE'
            elif ci_lo > decision_hi and all(v > decision_hi for v in order_medians):
                verdict = 'WIN'
            # Equivalence is an affirmative claim: the entire CI must fit inside the practical
            # band. Merely touching the band means the measurement cannot distinguish tie/loss.
            elif (ci_lo >= PRACTICAL_LO and ci_hi <= PRACTICAL_HI and
                  placement_uncertainty <= (PRACTICAL_HI - 1.0) + 1e-12):
                verdict = 'TIE'
            else:
                verdict = 'INCONCLUSIVE'
        else:
            legacy = {verdict_one(a) for _, a in indexed}
            verdict = legacy.pop() if len(legacy) == 1 else 'UNSTABLE'

        rows.append((verdict, ratio, ci_lo, ci_hi, med, hs, gs, placement_uncertainty, op))

    order = {'LOSE': 0, 'INCONCLUSIVE': 1, 'UNSTABLE': 2, 'WIN': 3, 'TIE': 4,
             'INSUFFICIENT': 5, 'UNCALIBRATED': 6}
    rows.sort(key=lambda row: (order.get(row[0], 99), row[1]))
    for verdict, ratio, ci_lo, ci_hi, med, hs, gs, placement_u, op in rows:
        print(f'{op}\t{ratio:.3f}\t{ci_lo:.3f}\t{ci_hi:.3f}\t{med:.3f}\t'
              f'{hs:.3f}\t{gs:.3f}\t{placement_u:.3f}\t{verdict}')
    for verdict in order:
        count = sum(row[0] == verdict for row in rows)
        if count:
            print(f'# {verdict}: {count}/{len(rows)}', file=sys.stderr)

    info_rows = []
    for op in ops:
        clusters = []
        by_order = {'HYLE_FIRST': [], 'HWY_FIRST': []}
        for i, run in enumerate(runs):
            rec = run.get(op)
            if not rec:
                continue
            common = sorted(set(rec['Ai']) & set(rec['Bi']))
            paired = [(rec['Ai'][k] / rec['Bi'][k]) / cal_factors[i] for k in common]
            if paired:
                clusters.append(paired)
            if metas[i]['order'] in by_order:
                by_order[metas[i]['order']].extend(paired)
        if not clusters:
            continue
        samples = [value for cluster in clusters for value in cluster]
        ratio = statistics.median(statistics.median(cluster) for cluster in clusters)
        ci_lo, ci_hi = bootstrap_cluster_median_ci(clusters, 'INFO:' + op)
        order_text = '/'.join(
            f'{statistics.median(by_order[name]):.3f}' if by_order[name] else '-'
            for name in ('HYLE_FIRST', 'HWY_FIRST'))
        info_rows.append((op, ratio, ci_lo, ci_hi, order_text, len(samples)))
    if info_rows:
        print('\n# informational replicated pairs (A/B; no Highway verdict)')
        print('# op\tratio_A_over_B\tci95_lo\tci95_hi\torders_H/G\tsamples')
        for op, ratio, ci_lo, ci_hi, order_text, n in info_rows:
            state = 'UNCALIBRATED' if suppress else 'INFO'
            print(f'# {op}\t{ratio:.3f}\t{ci_lo:.3f}\t{ci_hi:.3f}\t'
                  f'{order_text}\t{n}\t{state}')

    thirds = []
    for op in ops:
        for run in runs:
            if op in run and run[op]['third'] and run[op]['H']:
                for label, value in run[op]['third'].items():
                    thirds.append((op, label, value / max(run[op]['H'])))
    if thirds:
        print('\n# third / HYLE_best')
        seen = set()
        for op, label, ratio in thirds:
            if (op, label) in seen:
                continue
            seen.add((op, label))
            print(f'# {op}\t{label}\t{ratio:.3f}')

    if suppress:
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
