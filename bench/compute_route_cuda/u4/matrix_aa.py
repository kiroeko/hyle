"""One bounded same-workload AA qualification batch; product statistics stay canonical."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from hyle_perf import sampling_mutex
from hyle_perf.machine import load_registry, get_machine_profile, probe_machine, evaluate_machine_profile, expected_child_policy
from hyle_perf.runner import _apply_sampling_policy, _required_telemetry
from hyle_perf.environment import PERF_ENVIRONMENT_BLOCKLIST, PERF_ENVIRONMENT_FIXED, PERF_ENVIRONMENT_PREFIXES
from hyle_perf.nanobench import parse_nanobench, select_pair


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ctypes.windll.kernel32.SetErrorMode(32771)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError('Fixed AA batch output already exists; no automatic rerun')
    output.mkdir(parents=True)
    exe = args.build.resolve() / 'hyle_cuda_matrix_bench.exe'
    pairs = json.loads(Path(__file__).with_name('matrix-pairs.json').read_text())['pairs']
    profile = get_machine_profile(load_registry(), 'W1-WIN')
    probe = probe_machine()
    gate = evaluate_machine_profile(profile, probe, role='engineering', sampling_topology='native-runtime-pair')
    if gate['gate'] != 'PASS':
        raise RuntimeError(str(gate))
    policy = expected_child_policy(profile, sampling_topology='native-runtime-pair', probe=probe)
    runs = []
    with sampling_mutex(profile), _apply_sampling_policy(policy):
        for route in ('selected', 'peer'):
            for reverse in (False, True):
                for allocation in (False, True):
                    env = {k: v for k, v in os.environ.items()
                           if k not in PERF_ENVIRONMENT_BLOCKLIST and not any(k.startswith(p) for p in PERF_ENVIRONMENT_PREFIXES)}
                    env.update(PERF_ENVIRONMENT_FIXED)
                    env.update(HYLE_BENCH_AA_ROUTE=route, HYLE_PERF_CALLER_CPU=str(policy['caller_logical_cpu']),
                               HYLE_PERF_SAMPLING_TOPOLOGY='native-runtime-pair', HYLE_PERF_RUN_NONCE=uuid.uuid4().hex)
                    if reverse: env['HYLE_BENCH_RIVAL_FIRST'] = '1'
                    if allocation: env['HYLE_BENCH_PEER_ALLOCATION_FIRST'] = '1'
                    log = output / f'{route}-{int(reverse)}-{int(allocation)}.log'
                    result = subprocess.run([str(exe)], env=env, capture_output=True, text=True,
                        encoding='utf-8', errors='replace', timeout=600,
                        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.ABOVE_NORMAL_PRIORITY_CLASS)
                    # Keep stdout/stderr separate during capture so vendor logs cannot split metric rows.
                    text = result.stdout + result.stderr
                    log.write_text(text, encoding='utf-8')
                    if result.returncode:
                        raise RuntimeError('AA child failed: ' + str(log))
                    _required_telemetry(text, ['# HYLE_CUDA_MATRIX_RUN '], {
                        'execution_order': 'peer-first' if reverse else 'candidate-first',
                        'allocation_order': 'peer-first' if allocation else 'candidate-first',
                        'aa_route': route, 'paired_mode': 'nanobench-compare',
                        'affinity_topology': 'native-runtime-pair', 'caller_cpu': str(policy['caller_logical_cpu']),
                        'affinity_ok': '1', 'parent_mask': policy['affinity_mask'], 'priority': policy['priority'],
                        'run_nonce': env['HYLE_PERF_RUN_NONCE'], 'verify_only': '0'})
                    metrics = parse_nanobench(text)
                    ratios = {}
                    for pair in pairs:
                        a, b = select_pair(metrics, 'Candidate', 'Peer', title=pair['id'])
                        ratios[pair['id']] = a.operations_per_second / b.operations_per_second
                    runs.append({'route': route, 'reverse': reverse, 'allocation': allocation,
                                 'ratios': ratios, 'log': log.as_posix(), 'sha256': digest(log)})
                    print(json.dumps({'completed': len(runs), 'total': 8, 'pairs': len(ratios)}), flush=True)
    decisions = []
    for pair in pairs:
        errors, summaries = [], {}
        for route in ('selected', 'peer'):
            group = [r for r in runs if r['route'] == route]
            values = [r['ratios'][pair['id']] for r in group]
            medians = {'all': statistics.median(values)}
            for factor in ('reverse', 'allocation'):
                for level in (False, True):
                    medians[f'{factor}={level}'] = statistics.median(r['ratios'][pair['id']] for r in group if r[factor] == level)
            if any(not 1 / 1.10 <= v <= 1.10 for v in values): errors.append(route + ': individual instability over10%')
            if any(not 1 / 1.05 <= v <= 1.05 for v in medians.values()): errors.append(route + ': factor median bias over5%')
            summaries[route] = {'ratios': values, 'factor_medians': medians}
        decisions.append({'id': pair['id'], 'qualification': 'DIAGNOSTIC' if errors else 'QUALIFIED',
                          'errors': errors, 'summaries': summaries})
    record = {'schema': 'hyle.cuda-matrix-aa/v1', 'binary_sha256': digest(exe),
              'pairs_sha256': digest(Path(__file__).with_name('matrix-pairs.tsv')),
              'policy': policy, 'runs': runs, 'decisions': decisions,
              'qualified': sum(not d['errors'] for d in decisions), 'total': len(decisions)}
    (output / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'qualified': record['qualified'], 'total': record['total']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
