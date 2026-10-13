"""Finite CPU ragged A/A; use the existing machine, environment and statistics policy."""
import argparse
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
from hyle_perf.environment import PERF_ENVIRONMENT_BLOCKLIST, PERF_ENVIRONMENT_FIXED, PERF_ENVIRONMENT_PREFIXES
from hyle_perf.machine import evaluate_machine_profile, expected_child_policy, get_machine_profile, load_registry, probe_machine
from hyle_perf.runner import _apply_sampling_policy, _creation_flags
from hyle_perf.controls import AA_MAXIMUM_BIAS_PERCENT
from hyle_perf.nanobench import parse_nanobench, select_pair
from hyle_perf.stats import equivalence_band

CASES = ((1, 17, 1024, 768, 256, 1), (64, 128, 1024, 768, 256, 1), (1, 17, 31, 47, 64, 0))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.name != 'nt':
        raise RuntimeError('this finite CPU contract requires W1-WIN')
    build = args.build.resolve(strict=True)
    profile = get_machine_profile(load_registry(), 'W1-WIN')
    probe = probe_machine(paths=[('build', build)])
    gate = evaluate_machine_profile(profile, probe, role='engineering')
    if gate['gate'] != 'PASS':
        raise RuntimeError('CPU A/A machine gate failed: ' + str(gate['errors']))
    policy = expected_child_policy(profile)
    executables = {variant: build / 'bin' / name for variant, name in (
        ('primary', 'g01_cpu_ragged_bench.exe'),
        ('placement', 'g01_cpu_ragged_bench_placement.exe'))}
    for variant, path in list(executables.items()):
        if not path.is_file():
            executables[variant] = build / path.name
    identity = {'artifacts': {k: digest(p) for k, p in executables.items()},
                'validator': digest(Path(__file__)), 'machine': gate['physical_host_identity']}
    if len(set(identity['artifacts'].values())) != 2:
        raise RuntimeError('CPU placement artifact did not change')
    cache = dict(line.split('=', 1) for line in (build/'CMakeCache.txt').read_text().splitlines()
                 if '=' in line and not line.startswith(('#', '//')))
    peer = Path(cache['HYLE_CPU_ORIGINAL_REFERENCE_SOURCE:PATH'])
    if subprocess.check_output(['git', '-C', str(peer), 'rev-parse', 'HEAD'], text=True).strip() != 'ffa4e8b80930029a35991f94e7c8a93cd67730ab':
        raise RuntimeError('wrong original CPU peer pin')
    if subprocess.check_output(['git', '-C', str(peer), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('original CPU peer is dirty')
    # The runtime lock authenticates the native source. The adapter owns the
    # graph recipe only; it does not call candidate supports/lower or a new kernel.
    identity['reference_recipe'] = digest(Path(__file__).parent.parent/'cpu_reference.cc')
    receipt_path = args.output/'cpu-ragged-aa.json'
    ready = executables['primary'].parent/'g01_cpu_ragged_aa.ready'
    ready.write_text('FAIL\n')
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old.get('identity') != identity or old.get('gate') != 'PASS':
            raise RuntimeError('retained finite CPU A/A failed or identity changed')
        if any(digest(Path(row['log'])) != row['sha256'] for row in old['runs']):
            raise RuntimeError('retained CPU A/A log changed')
        ready.write_text('PASS\n')
        print(json.dumps({'gate': 'PASS', 'reused': str(receipt_path)}))
        return 0
    raw = args.output/'cpu-ragged-aa-raw'
    raw.mkdir(parents=True)
    runs = []
    with _apply_sampling_policy(policy):
        flags = _creation_flags(policy) | subprocess.CREATE_NO_WINDOW
        for route in ('native', 'selected'):
            for variant, executable in executables.items():
                for peer_first in (False, True):
                    for allocation_first in (False, True):
                        environment = {k: v for k, v in os.environ.items()
                            if k not in PERF_ENVIRONMENT_BLOCKLIST and not any(k.startswith(p) for p in PERF_ENVIRONMENT_PREFIXES)}
                        environment.update(PERF_ENVIRONMENT_FIXED)
                        environment.update(HYLE_BENCH_AA_ROUTE=route, HYLE_PERF_CALLER_CPU=str(policy['logical_cpu']),
                            HYLE_PERF_SAMPLING_TOPOLOGY='single-logical-cpu', HYLE_PERF_RUN_NONCE=uuid.uuid4().hex)
                        if peer_first: environment['HYLE_BENCH_RIVAL_FIRST'] = '1'
                        if allocation_first: environment['HYLE_BENCH_PEER_ALLOCATION_FIRST'] = '1'
                        log = raw/f'{route}-{variant}-{int(peer_first)}-{int(allocation_first)}.log'
                        with log.open('w', encoding='utf-8') as stream:
                            result = subprocess.run([str(executable)], env=environment, creationflags=flags,
                                stdout=stream, stderr=subprocess.STDOUT)
                        if result.returncode:
                            raise RuntimeError('CPU A/A executable failed: ' + str(log))
                        text = log.read_text(encoding='utf-8')
                        markers = [line for line in text.splitlines() if line.startswith('# HYLE_CPU_RAGGED_RUN ')]
                        if len(markers) != 1: raise RuntimeError('CPU telemetry missing or duplicated')
                        actual = dict(token.split('=', 1) for token in markers[0].split() if '=' in token)
                        expected = {'run_nonce': environment['HYLE_PERF_RUN_NONCE'], 'caller_cpu': str(policy['logical_cpu']),
                            'parent_mask': policy['affinity_mask'], 'priority': str(subprocess.ABOVE_NORMAL_PRIORITY_CLASS),
                            'affinity_ok': '1', 'verify_only': '0', 'aa_route': route,
                            'affinity_topology': 'single-logical-cpu', 'paired_mode': 'nanobench-compare',
                            'paired_rounds': '16',
                            'execution_order': 'peer-first' if peer_first else 'candidate-first',
                            'allocation_order': 'peer-first' if allocation_first else 'candidate-first',
                            'placement': '1' if variant == 'placement' else '0'}
                        if any(actual.get(k) != v for k, v in expected.items()):
                            raise RuntimeError('CPU A/A actual policy mismatch')
                        rows = parse_nanobench(text)
                        ratios = {}
                        for q1, q2, k1, k2, d, causal in CASES:
                            title = f'CPU ragged BF16 Q={q1},{q2} KV={k1},{k2} D={d} causal={causal}'
                            candidate, peer_row = select_pair(rows, 'HYLE CPU ragged', 'Original CPU ragged lifecycle', title=title)
                            ratios[title] = candidate.operations_per_second/peer_row.operations_per_second
                        runs.append({'route': route, 'variant': variant, 'execution_order': expected['execution_order'],
                            'allocation_order': expected['allocation_order'], 'ratios': ratios,
                            'log': str(log.resolve()), 'sha256': digest(log)})
                        print(json.dumps({'completed': len(runs), 'total': 16}), flush=True)
    summaries, errors = {}, []
    narrow = equivalence_band(AA_MAXIMUM_BIAS_PERCENT); gross = equivalence_band(10)
    for route in ('native', 'selected'):
        matching = [row for row in runs if row['route'] == route]
        for title in matching[0]['ratios']:
            values = [row['ratios'][title] for row in matching]
            medians = {'all': statistics.median(values)}
            for factor in ('variant', 'execution_order', 'allocation_order'):
                for level in sorted({row[factor] for row in matching}):
                    medians[f'{factor}={level}'] = statistics.median(row['ratios'][title] for row in matching if row[factor] == level)
            if any(not gross[0] <= value <= gross[1] for value in values): errors.append(route+title+': gross instability')
            if any(not narrow[0] <= value <= narrow[1] for value in medians.values()): errors.append(route+title+': factor median bias')
            summaries[route+'/'+title] = {'ratios': values, 'factor_medians': medians}
    receipt = {'gate': 'FAIL' if errors else 'PASS', 'identity': identity, 'runs': runs,
        'summaries': summaries, 'errors': errors}
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    if not errors: ready.write_text('PASS\n')
    print(json.dumps({'gate': receipt['gate'], 'errors': errors}))
    return 2 if errors else 0

if __name__ == '__main__':
    raise SystemExit(main())
