"""Finite CUDA A/A with the existing W1-WSL engineering and statistics policy."""
import argparse
import hashlib
import json
import os
import platform
import re
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

CASES = ((1, 17, 1024, 768, 256, 256, 1, 16), (64, 128, 1024, 768, 256, 256, 1, -1), (17, 1, 7, 47, 64, 128, 0, -1))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('this finite CUDA contract requires actual W1-WSL')
    build = args.build.resolve(strict=True)
    profile = get_machine_profile(load_registry(), 'W1-WSL')
    probe = probe_machine(paths=[('build', build)])
    gate = evaluate_machine_profile(profile, probe, role='engineering')
    if gate['gate'] != 'PASS':
        raise RuntimeError('CUDA A/A machine gate failed: ' + str(gate['errors']))
    policy = expected_child_policy(profile)
    executables = {variant: build / 'bin' / name for variant, name in (
        ('primary', 'g04_cuda_encoded_bench'),
        ('placement', 'g04_cuda_encoded_bench_placement'))}
    for variant, path in list(executables.items()):
        if not path.is_file():
            executables[variant] = build / path.name
    identity = {'artifacts': {k: digest(p) for k, p in executables.items()},
                'validator': digest(Path(__file__)), 'machine': gate['physical_host_identity']}
    if len(set(identity['artifacts'].values())) != 2:
        raise RuntimeError('CPU placement artifact did not change')
    cache = dict(line.split('=', 1) for line in (build/'CMakeCache.txt').read_text().splitlines()
                 if '=' in line and not line.startswith(('#', '//')))
    original = json.loads((build/'original-peer/original-flashinfer-source.json').read_text())
    receipt_file = ROOT/'evidence/correctness/inference-g04-g05-cuda-ce20-sync-fixed-numeric-rebind/receipt.json'
    preflight = json.loads(receipt_file.read_text())
    if preflight.get('state') != 'TERMINAL_PASS':
        raise RuntimeError('CUDA numerical preflight has not passed')
    numeric = preflight['successful_run']
    # Qualification requires a real Git checkout; the numerical preflight
    # authenticated an archive extraction. Location and archive transport may
    # differ, while the revision and every original header must be identical.
    for key in ('revision', 'inputs', 'all_original_headers', 'effective_headers', 'derivation', 'adaptation', 'status'):
        if original[key] != numeric['original_source'][key]:
            raise RuntimeError('original CUDA source identity changed: '+key)
    peer_root = Path(original['source'])
    if not (peer_root/'.git').exists() or \
            Path(subprocess.check_output(['git','-C',str(peer_root),'rev-parse','--show-toplevel'],text=True).strip()).resolve() != peer_root.resolve():
        raise RuntimeError('CUDA performance peer must own its Git checkout')
    prepared = Path(original['prepared_source']).resolve(strict=True)
    if prepared != (build/'original-peer/sync-fixed-flashinfer').resolve(strict=True):
        raise RuntimeError('unexpected sync-fixed CUDA peer root')
    for path, expected in original['effective_headers'].items():
        if digest(prepared/path) != expected:
            raise RuntimeError('effective CUDA peer header changed: '+path)
    for item in numeric['snapshot']['benchmark_inputs']:
        if digest(ROOT/item['path']) != item['sha256']:
            raise RuntimeError('CUDA numeric input changed: '+item['path'])
    sdk = Path(cache['Hyle_DIR:PATH']).parents[2]
    sdk_rows = numeric['sdk_artifacts']
    for item in sdk_rows:
        if digest(Path(item['path'])) != item['sha256']:
            raise RuntimeError('qualified CUDA SDK artifact changed')
    for item in numeric['sdk_recipes']+numeric['sdk_native_archives']:
        if digest(sdk/item['path']) != item['sha256']:
            raise RuntimeError('numeric CUDA SDK recipe/native archive changed')
    for item in numeric['snapshot']['product_inputs']:
        if item['path'].startswith('include/hyle/') and (sdk/item['path']).is_file() and digest(sdk/item['path']) != item['sha256']:
            raise RuntimeError('CUDA installed public input changed')
    identity['numeric_preflight'] = digest(receipt_file)
    identity['original_source'] = original
    identity['sdk_artifacts'] = sdk_rows
    identity['sdk_recipes'] = numeric['sdk_recipes']
    identity['sdk_native_archives'] = numeric['sdk_native_archives']
    receipt_path = args.output/'cuda-encoded-aa.json'
    ready = executables['primary'].parent/'g04_cuda_encoded_aa.ready'
    ready.write_text('FAIL\n')
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old.get('identity') != identity or old.get('gate') != 'PASS':
            raise RuntimeError('retained finite CUDA A/A failed or identity changed')
        if any(digest(Path(row['log'])) != row['sha256'] for row in old['runs']):
            raise RuntimeError('retained CUDA A/A log changed')
        ready.write_text('PASS\n')
        print(json.dumps({'gate': 'PASS', 'reused': str(receipt_path)}))
        return 0
    raw = args.output/'cuda-encoded-aa-raw'
    raw.mkdir(parents=True)
    runs = []
    with _apply_sampling_policy(policy):
        flags = _creation_flags(policy)
        for route in ('native', 'selected'):
            for variant, executable in executables.items():
                for peer_first in (False, True):
                    for allocation_first in (False, True):
                        environment = {k: v for k, v in os.environ.items()
                            if k not in PERF_ENVIRONMENT_BLOCKLIST and not any(k.startswith(p) for p in PERF_ENVIRONMENT_PREFIXES)}
                        environment.update(PERF_ENVIRONMENT_FIXED)
                        environment.update(HYLE_BENCH_AA_ROUTE=route, HYLE_PERF_RUN_NONCE=uuid.uuid4().hex)
                        if peer_first: environment['HYLE_BENCH_RIVAL_FIRST'] = '1'
                        if allocation_first: environment['HYLE_BENCH_PEER_ALLOCATION_FIRST'] = '1'
                        log = raw/f'{route}-{variant}-{int(peer_first)}-{int(allocation_first)}.log'
                        with log.open('w', encoding='utf-8') as stream:
                            result = subprocess.run([str(executable)], env=environment, creationflags=flags,
                                stdout=stream, stderr=subprocess.STDOUT)
                        if result.returncode:
                            raise RuntimeError('CUDA A/A executable failed: ' + str(log))
                        text = log.read_text(encoding='utf-8')
                        markers = [line for line in text.splitlines() if line.startswith('# HYLE_CUDA_ENCODED_RUN ')]
                        if len(markers) != 1: raise RuntimeError('CPU telemetry missing or duplicated')
                        actual = dict(token.split('=', 1) for token in markers[0].split() if '=' in token)
                        expected = {'run_nonce': environment['HYLE_PERF_RUN_NONCE'], 'caller_cpu': 'external',
                            'parent_mask': 'not-applicable', 'priority': 'not-applicable',
                            'affinity_ok': 'not-applicable', 'verify_only': '0', 'aa_route': route,
                            'affinity_topology': 'external', 'paired_mode': 'nanobench-compare',
                            'paired_rounds': '16',
                            'timing_schedule': 'episode-interleaved', 'arithmetic_group_size': '2048',
                            'execution_order': 'peer-first' if peer_first else 'candidate-first',
                            'allocation_order': 'peer-first' if allocation_first else 'candidate-first',
                            'placement': '1' if variant == 'placement' else '0'}
                        if any(actual.get(k) != v for k, v in expected.items()):
                            raise RuntimeError('CUDA A/A actual policy mismatch')
                        groups = re.findall(r'# CUDA_EPISODE_GROUP side=([01]) group=(\d+) count=(\d+) seconds=([0-9.eE+-]+)', text)
                        if len(groups) != 96 or any(int(n) != 2048 or float(seconds) <= 0 for _, _, n, seconds in groups):
                            raise RuntimeError('CUDA A/A complete arithmetic groups missing')
                        for first in range(0, 96, 32):
                            if {(int(s), int(g)) for s, g, _, _ in groups[first:first+32]} != {(s,g) for s in (0,1) for g in range(16)}:
                                raise RuntimeError('CUDA A/A group identity missing or duplicated')
                        facts = [re.sub(r' original=[01]$', '', line) for line in text.splitlines()
                            if line.startswith('# CUDA_ENCODED_ENV ')]
                        if len(facts) != 6 or len(set(facts)) != 1 or 'RTX 4070' not in facts[0] or 'sm=89' not in facts[0]:
                            raise RuntimeError('actual CUDA device/runtime facts missing or inconsistent')
                        if facts[0] != numeric['native_environment']:
                            raise RuntimeError('CUDA numeric/P1 device/runtime identity changed')
                        rows = parse_nanobench(text)
                        ratios = {}
                        for q1, q2, k1, k2, dk, dv, causal, window in CASES:
                            title = f'CUDA paged FP8 BF16 Q={q1},{q2} KV={k1},{k2} D={dk}/{dv} causal={causal} window={window}'
                            candidate, peer_row = select_pair(rows, 'HYLE CUDA encoded paged', 'FI69ff sync-fixed cold-expanded immutable lifecycle', title=title)
                            ratios[title] = candidate.operations_per_second/peer_row.operations_per_second
                        runs.append({'route': route, 'variant': variant, 'execution_order': expected['execution_order'],
                            'allocation_order': expected['allocation_order'], 'ratios': ratios,
                            'log': str(log.resolve()), 'sha256': digest(log)})
                        print(json.dumps({'completed': len(runs), 'total': 16}), flush=True)
    if any(digest(Path(item['path'])) != item['sha256'] for item in sdk_rows):
        raise RuntimeError('CUDA SDK artifact changed during A/A')
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
