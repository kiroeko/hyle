"""Finite original/selected resident A/A using existing v1 policy and parser."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from hyle_perf.environment import (PERF_ENVIRONMENT_BLOCKLIST,
    PERF_ENVIRONMENT_FIXED, PERF_ENVIRONMENT_PREFIXES)
from hyle_perf.machine import (applied_macos_qos, evaluate_machine_profile,
    get_machine_profile, load_registry, probe_machine)
from hyle_perf.nanobench import parse_nanobench, select_pair
from hyle_perf.controls import AA_MAXIMUM_BIAS_PERCENT
from hyle_perf.stats import equivalence_band

PIN = '456172ec733a135778adcd32d00e576a58232e45'
CASES = ((1, 17, 1024, 768, 256, 256, 1, 16), (64, 128, 1024, 768, 256, 256, 1, -1), (17, 1, 7, 47, 64, 128, 0, -1))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def marker(line):
    return dict(re.findall(r'(\w+)=(.*?)(?= \w+=|$)', line.split(' ', 2)[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or subprocess.check_output(
            ['/usr/sbin/sysctl', '-n', 'hw.model'], text=True).strip() != 'Mac16,12':
        raise RuntimeError('this finite P1 is qualified only on the actual MacBook Air')
    build = args.build.resolve(strict=True)
    profile = get_machine_profile(load_registry(), 'M1-MAC')
    probe = probe_machine(paths=[('build', build)], expected_qos='interactive')
    gate = evaluate_machine_profile(profile, probe, role='engineering')
    if gate['gate'] != 'PASS':
        raise RuntimeError('A/A machine gate failed: ' + str(gate['errors']))
    executables = {variant: build / name for variant, name in (
        ('primary', 'g04_vulkan_encoded_bench'), ('placement', 'g04_vulkan_encoded_bench_placement'))}
    for variant, path in list(executables.items()):
        if not path.is_file(): executables[variant] = build / 'bin' / path.name
    artifacts = {variant: digest(path) for variant, path in executables.items()}
    if artifacts['primary'] == artifacts['placement']:
        raise RuntimeError('placement executable identity did not change')
    identity = {'artifacts': artifacts, 'validator': digest(Path(__file__)), 'machine': gate['physical_host_identity']}
    generated = build/'reference-shaders'
    identity['original_shader_identity'] = digest(generated/'identity.json')
    identity['generated_spirv'] = digest(generated/'reference_spirv.h')
    shader_record = json.loads((generated/'identity.json').read_text())
    if shader_record['revision'] != PIN: raise RuntimeError('wrong shader pin')
    for shader in shader_record['outputs']:
        if digest(generated/(shader['name']+'.spv')) != shader['sha256']: raise RuntimeError('original shader changed')
    # New numerical preflight is mandatory; static review is insufficient.
    preflight = json.loads((ROOT/'evidence/correctness/inference-g04-g05-vulkan-encoded-page-preflight/receipt.json').read_text())
    if preflight['verdict'] != 'PASS_NUMERICAL_LIFECYCLE_PREFLIGHT_ONLY':
        raise RuntimeError('encoded page numerical preflight not qualified')
    for item in preflight['inputs']:
        if digest(Path(__file__).parent.parent/item['path']) != item['sha256']:
            raise RuntimeError('benchmark input changed after numerical preflight: ' + item['path'])
    for item in preflight['product_input_identity']['inputs']:
        if digest(ROOT/item['path']) != item['sha256']:
            raise RuntimeError('GPU/common product input changed: ' + item['path'])
    if identity['generated_spirv'] != preflight['original_spirv_header_sha256']:
        raise RuntimeError('original flash SPIRV changed after numerical preflight')
    cache = dict(line.split('=',1) for line in (build/'CMakeCache.txt').read_text().splitlines()
                 if '=' in line and not line.startswith(('#','//')))
    sdk = Path(cache['Hyle_DIR:PATH']).parents[2]
    sdk_record = json.loads((Path(__file__).parent/'qualified-sdk-artifacts.json').read_text())
    if str(sdk) != sdk_record['prefix']:
        raise RuntimeError('wrong qualified SDK prefix')
    sdk_rows = sdk_record['qualified_hyle_artifacts'] + sdk_record['supplemental_native_archives'] + sdk_record['qualified_sdk_recipes']
    for item in sdk_rows:
        if digest(sdk/item['path']) != item['sha256']:
            raise RuntimeError('qualified SDK artifact changed: ' + item['path'])
    installed_headers = {}
    for path in (sdk/'include/hyle').rglob('*.h'):
        relative = path.relative_to(sdk/'include');source = ROOT/'include'/relative
        if source.is_file() and digest(path) != digest(source):
            raise RuntimeError('installed public header differs: ' + str(relative))
        installed_headers[str(relative)] = digest(path)
    identity['qualified_sdk_artifacts'] = {item['path']:item['sha256'] for item in sdk_rows}
    identity['installed_headers'] = installed_headers
    identity['numerical_preflight'] = digest(ROOT/'evidence/correctness/inference-g04-g05-vulkan-encoded-page-preflight/receipt.json')
    receipt_path = args.output / 'vulkan-encoded-aa.json'
    ready = executables['primary'].parent / 'g04_vulkan_encoded_page_aa.ready'
    ready.write_text('FAIL\n')
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old.get('identity') != identity or old.get('gate') != 'PASS':
            raise RuntimeError('retained A/A identity changed or finite attempt failed')
        if any(digest(Path(row['log'])) != row['sha256'] for row in old['runs']):
            raise RuntimeError('retained A/A raw log changed')
        ready.write_text('PASS\n'); print(json.dumps({'gate': 'PASS', 'reused': str(receipt_path)})); return 0
    raw = args.output / 'vulkan-encoded-aa-raw'
    raw.mkdir(parents=True)  # Preserve partial failures; never replace a run.
    runs, environment_fact = [], None
    for route in ('native', 'selected'):
        for variant, executable in executables.items():
            for peer_first in (False, True):
                for allocation_first in (False, True):
                    environment = {k: v for k, v in os.environ.items()
                        if k not in PERF_ENVIRONMENT_BLOCKLIST and not any(k.startswith(p) for p in PERF_ENVIRONMENT_PREFIXES)}
                    environment.update(PERF_ENVIRONMENT_FIXED)
                    environment.update(HYLE_BENCH_AA_ROUTE=route, HYLE_PERF_APPLE_QOS='interactive',
                                       HYLE_PERF_RUN_NONCE=uuid.uuid4().hex)
                    if peer_first: environment['HYLE_BENCH_RIVAL_FIRST'] = '1'
                    if allocation_first: environment['HYLE_BENCH_PEER_ALLOCATION_FIRST'] = '1'
                    log = raw / f'{route}-{variant}-{int(peer_first)}-{int(allocation_first)}.log'
                    with log.open('w') as stream:
                        result = subprocess.run([str(executable)], env=environment, stdout=stream, stderr=subprocess.STDOUT)
                    if result.returncode: raise RuntimeError('A/A executable failed: ' + str(log))
                    text = log.read_text()
                    markers = [line for line in text.splitlines() if line.startswith('# HYLE_ENCODED_PAGE_RUN ')]
                    if len(markers) != 1: raise RuntimeError('missing or duplicate telemetry')
                    actual = marker(markers[0])
                    expected = {'run_nonce': environment['HYLE_PERF_RUN_NONCE'], 'apple_qos': 'user-interactive',
                        'qos_ok': '1', 'verify_only': '0', 'aa_route': route,
                        'execution_order': 'peer-first' if peer_first else 'candidate-first',
                        'allocation_order': 'peer-first' if allocation_first else 'candidate-first',
                        'placement': '1' if variant == 'placement' else '0', 'backend': 'Vulkan', 'paired_rounds': '16'}
                    if any(actual.get(k) != v for k, v in expected.items()): raise RuntimeError('A/A policy mismatch')
                    facts = [line for line in text.splitlines() if line.startswith('# ENCODED_PAGE_ENV ')]
                    if len(facts) != 6 or len(set(facts)) != 1 or 'device=Apple M4' not in facts[0] or 'backend=Vulkan' not in facts[0] or '456172ec' not in facts[0]:
                        raise RuntimeError('actual native device/library facts missing or inconsistent')
                    if environment_fact is None: environment_fact = facts[0]
                    elif environment_fact != facts[0]: raise RuntimeError('Vulkan facts changed during A/A')
                    pairs = parse_nanobench(text); ratios = {}
                    for q1, q2, k1, k2, d, dv, causal, window in CASES:
                        title = f'Vulkan page16 F16/Q8 Q={q1},{q2} KV={k1},{k2} D={d} Dv={dv} causal={causal} window={window}'
                        candidate, peer = select_pair(pairs, 'HYLE Vulkan encoded page16', 'Original Vulkan Q8 gathered page lifecycle', title=title)
                        ratios[title] = candidate.operations_per_second / peer.operations_per_second
                    runs.append({'route': route, 'variant': variant, 'execution_order': expected['execution_order'],
                        'allocation_order': expected['allocation_order'], 'ratios': ratios,
                        'log': str(log.resolve()), 'sha256': digest(log)})
                    print(json.dumps({'completed': len(runs), 'total': 16}), flush=True)
    if any(digest(sdk/item['path']) != item['sha256'] for item in sdk_rows):
        raise RuntimeError('SDK artifact changed during A/A')
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
            if any(not gross[0] <= value <= gross[1] for value in values): errors.append(route + title + ': gross instability')
            if any(not narrow[0] <= value <= narrow[1] for value in medians.values()): errors.append(route + title + ': factor median bias')
            summaries[route + '/' + title] = {'ratios': values, 'factor_medians': medians}
    receipt = {'gate': 'FAIL' if errors else 'PASS', 'identity': identity, 'runs': runs,
               'native_environment': environment_fact, 'summaries': summaries, 'errors': errors}
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    if not errors: ready.write_text('PASS\n')
    print(json.dumps({'gate': receipt['gate'], 'errors': errors})); return 2 if errors else 0


if __name__ == '__main__':
    with applied_macos_qos('interactive' if platform.system() == 'Darwin' else None):
        raise SystemExit(main())
