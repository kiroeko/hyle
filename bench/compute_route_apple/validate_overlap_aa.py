#!/usr/bin/env python3
"""One finite original-Metal/MLX A/A batch; use existing instrument adapters."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import uuid

from validate_b1_aa import (child_environment, digest, reference_identity,
    parse_nanobench, select_pair, applied_macos_qos, evaluate_machine_profile,
    get_machine_profile, load_registry, probe_machine)

TITLES = ('BF16 router M1 N512 K8192', 'BF16 router M16 N512 K8192',
    'BF16 grouped T5 S8 E8 N1024 K4096', 'F32 grouped T5 S8 E8 N1024 K4096')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or subprocess.check_output(
        ['/usr/sbin/sysctl', '-n', 'hw.model'], text=True).strip() != 'Mac16,12':
        raise RuntimeError('requires the registered MacBook Air')
    build = args.build.resolve(strict=True)
    ready = build / 'bin/hyle_apple_overlap_aa.ready'
    ready.write_text('FAIL\n')
    profile = get_machine_profile(load_registry(), 'M1-MAC')
    gate = evaluate_machine_profile(profile, probe_machine(paths=[('build', build)],
        expected_qos='interactive'), role='engineering')
    if gate['gate'] != 'PASS':
        raise RuntimeError('machine qualification failed: ' + str(gate['errors']))
    executables = {'primary': build / 'bin/hyle_compute_route_apple_overlap_bench',
        'placement': build / 'bin/hyle_compute_route_apple_overlap_bench_placement'}
    artifacts = {key: digest(path) for key, path in executables.items()}
    if len(set(artifacts.values())) != 2:
        raise RuntimeError('placement is not independently verified')
    identity = {'artifacts': artifacts, 'reference': reference_identity(build),
        'machine': gate['physical_host_identity'], 'validator': digest(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / 'apple-overlap-aa.json'
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old['identity'] != identity or old['gate'] != 'PASS' or any(
            digest(Path(run['log'])) != run['sha256'] for run in old['runs']):
            raise RuntimeError('finite AA identity/budget cannot be reset')
        ready.write_text('PASS\n')
        print(json.dumps({'gate': 'PASS', 'reused': str(receipt_path)})); return 0
    raw = args.output / 'apple-overlap-aa-raw'
    raw.mkdir()
    runs, environments, owners = [], set(), set()
    for route in ('selected', 'native'):
        for variant, exe in executables.items():
            for rival in (False, True):
                for allocation in (False, True):
                    env = child_environment()
                    nonce = uuid.uuid4().hex
                    env.update(HYLE_BENCH_AA_ROUTE=route, HYLE_PERF_RUN_NONCE=nonce)
                    if rival: env['HYLE_BENCH_RIVAL_FIRST'] = '1'
                    if allocation: env['HYLE_BENCH_PEER_ALLOCATION_FIRST'] = '1'
                    result = subprocess.run([str(exe)], env=env, capture_output=True, text=True, timeout=180)
                    text = result.stdout + result.stderr
                    log = raw / f'{route}-{variant}-{int(rival)}-{int(allocation)}.log'
                    log.write_text(text)
                    if result.returncode: raise RuntimeError('AA process failed: ' + str(log))
                    marker = next(line for line in text.splitlines() if line.startswith('# HYLE_APPLE_OVERLAP_RUN '))
                    for token in [f'aa={route}', f'run_nonce={nonce}', 'verify=0', 'qos_ok=1',
                        'execution=' + ('peer-first' if rival else 'candidate-first'),
                        'allocation=' + ('peer-first' if allocation else 'candidate-first')]:
                        if token not in marker: raise RuntimeError('AA telemetry mismatch')
                    facts = [line for line in text.splitlines() if line.startswith('# OVERLAP_ENV ')]
                    qos = [line for line in text.splitlines() if line.startswith('# OVERLAP_OWNER_QOS ')]
                    if len(facts) != 8 or len(set(facts)) != 1 or len(qos) != 8 or len(set(qos)) != 1 or qos[0].endswith(' 0'):
                        raise RuntimeError('actual native environment/owner policy missing or inconsistent')
                    environments.update(facts); owners.update(qos)
                    parsed = parse_nanobench(text)
                    ratios = {}
                    for title in TITLES:
                        c, p = select_pair(parsed, 'Original Metal ' + title, 'Original MLX ' + title, title=title)
                        ratios[title] = c.operations_per_second / p.operations_per_second
                    runs.append({'route': route, 'variant': variant, 'rival': rival, 'allocation': allocation,
                        'ratios': ratios, 'log': str(log.resolve()), 'sha256': digest(log)})
    errors, summaries = [], {}
    if len(environments) != 1 or len(owners) != 1: errors.append('native identity/policy changed within batch')
    for route in ('selected', 'native'):
        matching = [run for run in runs if run['route'] == route]
        for title in TITLES:
            values = [run['ratios'][title] for run in matching]
            medians = {'all': statistics.median(values)}
            for factor in ('variant', 'rival', 'allocation'):
                for level in set(run[factor] for run in matching):
                    medians[f'{factor}={level}'] = statistics.median(run['ratios'][title]
                        for run in matching if run[factor] == level)
            if any(not 1/1.10 <= value <= 1.10 for value in values): errors.append(route + ':' + title + ': gross instability')
            if any(not 1/1.05 <= value <= 1.05 for value in medians.values()): errors.append(route + ':' + title + ': factor bias')
            summaries[route + ':' + title] = {'ratios': values, 'medians': medians}
    receipt = {'schema': 'hyle.apple-overlap-aa/v1', 'identity': identity,
        'gate': 'FAIL' if errors else 'PASS', 'runs': runs, 'summaries': summaries, 'errors': errors}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'gate': receipt['gate'], 'receipt': str(receipt_path), 'errors': errors}))
    if not errors: ready.write_text('PASS\n')
    return 2 if errors else 0

if __name__ == '__main__':
    with applied_macos_qos('interactive'):
        raise SystemExit(main())
