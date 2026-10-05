#!/usr/bin/env python3
"""Bounded D5 native workload A/A; formal statistics stay in hyle-perf."""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
from hyle_perf.environment import (  # noqa: E402
    PERF_ENVIRONMENT_BLOCKLIST, PERF_ENVIRONMENT_FIXED, PERF_ENVIRONMENT_PREFIXES)
from hyle_perf.machine import (  # noqa: E402
    applied_macos_qos, evaluate_machine_profile, expected_child_policy,
    get_machine_profile, load_registry, probe_machine)
from hyle_perf.nanobench import parse_nanobench, select_pair  # noqa: E402
from hyle_perf.runner import _apply_sampling_policy, _required_telemetry  # noqa: E402

PREFIX = '# HYLE_D5_FRAGMENT_RUN '
TITLES = ('GLM 5.3 Flash official FP8 MoE H4096 N2048 top8 cycle2',
          'DeepSeek V4 Flash official FP4 FP8 MoE H4096 N2048 top6 cycle3')


def digest(path, *, source=False):
    data = Path(path).read_bytes()
    # Git working-tree EOL conversion differs between Windows and the Mac
    # validation copy. Only source text is normalized; SDK/input bytes never are.
    if source:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def check_files(base, files, *, source=False):
    for name, expected in files.items():
        if digest(base / name, source=source) != expected:
            raise RuntimeError('input identity changed: ' + str(base / name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', choices=('cpu', 'apple', 'vulkan'), required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    windows = os.name == 'nt'
    if windows:
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    build, output = args.build.resolve(), args.output.resolve()
    exe = build / ('hyle_graph_route_' + args.route + '_d5_bench' + ('.exe' if windows else ''))
    input_path = Path(__file__).with_name('inputs.json')
    declared = json.loads(input_path.read_text(encoding='utf-8'))
    route = declared['routes'][args.route]
    paths = json.loads((build / 'd5-input-paths.json').read_text(encoding='utf-8'))
    sdk = Path(paths['sdk']).resolve()
    sdk_manifest = ROOT / route['sdk_manifest']
    if digest(sdk_manifest, source=True) != route['sdk_manifest_sha256']:
        raise RuntimeError('SDK manifest identity changed')
    check_files(sdk, json.loads(sdk_manifest.read_text(encoding='utf-8')))
    check_files(ROOT, declared['source_files'], source=True)
    check_files(Path(declared['torch_sdk']), declared['torch_files'])
    reference = Path(paths['reference']).resolve()
    check_files(reference, declared['source_files'], source=True)
    identity = {'binary': digest(exe), 'inputs': digest(input_path, source=True),
                'route': args.route, 'sdk_manifest': route['sdk_manifest_sha256']}
    receipt = output / 'd5-workload-aa.json'
    ready = build / 'd5-fragment-aa.ready'
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding='utf-8'))
        if previous['identity'] != identity or previous['gate'] != 'PASS':
            raise RuntimeError('existing A/A receipt failed or belongs to different inputs')
        for run in previous['runs']:
            if digest(run['log']) != run['sha256']:
                raise RuntimeError('retained A/A log changed')
        if ready.read_text(encoding='ascii').strip() != 'PASS':
            raise RuntimeError('A/A ready marker missing')
        print('reuse identical workload A/A PASS')
        return 0
    if ready.exists():
        raise RuntimeError('unbound A/A ready marker')
    profile = get_machine_profile(load_registry(), 'W1-WIN' if windows else 'M1-MAC')
    topology = 'native-runtime-pair' if windows else 'single-logical-cpu'
    qos = None if windows else 'interactive'
    raw = output / 'd5-aa-raw'
    raw.mkdir(parents=True)
    runs = []
    with applied_macos_qos(qos):
        probe = probe_machine(expected_qos=qos)
        gate = evaluate_machine_profile(profile, probe, role='engineering',
                                        sampling_topology=topology)
        if gate['gate'] != 'PASS':
            raise RuntimeError('machine profile: ' + '; '.join(gate['errors']))
        policy = expected_child_policy(profile, sampling_topology=topology, probe=probe)
        with _apply_sampling_policy(policy):
            for arm in ('selected', 'peer'):
                for peer_first in (False, True):
                    for allocation_first in (False, True):
                        env = {k: v for k, v in os.environ.items()
                               if k not in PERF_ENVIRONMENT_BLOCKLIST and
                               not any(k.startswith(p) for p in PERF_ENVIRONMENT_PREFIXES)}
                        env.update(PERF_ENVIRONMENT_FIXED)
                        env.update(HYLE_BENCH_AA_ROUTE=arm, HYLE_PERF_RUN_NONCE=uuid.uuid4().hex)
                        expected = {'execution_order': 'peer-first' if peer_first else 'candidate-first',
                                    'allocation_order': 'peer-first' if allocation_first else 'candidate-first',
                                    'aa_route': arm, 'paired_mode': 'native-fixed-cycle',
                                    'run_nonce': env['HYLE_PERF_RUN_NONCE'], 'verify_only': '0'}
                        if windows:
                            env.update(HYLE_PERF_CALLER_CPU=str(policy['caller_logical_cpu']),
                                       HYLE_PERF_SAMPLING_TOPOLOGY=topology)
                            expected.update(affinity_topology=topology,
                                caller_cpu=str(policy['caller_logical_cpu']), affinity_ok='1',
                                parent_mask=policy['affinity_mask'], priority=policy['priority'])
                        else:
                            env['HYLE_PERF_APPLE_QOS'] = qos
                            expected.update(apple_qos='user-interactive', qos_ok='1')
                        if peer_first:
                            env['HYLE_BENCH_RIVAL_FIRST'] = '1'
                        if allocation_first:
                            env['HYLE_BENCH_PEER_ALLOCATION_FIRST'] = '1'
                        log = raw / f'{arm}-{int(peer_first)}-{int(allocation_first)}.log'
                        flags = (subprocess.CREATE_NO_WINDOW | subprocess.ABOVE_NORMAL_PRIORITY_CLASS
                                 if windows else 0)
                        result = subprocess.run([str(exe)], env=env, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=180,
                            creationflags=flags)
                        text = result.stdout + result.stderr
                        log.write_bytes(text.encode('utf-8'))
                        if result.returncode:
                            raise RuntimeError('A/A child failed: ' + str(log))
                        _required_telemetry(text, [PREFIX], expected)
                        native = route.get('native_environment')
                        if native:
                            facts = [line for line in text.splitlines()
                                     if line.startswith('# D5_' + args.route.upper() + '_ENV ')]
                            if len(facts) != 4 or set(facts) != {native}:
                                raise RuntimeError('native environment outside frozen identity')
                        rows = parse_nanobench(text)
                        ratios = {}
                        for title in TITLES:
                            candidate, peer = select_pair(rows, 'Graph complete fragment',
                                'Direct complete fragment', title=title)
                            ratios[title] = candidate.operations_per_second / peer.operations_per_second
                        runs.append({'route': arm, 'peer_first': peer_first,
                                     'allocation_first': allocation_first, 'ratios': ratios,
                                     'log': str(log), 'sha256': digest(log)})
                        print(json.dumps({'completed': len(runs), 'total': 8, 'ratios': ratios}), flush=True)
    errors, summaries = [], {}
    for arm in ('selected', 'peer'):
        group = [r for r in runs if r['route'] == arm]
        for title in TITLES:
            values = [r['ratios'][title] for r in group]
            medians = {'all': statistics.median(values)}
            for factor in ('peer_first', 'allocation_first'):
                for level in (False, True):
                    medians[f'{factor}={level}'] = statistics.median(
                        r['ratios'][title] for r in group if r[factor] == level)
            if any(not 1 / 1.10 <= v <= 1.10 for v in values):
                errors.append(f'{arm}/{title}: gross process instability over 10%')
            if any(not 1 / 1.05 <= v <= 1.05 for v in medians.values()):
                errors.append(f'{arm}/{title}: factor median bias over 5%')
            summaries[f'{arm}/{title}'] = {'ratios': values, 'factor_medians': medians}
    record = {'schema': 'hyle.d5-native-workload-aa/v1', 'gate': 'FAIL' if errors else 'PASS',
              'identity': identity, 'policy': policy, 'native_environment': route.get('native_environment'),
              'runs': runs, 'summaries': summaries, 'errors': errors}
    receipt.write_bytes((json.dumps(record, indent=2) + '\n').encode('utf-8'))
    if not errors:
        ready.write_text('PASS\n', encoding='ascii')
    print(json.dumps({'gate': record['gate'], 'errors': errors}), flush=True)
    return 2 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
