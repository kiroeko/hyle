#!/usr/bin/env python3
"""One bounded workload A/A validator; product statistics stay in hyle-perf."""
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from hyle_perf.environment import (  # noqa: E402
    PERF_ENVIRONMENT_BLOCKLIST, PERF_ENVIRONMENT_FIXED, PERF_ENVIRONMENT_PREFIXES)
from hyle_perf.machine import (  # noqa: E402
    evaluate_machine_profile, expected_child_policy, get_machine_profile,
    load_registry, probe_machine)
from hyle_perf.nanobench import parse_nanobench, select_pair  # noqa: E402
from hyle_perf.runner import _apply_sampling_policy, _required_telemetry  # noqa: E402

PREFIX = '# HYLE_U4_CUDA_RUN '
TITLES = ('Qwen BF16 RMSNorm T1 H2048', 'Qwen BF16 RMSNorm T32 H2048',
          'Qwen BF16 packed SiLU T1 H6144', 'Qwen BF16 packed SiLU T32 H6144',
          'Qwen F32 RMSNorm T1 H2048', 'Qwen F32 RMSNorm T32 H2048')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_files(base, files):
    for name, expected in files.items():
        if digest(base / name) != expected:
            raise RuntimeError('input identity changed: ' + str(base / name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    build, output = args.build.resolve(), args.output.resolve()
    exe = build / 'hyle_cuda_provider_choices_bench.exe'
    identity = {'binary': digest(exe),
                'consumer': digest(Path(__file__).with_name('consumer.json')),
                'source': digest(Path(__file__).with_name('provider_choices.cc'))}
    receipt = output / 'u4-cuda-workload-aa.json'
    ready = build / 'u4-cuda-choices-aa.ready'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['identity'] != identity or previous['gate'] != 'PASS':
            raise RuntimeError('existing A/A receipt is failed or belongs to different inputs')
        for run in previous['runs']:
            if digest(run['log']) != run['sha256']:
                raise RuntimeError('retained A/A log changed')
        if ready.read_text().strip() != 'PASS':
            raise RuntimeError('A/A ready marker missing')
        print('reuse identical workload A/A PASS')
        return 0
    if ready.exists():
        raise RuntimeError('unbound A/A ready marker')
    profile = get_machine_profile(load_registry(), 'W1-WIN')
    probe = probe_machine()
    gate = evaluate_machine_profile(profile, probe, role='engineering',
                                    sampling_topology='native-runtime-pair')
    if gate['gate'] != 'PASS':
        raise RuntimeError('machine profile: ' + '; '.join(gate['errors']))
    policy = expected_child_policy(profile, sampling_topology='native-runtime-pair', probe=probe)
    raw = output / 'u4-cuda-aa-raw'
    raw.mkdir(parents=True)
    runs, native_facts = [], None
    with _apply_sampling_policy(policy):
        for route in ('selected', 'peer'):
            for peer_first in (False, True):
                for allocation_first in (False, True):
                    env = {k: v for k, v in os.environ.items()
                           if k not in PERF_ENVIRONMENT_BLOCKLIST and
                           not any(k.startswith(p) for p in PERF_ENVIRONMENT_PREFIXES)}
                    env.update(PERF_ENVIRONMENT_FIXED)
                    env.update(HYLE_BENCH_AA_ROUTE=route, HYLE_PERF_RUN_NONCE=uuid.uuid4().hex,
                               HYLE_PERF_CALLER_CPU=str(policy['caller_logical_cpu']),
                               HYLE_PERF_SAMPLING_TOPOLOGY='native-runtime-pair')
                    if peer_first:
                        env['HYLE_BENCH_RIVAL_FIRST'] = '1'
                    if allocation_first:
                        env['HYLE_BENCH_PEER_ALLOCATION_FIRST'] = '1'
                    log = raw / f'{route}-{int(peer_first)}-{int(allocation_first)}.log'
                    result = subprocess.run([str(exe)], env=env, capture_output=True,
                        text=True, encoding='utf-8', errors='replace', timeout=120,
                        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.ABOVE_NORMAL_PRIORITY_CLASS)
                    text = result.stdout + result.stderr
                    log.write_bytes(text.encode('utf-8'))
                    if result.returncode:
                        raise RuntimeError('A/A child failed: ' + str(log))
                    expected = {'execution_order': 'peer-first' if peer_first else 'candidate-first',
                                'allocation_order': 'peer-first' if allocation_first else 'candidate-first',
                                'aa_route': route, 'paired_mode': 'nanobench-compare',
                                'affinity_topology': 'native-runtime-pair',
                                'caller_cpu': str(policy['caller_logical_cpu']), 'affinity_ok': '1',
                                'parent_mask': policy['affinity_mask'], 'priority': policy['priority'],
                                'run_nonce': env['HYLE_PERF_RUN_NONCE'], 'verify_only': '0'}
                    _required_telemetry(text, [PREFIX], expected)
                    facts = [line for line in text.splitlines() if line.startswith('# U4_CUDA_ENV ')]
                    if len(facts) != 12 or any('sm=89 driver=617.14 driver_api=13040 runtime=13030 toolkit=13030' not in f for f in facts):
                        raise RuntimeError('actual CUDA identity outside frozen SM89/CUDA13.3 domain')
                    native_facts = sorted(set(facts))
                    rows = parse_nanobench(text)
                    ratios = {}
                    for title in TITLES:
                        candidate, peer = select_pair(rows, 'Mature provider',
                            'HYLE native', title=title)
                        ratios[title] = candidate.operations_per_second / peer.operations_per_second
                    runs.append({'route': route, 'peer_first': peer_first,
                                 'allocation_first': allocation_first, 'ratios': ratios,
                                 'log': str(log), 'sha256': digest(log)})
                    print(json.dumps({'completed': len(runs), 'total': 8, 'ratios': ratios}), flush=True)
    errors, summaries = [], {}
    for route in ('selected', 'peer'):
        group = [r for r in runs if r['route'] == route]
        for title in TITLES:
            values = [r['ratios'][title] for r in group]
            medians = {'all': statistics.median(values)}
            for factor in ('peer_first', 'allocation_first'):
                for level in (False, True):
                    medians[f'{factor}={level}'] = statistics.median(
                        r['ratios'][title] for r in group if r[factor] == level)
            if any(not 1 / 1.10 <= v <= 1.10 for v in values):
                errors.append(f'{route}/{title}: gross process instability over 10%')
            if any(not 1 / 1.05 <= v <= 1.05 for v in medians.values()):
                errors.append(f'{route}/{title}: factor median bias over 5%')
            summaries[f'{route}/{title}'] = {'ratios': values, 'factor_medians': medians}
    record = {'schema': 'hyle.u4-cuda-workload-aa/v1', 'gate': 'FAIL' if errors else 'PASS',
              'identity': identity, 'native_environment': native_facts, 'policy': policy,
              'runs': runs, 'summaries': summaries, 'errors': errors}
    receipt.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    if not errors:
        ready.write_text('PASS\n', encoding='ascii')
    print(json.dumps({'gate': record['gate'], 'errors': errors}), flush=True)
    return 2 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
