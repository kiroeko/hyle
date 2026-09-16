#!/usr/bin/env python3
"""Finite complete-region Apple A/A qualification; statistics remain in hyle-perf."""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from hyle_perf.environment import (  # noqa: E402
    PERF_ENVIRONMENT_BLOCKLIST, PERF_ENVIRONMENT_FIXED, PERF_ENVIRONMENT_PREFIXES)
from hyle_perf.nanobench import parse_nanobench, select_pair  # noqa: E402
from hyle_perf.machine import (  # noqa: E402
    applied_macos_qos, evaluate_machine_profile, get_machine_profile,
    load_registry, probe_machine)

PIN = '1f8e74e3f12f31365464a6867c6579f0e9b29d85'
PREFIX = '# HYLE_APPLE_FFN_RUN '


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child_environment() -> dict[str, str]:
    result = {k: v for k, v in os.environ.items()
              if k not in PERF_ENVIRONMENT_BLOCKLIST and not any(
                  k.startswith(p) for p in PERF_ENVIRONMENT_PREFIXES)}
    result.update(PERF_ENVIRONMENT_FIXED)
    result['HYLE_PERF_APPLE_QOS'] = 'interactive'
    return result


def reference_identity(build: Path) -> dict:
    cache = (build / 'CMakeCache.txt').read_text()
    match = re.search(r'^HYLE_APPLE_REFERENCE_SOURCE_DIR:[^=]+=(.+)$', cache, re.M)
    if not match:
        raise RuntimeError('missing pinned MLX reference checkout')
    peer = Path(match[1])
    actual = Path((build / 'bin/apple-mlx-source-path.txt').read_text().strip())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=peer, text=True).strip()
    if revision != PIN or subprocess.check_output(['git', 'status', '--porcelain'], cwd=peer):
        raise RuntimeError('MLX reference is not clean at the pinned revision')
    entries = subprocess.check_output(['git', 'ls-tree', '-rz', 'HEAD'], cwd=peer)
    count = 0
    for item in entries.split(b'\0'):
        if not item:
            continue
        metadata, raw_name = item.split(b'\t', 1)
        mode, kind, expected = metadata.split()
        if kind != b'blob':
            raise RuntimeError('unsupported nested MLX Git object')
        path = actual / os.fsdecode(raw_name)
        data = os.fsencode(os.readlink(path)) if mode == b'120000' else path.read_bytes()
        observed = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if observed != expected.decode():
            raise RuntimeError(f'compiled locked MLX differs from reference: {path}')
        count += 1
    return {'revision': revision, 'tree': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD^{tree}'], cwd=peer, text=True).strip(),
        'compiled_blob_count': count, 'metallib': digest(build / 'bin/mlx.metallib')}


def telemetry(text: str) -> dict[str, str]:
    lines = [s[len(PREFIX):] for s in text.splitlines() if s.startswith(PREFIX)]
    if len(lines) != 1:
        raise RuntimeError('expected one Apple paired-run telemetry marker')
    pairs = re.findall(r'(\w+)=(.*?)(?= \w+=|$)', lines[0])
    if len(dict(pairs)) != len(pairs):
        raise RuntimeError('duplicate Apple telemetry key')
    return dict(pairs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('compute', 'graph'), required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        raise RuntimeError('this Apple qualification requires the M1-MAC Air')
    model = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'hw.model'], text=True).strip()
    if model != 'Mac16,12':
        raise RuntimeError(f'Mac mini/other model cannot sign M1-MAC: {model}')
    build = args.build.resolve(strict=True)
    ready = build / 'bin' / f'hyle_apple_{args.mode}_aa.ready'
    ready.write_text('FAIL\n')
    probe = probe_machine(paths=[('build', build)], expected_qos='interactive')
    profile = get_machine_profile(load_registry(), 'M1-MAC')
    gate = evaluate_machine_profile(profile, probe, role='engineering')
    if gate['gate'] != 'PASS':
        raise RuntimeError(f'Apple A/A machine gate: {gate["errors"]}')
    stem = f'hyle_{args.mode}_route_apple_bench'
    executables = {v: build / 'bin' / (stem + suffix) for v, suffix in
                   (('primary', ''), ('placement', '_placement'))}
    artifacts = {v: digest(exe) for v, exe in executables.items()}
    if artifacts['primary'] == artifacts['placement']:
        raise RuntimeError('placement did not change the executable')
    identity = {'artifacts': artifacts, 'reference': reference_identity(build),
                'validator': digest(Path(__file__)),
                'machine_identity': gate['physical_host_identity'],
                'os': subprocess.check_output(['/usr/bin/sw_vers'], text=True),
                'model': model, 'caller_qos': 'interactive'}
    receipt_path = args.output / f'apple-{args.mode}-aa.json'
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old.get('identity') != identity or old.get('gate') != 'PASS':
            raise RuntimeError('A/A identity changed or previous finite attempt failed')
        if any(digest(Path(run['log'])) != run['sha256'] for run in old['runs']):
            raise RuntimeError('A/A retained raw evidence changed')
        print(json.dumps({'gate': 'PASS', 'reused_receipt': str(receipt_path),
                          'receipt_sha256': digest(receipt_path)}))
        ready.write_text('PASS\n')
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / f'apple-{args.mode}-aa-raw'
    raw.mkdir()
    runs = []
    native_facts = None
    owner_qos = None
    kind = 'Direct FFN' if args.mode == 'compute' else 'Graph FFN'
    for route in ('native', 'selected'):
        for variant, exe in executables.items():
            for peer_first in (False, True):
                for allocation_first in (False, True):
                    environment = child_environment()
                    environment.update(HYLE_BENCH_AA_ROUTE=route,
                                       HYLE_PERF_RUN_NONCE=uuid.uuid4().hex)
                    if peer_first:
                        environment['HYLE_BENCH_RIVAL_FIRST'] = '1'
                    if allocation_first:
                        environment['HYLE_BENCH_PEER_ALLOCATION_FIRST'] = '1'
                    log = raw / f'{route}-{variant}-{int(peer_first)}-{int(allocation_first)}.log'
                    try:
                        result = subprocess.run([str(exe)], env=environment, capture_output=True,
                                                text=True, timeout=120)
                    except subprocess.TimeoutExpired as exc:
                        log.write_bytes((exc.stdout or b'') + (exc.stderr or b''))
                        raise RuntimeError(f'finite A/A progress deadline: {log}') from exc
                    text = result.stdout + result.stderr
                    log.write_text(text)
                    if result.returncode:
                        raise RuntimeError(f'Apple A/A failed: {log}')
                    observed = telemetry(text)
                    expected = {'mode': 'direct' if args.mode == 'compute' else 'graph',
                                'execution_order': 'peer-first' if peer_first else 'candidate-first',
                                'allocation_order': 'peer-first' if allocation_first else 'candidate-first',
                                'aa_route': route, 'paired_mode': 'nanobench-compare',
                                'apple_qos': 'user-interactive', 'qos_ok': '1',
                                'run_nonce': environment['HYLE_PERF_RUN_NONCE'],
                                'verify_only': '0', 'natural_route': '0'}
                    if any(observed.get(k) != v for k, v in expected.items()):
                        raise RuntimeError(f'Apple A/A telemetry mismatch: {observed}')
                    facts = [s for s in text.splitlines() if s.startswith('# APPLE_ENV ')]
                    qos = re.findall(r'^# APPLE_OWNER_QOS value=(\d+)$', text, re.M)
                    if len(facts) != 4 or len(set(facts)) != 1 or PIN not in facts[0]:
                        raise RuntimeError('actual native environment missing or inconsistent')
                    if len(qos) != 4 or len(set(qos)) != 1 or int(qos[0]) == 0:
                        raise RuntimeError('actual native owner QoS missing or inconsistent')
                    if native_facts is None:
                        native_facts, owner_qos = facts[0], qos[0]
                    elif (native_facts, owner_qos) != (facts[0], qos[0]):
                        raise RuntimeError('native environment/owner policy changed within qualification')
                    pairs = parse_nanobench(text)
                    ratios = {}
                    for rows in (1, 16):
                        title = f'{kind} M={rows} K=4096 H=11008 N=4096'
                        c, p = select_pair(pairs, 'HYLE Apple ' + title,
                                           'MLX lifecycle ' + title, title=title)
                        ratios[str(rows)] = c.operations_per_second / p.operations_per_second
                    runs.append({'route': route, 'variant': variant,
                                 'execution_order': expected['execution_order'],
                                 'allocation_order': expected['allocation_order'],
                                 'ratios': ratios, 'log': str(log.resolve()), 'sha256': digest(log)})
    summaries, errors = {}, []
    for route in ('native', 'selected'):
        matching = [r for r in runs if r['route'] == route]
        for rows in ('1', '16'):
            values = [r['ratios'][rows] for r in matching]
            medians = {'all': statistics.median(values)}
            for factor in ('variant', 'execution_order', 'allocation_order'):
                for level in sorted({r[factor] for r in matching}):
                    medians[f'{factor}={level}'] = statistics.median(
                        r['ratios'][rows] for r in matching if r[factor] == level)
            if any(not 1 / 1.10 <= x <= 1.10 for x in values):
                errors.append(f'{route} M{rows}: gross process instability exceeds 10%')
            if any(not 1 / 1.05 <= x <= 1.05 for x in medians.values()):
                errors.append(f'{route} M{rows}: factor median bias exceeds 5%')
            summaries[f'{route}-M{rows}'] = {'ratios': values, 'factor_medians': medians}
    receipt = {'schema': 'hyle.apple-ffn-aa/v1', 'gate': 'FAIL' if errors else 'PASS',
               'identity': identity, 'runs': runs, 'native_environment': native_facts,
               'observed_owner_qos': owner_qos, 'summaries': summaries, 'errors': errors}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'gate': receipt['gate'], 'receipt': str(receipt_path), 'errors': errors}))
    if not errors:
        ready.write_text('PASS\n')
    return 2 if errors else 0


if __name__ == '__main__':
    try:
        with applied_macos_qos('interactive'):
            raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f'Apple FFN A/A: {exc}', file=sys.stderr)
        raise SystemExit(2)
