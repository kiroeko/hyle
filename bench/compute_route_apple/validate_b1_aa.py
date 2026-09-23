#!/usr/bin/env python3
"""Finite A/A qualification for the Apple M2 B1 complete-region bundle."""
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
PREFIX = '# HYLE_APPLE_B1_RUN '
TITLE = 'Graph B1 BF16 Scatter Norm Gate R=17 W=2048'


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
    revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=peer, text=True).strip()
    if revision != PIN or subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=peer):
        raise RuntimeError('MLX reference is not clean at the pinned revision')
    manifest_path = ROOT / 'third_party/mlx_cast/patches.json'
    manifest = json.loads(manifest_path.read_text())
    patches = {entry['path']: entry for entry in manifest['files']}
    if len(patches) != len(manifest['files']):
        raise RuntimeError('duplicate path in locked MLX patch manifest')
    entries = subprocess.check_output(['git', 'ls-tree', '-rz', 'HEAD'], cwd=peer)
    count = 0
    observed_patches = {}
    for item in entries.split(b'\0'):
        if not item:
            continue
        metadata, raw_name = item.split(b'\t', 1)
        mode, kind, expected = metadata.split()
        if kind != b'blob':
            raise RuntimeError('unsupported nested MLX Git object')
        path = actual / os.fsdecode(raw_name)
        data = os.fsencode(os.readlink(path)) if mode == b'120000' else path.read_bytes()
        relative = os.fsdecode(raw_name)
        patch = patches.get(relative)
        if patch is None:
            observed = hashlib.sha1(
                b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if observed != expected.decode():
                raise RuntimeError(
                    f'compiled locked MLX differs outside admitted patches: {path}')
        else:
            upstream = peer / relative
            upstream_data = (os.fsencode(os.readlink(upstream))
                             if mode == b'120000' else upstream.read_bytes())
            upstream_hash = hashlib.sha256(upstream_data).hexdigest()
            result_hash = hashlib.sha256(data).hexdigest()
            if upstream_hash != patch['before_sha256']:
                raise RuntimeError(
                    f'locked MLX patch input differs from manifest: {relative}')
            if result_hash != patch['after_sha256']:
                raise RuntimeError(
                    f'compiled MLX patch result differs from manifest: {relative}')
            observed_patches[relative] = {
                'before_sha256': upstream_hash,
                'after_sha256': result_hash,
            }
        count += 1
    if set(observed_patches) != set(patches):
        raise RuntimeError('compiled MLX did not consume the exact locked patch set')
    return {
        'revision': revision,
        'tree': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD^{tree}'], cwd=peer, text=True).strip(),
        'compiled_blob_count': count,
        'patch_manifest_sha256': digest(manifest_path),
        'patched_files': observed_patches,
        'metallib': digest(build / 'bin/mlx.metallib'),
    }


def telemetry(text: str) -> dict[str, str]:
    lines = [line[len(PREFIX):] for line in text.splitlines()
             if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise RuntimeError('expected one Apple B1 paired-run telemetry marker')
    pairs = re.findall(r'(\w+)=(.*?)(?= \w+=|$)', lines[0])
    if len(dict(pairs)) != len(pairs):
        raise RuntimeError('duplicate Apple B1 telemetry key')
    return dict(pairs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        raise RuntimeError('this Apple qualification requires the M1-MAC Air')
    model = subprocess.check_output(
        ['/usr/sbin/sysctl', '-n', 'hw.model'], text=True).strip()
    if model != 'Mac16,12':
        raise RuntimeError(f'Mac mini/other model cannot sign M1-MAC: {model}')
    build = args.build.resolve(strict=True)
    executable = build / 'bin/hyle_graph_route_apple_b1_bench'
    ready = build / 'bin/hyle_apple_b1_aa.ready'
    ready.write_text('FAIL\n')
    probe = probe_machine(paths=[('build', build)], expected_qos='interactive')
    profile = get_machine_profile(load_registry(), 'M1-MAC')
    gate = evaluate_machine_profile(profile, probe, role='engineering')
    if gate['gate'] != 'PASS':
        raise RuntimeError(f'Apple B1 A/A machine gate: {gate["errors"]}')
    identity = {
        'artifact': digest(executable),
        'reference': reference_identity(build),
        'validator': digest(Path(__file__)),
        'machine_identity': gate['physical_host_identity'],
        'os': subprocess.check_output(['/usr/bin/sw_vers'], text=True),
        'model': model,
        'caller_qos': 'interactive',
    }
    receipt_path = args.output / 'apple-b1-aa.json'
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
    raw = args.output / 'apple-b1-aa-raw'
    raw.mkdir()
    runs = []
    native_facts = None
    owner_qos = None
    for route in ('native', 'selected'):
        for peer_first in (False, True):
            for allocation_first in (False, True):
                environment = child_environment()
                environment.update(
                    HYLE_BENCH_AA_ROUTE=route,
                    HYLE_PERF_RUN_NONCE=uuid.uuid4().hex)
                if peer_first:
                    environment['HYLE_BENCH_RIVAL_FIRST'] = '1'
                if allocation_first:
                    environment['HYLE_BENCH_PEER_ALLOCATION_FIRST'] = '1'
                log = raw / f'{route}-{int(peer_first)}-{int(allocation_first)}.log'
                try:
                    result = subprocess.run(
                        [str(executable)], env=environment, capture_output=True,
                        text=True, timeout=120)
                except subprocess.TimeoutExpired as exc:
                    log.write_bytes((exc.stdout or b'') + (exc.stderr or b''))
                    raise RuntimeError(
                        f'finite Apple B1 A/A progress deadline: {log}') from exc
                text = result.stdout + result.stderr
                log.write_text(text)
                if result.returncode:
                    raise RuntimeError(f'Apple B1 A/A failed: {log}')
                observed = telemetry(text)
                expected = {
                    'mode': 'graph',
                    'execution_order': ('peer-first' if peer_first
                                        else 'candidate-first'),
                    'allocation_order': ('peer-first' if allocation_first
                                         else 'candidate-first'),
                    'aa_route': route,
                    'paired_mode': 'nanobench-compare',
                    'apple_qos': 'user-interactive',
                    'qos_ok': '1',
                    'run_nonce': environment['HYLE_PERF_RUN_NONCE'],
                    'verify_only': '0',
                }
                if any(observed.get(key) != value
                       for key, value in expected.items()):
                    raise RuntimeError(
                        f'Apple B1 A/A telemetry mismatch: {observed}')
                facts = [line for line in text.splitlines()
                         if line.startswith('# APPLE_ENV ')]
                qos = re.findall(r'^# APPLE_OWNER_QOS value=(\d+)$', text, re.M)
                if len(facts) != 2 or len(set(facts)) != 1 or PIN not in facts[0]:
                    raise RuntimeError(
                        'actual Apple B1 native environment missing or inconsistent')
                if len(qos) != 2 or len(set(qos)) != 1 or int(qos[0]) == 0:
                    raise RuntimeError(
                        'actual Apple B1 native owner QoS missing or inconsistent')
                if native_facts is None:
                    native_facts, owner_qos = facts[0], qos[0]
                elif (native_facts, owner_qos) != (facts[0], qos[0]):
                    raise RuntimeError(
                        'Apple B1 environment/owner policy changed within qualification')
                pairs = parse_nanobench(text)
                candidate, peer = select_pair(
                    pairs, 'HYLE Apple ' + TITLE, 'MLX lifecycle ' + TITLE,
                    title=TITLE)
                runs.append({
                    'route': route,
                    'execution_order': expected['execution_order'],
                    'allocation_order': expected['allocation_order'],
                    'ratio': candidate.operations_per_second /
                             peer.operations_per_second,
                    'log': str(log.resolve()),
                    'sha256': digest(log),
                })

    summaries, errors = {}, []
    for route in ('native', 'selected'):
        matching = [run for run in runs if run['route'] == route]
        values = [run['ratio'] for run in matching]
        medians = {'all': statistics.median(values)}
        for factor in ('execution_order', 'allocation_order'):
            for level in sorted({run[factor] for run in matching}):
                medians[f'{factor}={level}'] = statistics.median(
                    run['ratio'] for run in matching if run[factor] == level)
        if any(not 1 / 1.10 <= value <= 1.10 for value in values):
            errors.append(f'{route}: gross process instability exceeds 10%')
        if any(not 1 / 1.05 <= value <= 1.05 for value in medians.values()):
            errors.append(f'{route}: factor median bias exceeds 5%')
        summaries[route] = {'ratios': values, 'factor_medians': medians}
    receipt = {
        'schema': 'hyle.apple-b1-aa/v1',
        'gate': 'FAIL' if errors else 'PASS',
        'identity': identity,
        'runs': runs,
        'native_environment': native_facts,
        'observed_owner_qos': owner_qos,
        'summaries': summaries,
        'errors': errors,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'gate': receipt['gate'], 'receipt': str(receipt_path),
                      'errors': errors}))
    if not errors:
        ready.write_text('PASS\n')
    return 2 if errors else 0


if __name__ == '__main__':
    try:
        with applied_macos_qos('interactive'):
            raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f'Apple B1 A/A: {exc}', file=sys.stderr)
        raise SystemExit(2)
