#!/usr/bin/env python3
"""Finite full-FFN Vulkan A/A; reuse the common machine and statistics contracts."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
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
    applied_macos_qos, evaluate_machine_profile, expected_child_policy,
    get_machine_profile, load_registry, probe_machine)

PIN = '456172ec733a135778adcd32d00e576a58232e45'
PREFIX = '# HYLE_VULKAN_FFN_RUN '


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child_environment() -> dict[str, str]:
    result = {k: v for k, v in os.environ.items()
              if k not in PERF_ENVIRONMENT_BLOCKLIST and not any(
                  k.startswith(p) for p in PERF_ENVIRONMENT_PREFIXES)}
    result.update(PERF_ENVIRONMENT_FIXED)
    return result


def marker(text: str, prefix: str) -> dict[str, str]:
    pairs = re.findall(r'(\w+)=(.*?)(?= \w+=|$)', text[len(prefix):])
    if not text.startswith(prefix) or len(dict(pairs)) != len(pairs):
        raise RuntimeError('invalid or duplicate telemetry key')
    return dict(pairs)


def reference_identity(build: Path) -> dict:
    cache = (build / 'CMakeCache.txt').read_text(encoding='utf-8')
    match = re.search(r'^HYLE_VULKAN_REFERENCE_SOURCE_DIR:[^=]+=(.+)$', cache, re.M)
    if not match:
        raise RuntimeError('missing independent pinned ggml checkout')
    peer = Path(match[1])
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=peer, text=True).strip()
    if revision != PIN or subprocess.check_output(['git', 'status', '--porcelain'], cwd=peer):
        raise RuntimeError('ggml reference is not clean at the pinned revision')
    generated = build / 'bench/compute_route_vulkan/reference-shaders'
    manifest = generated / 'identity.json'
    recorded = json.loads(manifest.read_text(encoding='utf-8'))
    if recorded['revision'] != PIN:
        raise RuntimeError('reference shaders were generated from another revision')
    for name, expected in recorded['source_blobs_sha256'].items():
        relative = 'src/ggml-vulkan/vulkan-shaders/' + name
        blob = subprocess.check_output(['git', 'show', PIN + ':' + relative], cwd=peer)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise RuntimeError('reference source identity mismatch: ' + name)
        if (peer / relative).read_bytes().replace(b'\r\n', b'\n') != blob.replace(b'\r\n', b'\n'):
            raise RuntimeError('reference working file mismatch: ' + name)
    compilers = set()
    for output in recorded['outputs']:
        if digest(generated / (output['name'] + '.spv')) != output['sha256']:
            raise RuntimeError('reference SPIR-V identity mismatch')
        compilers.add(output['command'][0])
    if len(compilers) != 1:
        raise RuntimeError('inconsistent reference shader compiler')
    compiler = Path(compilers.pop())
    return {'revision': revision, 'manifest': digest(manifest),
            'compiled_header': digest(generated / 'reference_spirv.h'),
            'compiler': digest(compiler),
            'compiler_version': subprocess.check_output([str(compiler), '--version'], text=True)}


def windows_runtime(build: Path) -> dict:
    # Every directly required DLL is beside the executable before launch. The
    # loader still discovers the actual installed ICD through its normal API.
    receipt = {}
    for name in ('msvcp140.dll', 'msvcp140_atomic_wait.dll', 'vcruntime140.dll',
                 'vcruntime140_1.dll', 'vulkan-1.dll'):
        source = Path('C:/Windows/System32') / name
        destination = build / 'bin' / name
        if not destination.exists():
            shutil.copy2(source, destination)
        if digest(destination) != digest(source):
            raise RuntimeError('side-by-side runtime identity changed: ' + name)
        receipt[name] = digest(destination)
    return receipt


def gpu_snapshot(windows: bool) -> dict:
    if not windows:
        return {}
    keys = ['name', 'uuid', 'driver_version', 'pstate', 'temperature.gpu',
            'power.draw', 'clocks.sm', 'clocks.mem', 'utilization.gpu',
            'clocks_event_reasons.sw_thermal_slowdown',
            'clocks_event_reasons.hw_thermal_slowdown']
    result = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(keys),
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True, timeout=20)
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(keys):
        raise RuntimeError('expected exactly one W1-WIN GPU')
    facts = dict(zip(keys, rows[0]))
    if (facts['name'], facts['uuid'], facts['driver_version']) != (
            'NVIDIA GeForce RTX 4070', 'GPU-e7376560-6f5d-ec19-6ee4-3dd3a2c81ef8', '610.62'):
        raise RuntimeError('GPU/driver outside the R6 W1 qualification')
    if any(facts[k] != 'Not Active' for k in keys[-2:]):
        raise RuntimeError('GPU thermal slowdown is active')
    return facts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('compute', 'graph'), required=True)
    parser.add_argument('--profile', choices=('W1-WIN', 'M1-MAC'), required=True)
    args = parser.parse_args()
    windows = args.profile == 'W1-WIN'
    if (windows and os.name != 'nt') or (not windows and platform.system() != 'Darwin'):
        raise RuntimeError('profile does not match the actual OS')
    if not windows:
        model = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'hw.model'], text=True).strip()
        if model != 'Mac16,12':
            raise RuntimeError('only the registered Air can sign M1-MAC')
    build = args.build.resolve(strict=True)
    ready = build / 'bin' / f'hyle_vulkan_{args.mode}_aa.ready'
    ready.write_text('FAIL\n', encoding='ascii')
    topology = 'native-runtime-pair' if windows else 'single-logical-cpu'
    profile = get_machine_profile(load_registry(), args.profile)
    probe = probe_machine(paths=[('build', build)], **({} if windows else {'expected_qos': 'interactive'}))
    gate = evaluate_machine_profile(profile, probe, role='engineering', sampling_topology=topology)
    if gate['gate'] != 'PASS':
        raise RuntimeError(f'Vulkan A/A machine gate failed: {gate["errors"]}')
    policy = expected_child_policy(profile, probe=probe, sampling_topology=topology)
    runtime = {}
    if windows:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.SetErrorMode(0x8003)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.GetProcessAffinityMask.argtypes = [ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        process = kernel32.GetCurrentProcess()
        if not kernel32.SetProcessAffinityMask(process, int(policy['affinity_mask'], 16)):
            raise ctypes.WinError(ctypes.get_last_error())
        observed, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(process, ctypes.byref(observed), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        if observed.value != int(policy['affinity_mask'], 16):
            raise RuntimeError('A/A process mask differs from the P1 policy')
        runtime = windows_runtime(build)
    stem = f'hyle_{args.mode}_route_vulkan_bench'
    executables = {variant: build / 'bin' / (stem + suffix + ('.exe' if windows else ''))
                   for variant, suffix in (('primary', ''), ('placement', '_placement'))}
    artifacts = {v: digest(exe) for v, exe in executables.items()}
    if artifacts['primary'] == artifacts['placement']:
        raise RuntimeError('placement did not change the executable')
    identity = {'artifacts': artifacts, 'runtime': runtime, 'reference': reference_identity(build),
                'validator': digest(Path(__file__)), 'child_policy': policy,
                'machine_identity': gate['physical_host_identity']}
    receipt_path = args.output / f'vulkan-{args.mode}-aa.json'
    before = gpu_snapshot(windows)
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text(encoding='utf-8'))
        if old.get('identity') != identity or old.get('gate') != 'PASS':
            raise RuntimeError('A/A identity changed or previous finite attempt failed')
        if any(digest(Path(run['log'])) != run['sha256'] for run in old['runs']):
            raise RuntimeError('retained A/A raw log changed')
        ready.write_text('PASS\n', encoding='ascii')
        print(json.dumps({'gate': 'PASS', 'reused_receipt': str(receipt_path),
                          'receipt_sha256': digest(receipt_path), 'gpu_after': before}))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / f'vulkan-{args.mode}-aa-raw'
    raw.mkdir()  # A partial failed attempt also cannot be silently overwritten.
    runs, native_facts = [], None
    kind = 'Direct FFN' if args.mode == 'compute' else 'Graph FFN'
    for route in ('native', 'selected'):
        for variant, executable in executables.items():
            for peer_first in (False, True):
                for allocation_first in (False, True):
                    environment = child_environment()
                    environment.update(HYLE_BENCH_AA_ROUTE=route, HYLE_PERF_RUN_NONCE=uuid.uuid4().hex)
                    if windows:
                        environment.update(HYLE_PERF_CALLER_CPU=str(policy['caller_logical_cpu']),
                                           HYLE_PERF_PARENT_MASK=policy['affinity_mask'],
                                           HYLE_PERF_PRIORITY=policy['priority'],
                                           HYLE_PERF_SAMPLING_TOPOLOGY=topology)
                    else:
                        environment['HYLE_PERF_APPLE_QOS'] = 'interactive'
                    if peer_first:
                        environment['HYLE_BENCH_RIVAL_FIRST'] = '1'
                    if allocation_first:
                        environment['HYLE_BENCH_PEER_ALLOCATION_FIRST'] = '1'
                    log = raw / f'{route}-{variant}-{int(peer_first)}-{int(allocation_first)}.log'
                    try:
                        result = subprocess.run([str(executable)], env=environment, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=120,
                            creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS if windows else 0)
                    except subprocess.TimeoutExpired as exc:
                        log.write_bytes((exc.stdout or b'') + (exc.stderr or b''))
                        raise RuntimeError('finite A/A progress deadline: ' + str(log)) from exc
                    text = result.stdout + result.stderr
                    log.write_bytes(text.encode('utf-8'))
                    if result.returncode:
                        raise RuntimeError('Vulkan A/A executable failed: ' + str(log))
                    lines = [s for s in text.splitlines() if s.startswith(PREFIX)]
                    if len(lines) != 1:
                        raise RuntimeError('expected exactly one paired-run telemetry marker')
                    observed = marker(lines[0], PREFIX)
                    expected = {'mode': 'direct' if args.mode == 'compute' else 'graph',
                                'execution_order': 'peer-first' if peer_first else 'candidate-first',
                                'allocation_order': 'peer-first' if allocation_first else 'candidate-first',
                                'aa_route': route, 'paired_mode': 'nanobench-compare',
                                'run_nonce': environment['HYLE_PERF_RUN_NONCE'],
                                'verify_only': '0',
                                'placement': '1' if variant == 'placement' else '0'}
                    if windows:
                        expected.update(affinity_topology=topology,
                                        caller_cpu=str(policy['caller_logical_cpu']),
                                        parent_mask=policy['affinity_mask'], priority=policy['priority'],
                                        affinity_ok='1')
                    else:
                        expected.update(apple_qos='user-interactive', qos_ok='1')
                    if any(observed.get(k) != v for k, v in expected.items()):
                        raise RuntimeError(f'A/A child policy mismatch: {observed}')
                    facts = [s for s in text.splitlines() if s.startswith('# VULKAN_ENV ')]
                    if len(facts) != 4 or len(set(facts)) != 1 or PIN not in facts[0]:
                        raise RuntimeError('actual Vulkan environment missing or inconsistent')
                    device = marker(facts[0], '# VULKAN_ENV ')
                    expected_device = 'NVIDIA GeForce RTX 4070' if windows else 'Apple M4'
                    if device.get('device') != expected_device:
                        raise RuntimeError('actual Vulkan device is outside the declared profile')
                    if native_facts is None:
                        native_facts = facts[0]
                    elif native_facts != facts[0]:
                        raise RuntimeError('Vulkan device/driver/compiler changed within qualification')
                    pairs, ratios = parse_nanobench(text), {}
                    for rows in (1, 16):
                        title = f'{kind} M={rows} K=4096 H=11008 N=4096'
                        candidate, peer = select_pair(pairs, 'HYLE Vulkan ' + title,
                                                       'ggml shader lifecycle ' + title, title=title)
                        ratios[str(rows)] = candidate.operations_per_second / peer.operations_per_second
                    runs.append({'route': route, 'variant': variant,
                                 'execution_order': expected['execution_order'],
                                 'allocation_order': expected['allocation_order'], 'ratios': ratios,
                                 'log': str(log.resolve()), 'sha256': digest(log)})
                    print(json.dumps({'completed': len(runs), 'total': 16, 'ratios': ratios}), flush=True)
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
            if any(not 1 / 1.10 <= v <= 1.10 for v in values):
                errors.append(f'{route} M{rows}: gross process instability exceeds 10%')
            if any(not 1 / 1.05 <= v <= 1.05 for v in medians.values()):
                errors.append(f'{route} M{rows}: factor median bias exceeds 5%')
            summaries[f'{route}-M{rows}'] = {'ratios': values, 'factor_medians': medians}
    receipt = {'schema': 'hyle.vulkan-ffn-aa/v1', 'gate': 'FAIL' if errors else 'PASS',
               'identity': identity, 'gpu_before': before, 'gpu_after': gpu_snapshot(windows),
               'native_environment': native_facts, 'runs': runs, 'summaries': summaries, 'errors': errors}
    receipt_path.write_bytes((json.dumps(receipt, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'gate': receipt['gate'], 'receipt': str(receipt_path), 'errors': errors}))
    if not errors:
        ready.write_text('PASS\n', encoding='ascii')
    return 2 if errors else 0


if __name__ == '__main__':
    try:
        with applied_macos_qos('interactive' if platform.system() == 'Darwin' else None):
            raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f'Vulkan FFN A/A: {exc}', file=sys.stderr)
        raise SystemExit(2)
