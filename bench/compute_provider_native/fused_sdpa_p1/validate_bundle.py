"""Verify frozen installed SDK and fresh workload A/A; reuse common parser/policy."""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import statistics
import struct
import subprocess
import sys
import time
import types

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))


def loaded_code(path):
    data = path.read_bytes()
    pe = struct.unpack_from('<I', data, 60)[0]
    count = struct.unpack_from('<H', data, pe + 6)[0]
    optional = struct.unpack_from('<H', data, pe + 20)[0]
    for index in range(count):
        offset = pe + 24 + optional + 40 * index
        if data[offset:offset + 8].rstrip(b'\0') == b'.text':
            size, start = struct.unpack_from('<II', data, offset + 16)
            return hashlib.sha256(data[start:start + size]).hexdigest()
    raise RuntimeError('executable has no .text section')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', required=True)
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    build = pathlib.Path(args.build).resolve()
    exe = build / 'hyle_native_domain_bench.exe'
    sdk = json.loads((HERE / 'sdk-identity.json').read_text())
    prefix = pathlib.Path(sdk['prefix'])
    for name, digest in sdk['files'].items():
        assert hashlib.sha256((prefix / name).read_bytes()).hexdigest() == digest, name
    for name, digest in sdk['source_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for name, digest in sdk['runtime_files'].items():
        assert hashlib.sha256((build / name).read_bytes()).hexdigest() == digest, name
    spec = importlib.util.spec_from_file_location(
        'reuse_compute_aa', ROOT / 'scripts/validate-compute-route-p1-bench.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.PREFIX = '# HYLE_NATIVE_DOMAIN_RUN '
    contract = json.loads((HERE / 'full-catalog-contract.json').read_text())
    old.WORKLOADS = tuple((w['id'], w['pair']['candidate'], w['pair']['peer'])
                          for w in contract['workloads'])
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(build.glob('*.dll'))}
    artifacts[exe.name + '.text'] = loaded_code(exe)
    if not args.collect:
        evidence = json.loads((HERE / 'preflight.json').read_text())
        assert evidence['artifacts'] == artifacts, 'loaded artifact changed'
        catalog = json.loads((HERE / 'contract.json').read_text())
        assert {w['id'] for w in catalog['workloads']} == set(evidence['passed'])
        assert hashlib.sha256((HERE / 'native_domain_bench.cc').read_bytes()).hexdigest() == evidence['source_sha256']
        print(json.dumps({'gate': 'PASS', 'qualified_cases': len(evidence['passed']),
                          'unqualified_cases': len(evidence['failed']),
                          'fresh_aa_sha256': hashlib.sha256((HERE / 'preflight.json').read_bytes()).hexdigest()}))
        return
    raw = build / ('three-preflight-' + str(time.time_ns()))
    raw.mkdir()
    counter = 0
    original_subprocess_run = subprocess.run

    def launch(command, **kwargs):
        nonlocal counter
        if os.name == 'nt':
            kwargs['creationflags'] = kwargs.get('creationflags', 0) | 0x08000000
        result = original_subprocess_run(command, **kwargs)
        counter += 1
        (raw / (str(counter) + '.log')).write_text(
            (result.stdout or '') + (result.stderr or ''), encoding='utf-8')
        return result

    old.subprocess = types.SimpleNamespace(run=launch)
    env = old._environment('none', False, False, '2')
    env['HYLE_BENCH_VERIFY_ONLY'] = '1'
    numeric = launch([str(exe)], capture_output=True, text=True, encoding='utf-8',
                     env=env, timeout=120)
    assert numeric.returncode == 0, numeric.stderr
    runs = []
    for route in ('native', 'selected'):
        for execution in (False, True):
            for allocation in (False, True):
                runs.append(old._run(exe, route=route, peer_first=execution,
                                     peer_allocation_first=allocation, caller_cpu='2'))
    passed, failed = [], {}
    summaries = {}
    for identifier, _, _ in old.WORKLOADS:
        reasons = []
        for route in ('native', 'selected'):
            matching = [r for r in runs if r['route'] == route]
            values = [r['ratios'][identifier] for r in matching]
            medians = {'all': statistics.median(values)}
            for factor in ('execution_order', 'allocation_order'):
                for level in ('peer-first', 'candidate-first'):
                    medians[factor + '=' + level] = statistics.median(
                        r['ratios'][identifier] for r in matching if r[factor] == level)
            summaries[route + ':' + identifier] = medians
            if any(not 1 / 1.10 <= value <= 1.10 for value in values):
                reasons.append(route + ':gross-instability')
            if any(not 1 / 1.05 <= value <= 1.05 for value in medians.values()):
                reasons.append(route + ':fixed-factor-bias')
        if reasons:
            failed[identifier] = reasons
        else:
            passed.append(identifier)
    evidence = {'schema': 'hyle.cpu-three-operator-aa/v1',
                'gate': 'PASS' if not failed else 'INCONCLUSIVE',
                'passed': passed, 'failed': failed, 'runs': runs, 'summaries': summaries,
                'artifacts': artifacts, 'raw_directory': str(raw),
                'source_sha256': hashlib.sha256((HERE / 'native_domain_bench.cc').read_bytes()).hexdigest()}
    (HERE / 'preflight.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'passed': len(passed), 'failed': failed, 'raw': str(raw)}))


if __name__ == '__main__':
    main()
