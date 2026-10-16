"""Finite route-local A/A; existing instrument parser, machine and bias policy."""
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
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from hyle_perf.environment import PERF_ENVIRONMENT_BLOCKLIST, PERF_ENVIRONMENT_FIXED, PERF_ENVIRONMENT_PREFIXES
from hyle_perf.machine import applied_macos_qos, evaluate_machine_profile, get_machine_profile, load_registry, probe_machine
from hyle_perf.nanobench import parse_nanobench, select_pair
from hyle_perf.controls import AA_MAXIMUM_BIAS_PERCENT
from hyle_perf.stats import equivalence_band

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    air = platform.system() == 'Darwin'
    route, key, machine, n = ('Apple', 'apple', 'M1-MAC', 2048) if air else ('CUDA', 'cuda', 'W1-WSL', 6144)
    if air and subprocess.check_output(['/usr/sbin/sysctl', '-n', 'hw.model'], text=True).strip() != 'Mac16,12':
        raise RuntimeError('requires actual MacBook Air')
    gate = evaluate_machine_profile(get_machine_profile(load_registry(), machine),
        probe_machine(paths=[('build', a.build)], expected_qos='interactive' if air else None), role='engineering')
    if gate['gate'] != 'PASS':
        raise RuntimeError('A/A machine gate failed: ' + str(gate['errors']))
    executables = {v: a.build/'bin'/('g07_packed_'+v) for v in ['primary', 'placement']}
    artifacts = {v: digest(f) for v, f in executables.items()}
    if len(set(artifacts.values())) != 2:
        raise RuntimeError('placement did not change artifact')
    specification = Path(__file__).parent/('qualified-sdk-'+key+'.json')
    sdk_identity = json.loads(specification.read_text())
    delivery = (ROOT/sdk_identity['receipt']).parent
    receipt = json.loads((delivery/'receipt.json').read_text())
    if digest(delivery/'raw.zip') != receipt['raw_archive']['sha256']:
        raise RuntimeError('SDK receipt archive changed')
    with zipfile.ZipFile(delivery/'raw.zip') as z:
        sdk = json.loads(z.read(sdk_identity['result_archive_entry']))
    if sdk['candidate'] != sdk_identity['candidate']:
        raise RuntimeError('wrong qualified product candidate')
    cache = dict(x.split('=',1) for x in (a.build/'CMakeCache.txt').read_text().splitlines()
                 if '=' in x and not x.startswith(('#','//')))
    if str(Path(cache['Hyle_DIR:PATH']).parents[2]) != sdk_identity['prefix']:
        raise RuntimeError('wrong qualified installed SDK prefix')
    for row in sdk['owner_artifacts']:
        if digest(Path(row['path'])) != row['sha256']:
            raise RuntimeError('qualified SDK owner changed')
    provenance = a.build/'original669/source-provenance.json'
    original = json.loads(provenance.read_text())
    if original['revision'] != '669121418669b31f6849c74f64362aaf8d6eb919' or original['route'] != key:
        raise RuntimeError('original source pin differs')
    for row in original['sources']:
        if digest(provenance.parent/Path(row['path']).name) != row['generated_sha256']:
            raise RuntimeError('generated original source changed')
    identity = dict(artifacts=artifacts, validator=digest(Path(__file__)), machine=gate['physical_host_identity'],
        original_source=digest(provenance), sdk_record=digest(delivery/'receipt.json'), sdk_identity=digest(specification),
        benchmark_source=digest(Path(__file__).parent/'packed_bench.cc'))
    ready = a.build/'bin/g07_packed_aa.ready'
    ready.write_text('FAIL\n')
    target = a.output/'packed-aa.json'
    if target.exists():
        old = json.loads(target.read_text())
        if old.get('gate') != 'PASS' or old.get('identity') != identity:
            raise RuntimeError('retained A/A failed or identity changed; no automatic repeat')
        if any(digest(Path(row['log'])) != row['sha256'] for row in old['runs']):
            raise RuntimeError('retained A/A raw changed')
        ready.write_text('PASS\n')
        return 0
    raw = a.output/'packed-aa-raw'
    raw.mkdir(parents=True)  # Never overwrite a partial finite qualification.
    def environment():
        e = {k:v for k,v in os.environ.items() if k not in PERF_ENVIRONMENT_BLOCKLIST
             and not any(k.startswith(x) for x in PERF_ENVIRONMENT_PREFIXES)}
        e.update(PERF_ENVIRONMENT_FIXED)
        e.update(HYLE_PERF_RUN_NONCE=uuid.uuid4().hex, HYLE_PERF_APPLE_QOS='interactive')
        return e
    def execute(exe, log, e):
        with log.open('w') as stream:
            r = subprocess.run([str(exe)], env=e, stdout=stream, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError('numerical/A-A process failed: '+str(log))
        text = log.read_text()
        expected = [f'route={route} original={old} M={m} ' for m in [1,64] for old in [0,1]]
        # A/A route duplicates the same arm, so verify four numerical records
        # and their actual route/shape instead of requiring one of each arm.
        records = [x for x in text.splitlines() if x.startswith('# G07_ORACLE ')]
        if len(records) != 4 or any(sum(f'route={route} ' in x and f'M={m} ' in x for x in records) != 2 for m in [1,64]):
            raise RuntimeError('missing independent numerical preflight')
        return text
    for variant, exe in executables.items():
        e = environment();e['HYLE_BENCH_VERIFY_ONLY']='1'
        execute(exe, raw/('preflight-'+variant+'.log'), e)
    runs = []
    for arm in ['native','selected']:
        for variant, exe in executables.items():
            for execution in [False, True]:
                for allocation in [False, True]:
                    e = environment();e['HYLE_BENCH_AA_ROUTE']=arm
                    if execution:e['HYLE_BENCH_RIVAL_FIRST']='1'
                    if allocation:e['HYLE_BENCH_PEER_ALLOCATION_FIRST']='1'
                    log = raw/f'{arm}-{variant}-{int(execution)}-{int(allocation)}.log'
                    text = execute(exe, log, e)
                    markers = [x for x in text.splitlines() if x.startswith('# HYLE_PACKED_RUN ')]
                    if len(markers) != 1:
                        raise RuntimeError('missing/duplicate telemetry')
                    actual = dict(re.findall(r'(\w+)=(.*?)(?= \w+=|$)', markers[0].split(' ',2)[2]))
                    expected = dict(run_nonce=e['HYLE_PERF_RUN_NONCE'], route=route, verify_only='0', aa_route=arm,
                        execution_order='peer-first' if execution else 'candidate-first',
                        allocation_order='peer-first' if allocation else 'candidate-first',
                        placement='1' if variant=='placement' else '0', paired_rounds='16' if air else '104',
                        apple_qos='user-interactive' if air else 'not-applicable', qos_ok='1' if air else 'not-applicable')
                    if any(actual.get(k)!=v for k,v in expected.items()):
                        raise RuntimeError('A/A policy mismatch')
                    pairs = parse_nanobench(text);ratios = {}
                    for m in [1,64]:
                        title = f'{route} resident packed M={m} K=2048 N={n}'
                        c,p = select_pair(pairs, f'HYLE {route} resident packed', f'Original669 {route} packed', title=title)
                        ratios[title] = c.operations_per_second/p.operations_per_second
                    runs.append(dict(arm=arm, variant=variant, execution_order=expected['execution_order'],
                        allocation_order=expected['allocation_order'], ratios=ratios, log=str(log.resolve()), sha256=digest(log)))
                    print(json.dumps(dict(completed=len(runs), total=16)), flush=True)
    summaries, errors = {}, []
    narrow, gross = equivalence_band(AA_MAXIMUM_BIAS_PERCENT), equivalence_band(10)
    for arm in ['native','selected']:
        matching = [r for r in runs if r['arm']==arm]
        for title in matching[0]['ratios']:
            values = [r['ratios'][title] for r in matching];medians = {'all':statistics.median(values)}
            for factor in ['variant','execution_order','allocation_order']:
                for level in sorted({r[factor] for r in matching}):
                    medians[factor+'='+level] = statistics.median(r['ratios'][title] for r in matching if r[factor]==level)
            if any(not gross[0]<=x<=gross[1] for x in values):errors.append(arm+title+': gross instability')
            if any(not narrow[0]<=x<=narrow[1] for x in medians.values()):errors.append(arm+title+': factor median bias')
            summaries[arm+'/'+title] = dict(ratios=values, factor_medians=medians)
    for row in sdk['owner_artifacts']:
        if digest(Path(row['path']))!=row['sha256']:raise RuntimeError('SDK changed during A/A')
    result = dict(gate='FAIL' if errors else 'PASS', identity=identity, runs=runs, summaries=summaries, errors=errors)
    target.write_text(json.dumps(result,indent=2)+'\n')
    if not errors:ready.write_text('PASS\n')
    print(json.dumps(dict(gate=result['gate'], errors=errors)))
    return 2 if errors else 0

if __name__ == '__main__':
    with applied_macos_qos('interactive' if platform.system()=='Darwin' else None):
        raise SystemExit(main())
