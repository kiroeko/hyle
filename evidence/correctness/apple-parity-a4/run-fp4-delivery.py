import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

repo = Path('/Users/kiro/hyle')
out = Path('/Users/kiro/hyle-scratch/apple-parity-delivery')
raw = Path('/Users/kiro/hyle-scratch/apple-parity')
shared = Path('/Users/kiro/hyle-scratch/d5-graph-825c41f8')
source = Path('/Users/kiro/hyle-scratch/d5-apple-delivery/source-build')
consumer = Path('/Users/kiro/hyle-scratch/d5-apple-delivery/consumer')
candidate = sys.argv[1]
if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
    raise RuntimeError('unexpected changes in validation copy')
subprocess.run(['git', '-C', str(repo), 'fetch', '-q', str(raw / 'fp4-delivery.bundle'), 'HEAD'], check=True)
subprocess.run(['git', '-C', str(repo), 'reset', '--hard', '-q', candidate], check=True)
env = dict(os.environ)
env['PATH'] = '/Users/kiro/hyle-verify-tools/metal-17F109/shim:/opt/homebrew/bin:' + env['PATH']
name = 'fp4-delivery-' + candidate[:8]
resume = len(sys.argv) > 2 and sys.argv[2] == 'resume-license-check'
receipt_path = out / (name + '-receipt.json')
receipt = json.loads(receipt_path.read_text()) if resume else {'candidate': candidate, 'steps': [], 'sdks': {}}
if receipt['candidate'] != candidate: raise RuntimeError('delivery identity changed')
def run(label, argv, runenv=env):
    print('START ' + label, flush=True)
    start = time.monotonic()
    log = out / (name + '-' + label + '.log')
    with log.open('w') as file:
        code = subprocess.run(argv, cwd=str(repo), env=runenv, stdout=file, stderr=subprocess.STDOUT).returncode
    receipt['steps'].append({'name': label, 'returncode': code, 'seconds': time.monotonic() - start,
        'log': str(log), 'sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
    (out / (name + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print('END ' + label + ' exit=' + str(code), flush=True)
    if code:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-25:]), flush=True)
        raise SystemExit(code)

cmake = '/opt/homebrew/bin/cmake'
if not resume:
    run('cold-build', [cmake, '--build', str(shared), '--parallel', '2', '--target',
        'hyle_compute_route_apple_test', 'hyle_graph_route_apple_test'])
    for label, target, suite in [('direct', 'compute', 'AppleMoeFragments'),
            ('graph', 'graph', 'AppleGraphMoeFragments')]:
        run('cold-' + label, [str(shared / ('bin/hyle_' + target + '_route_apple_test')),
            '--gtest_filter=' + suite + '.D5OfficialDeepSeekFullMoeColdBoundary',
            '--gtest_output=xml:' + str(out / (name + '-cold-' + label + '.xml'))])
    for file in ['CMakeLists.txt', 'main.cc']:
        shutil.copyfile(repo / 'tests/compute_route_apple/parity_delivery' / file, consumer / file)
    run('shared-configure', [cmake, '-S', str(repo), '-B', str(shared), '-DHYLE_BUILD_TESTS=OFF',
        '-DHYLE_INSTALL=ON', '-DHYLE_TEST_INSTRUMENTATION=OFF', '-DHYLE_BUILD_BENCH=OFF'])
    run('shared-build', [cmake, '--build', str(shared), '--parallel', '2', '--target', 'hyle_compute', 'hyle_graph'])
    run('shared-install', [cmake, '--install', str(shared), '--prefix', str(out / 'shared-relocated')])
    run('source-build', [cmake, '--build', str(source), '--parallel', '2', '--target', 'apple_parity_consumer'])
    run('static-install', [cmake, '--install', str(source), '--prefix', str(out / 'static-relocated')])
offline = dict(env)
offline.update(HTTPS_PROXY='http://127.0.0.1:9', HTTP_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9')
for label in ['static', 'shared']:
    prefix, build = out / (label + '-relocated'), out / (label + '-consumer')
    if not resume or label != 'static':
        run(label + '-consumer-configure', [cmake, '-S', str(consumer), '-B', str(build),
            '-DCMAKE_PREFIX_PATH=' + str(prefix), '-DFETCHCONTENT_FULLY_DISCONNECTED=ON'], offline)
        run(label + '-consumer-build', [cmake, '--build', str(build), '--parallel', '2', '--target', 'apple_parity_consumer'])
    names = sorted(p.name for p in prefix.rglob('*') if p.is_file() and 'licenses' in p.parts)
    for dependency in ['apple-ds4_quantization-LICENSE', 'apple-deepseek_quantization-LICENSE',
            'apple-deepseek-TILELANG-LICENSE', 'apple-deepseek-CUTLASS-LICENSE']:
        if dependency not in names:
            raise RuntimeError('missing FP4 source dependency license: ' + dependency)
    receipt['sdks'][label] = {'licenses': names}
runtime = dict(env)
runtime['PATH'] = '/usr/bin:/bin'
for key in list(runtime):
    if key.startswith('DYLD_'): runtime.pop(key)
runtime['DYLD_PRINT_LIBRARIES'] = '1'
for label, build, producer in [('source', source, None),
        ('static', out / 'static-consumer', source), ('shared', out / 'shared-consumer', shared)]:
    hidden = []
    try:
        if producer:
            paths = [producer / 'bin']
            paths += [p for p in producer.rglob('mlx.metallib') if producer / 'bin' not in p.parents]
            for original in paths:
                if not original.exists(): continue
                renamed = original.with_name(original.name + '.fp4-delivery-hidden')
                if renamed.exists(): raise RuntimeError('stale hidden asset')
                original.rename(renamed); hidden.append((original, renamed))
        run(label + '-positive', [str(build / 'apple_parity_consumer')], runtime)
        if producer and str(producer) in (out / (name + '-' + label + '-positive.log')).read_text(errors='replace'):
            raise RuntimeError('runtime loaded hidden producer')
    finally:
        for original, renamed in reversed(hidden): renamed.rename(original)
(out / (name + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
print('PASS FP4 cold boundaries, public source/static/shared/offline/relocation and source licenses', flush=True)
