import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

repo = Path('/Users/kiro/hyle')
root = Path('/Users/kiro/hyle-scratch/apple-parity-delivery')
root.mkdir(parents=True, exist_ok=True)
candidate = sys.argv[1]
if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
    raise RuntimeError('validation copy has unexpected changes')
subprocess.run(['git', '-C', str(repo), 'fetch', '-q',
    '/Users/kiro/hyle-scratch/apple-parity/delivery.bundle', 'HEAD'], check=True)
subprocess.run(['git', '-C', str(repo), 'reset', '--hard', '-q', candidate], check=True)
env = dict(os.environ)
env['PATH'] = '/Users/kiro/hyle-verify-tools/metal-17F109/shim:/opt/homebrew/bin:' + env['PATH']
cmake = '/opt/homebrew/bin/cmake'
consumer = Path('/Users/kiro/hyle-scratch/d5-apple-delivery/consumer')
for name in ['CMakeLists.txt', 'main.cc']:
    shutil.copyfile(repo / 'tests/compute_route_apple/parity_delivery' / name, consumer / name)
receipt = {'candidate': candidate, 'steps': [], 'consumer': {
    name: hashlib.sha256((consumer / name).read_bytes()).hexdigest()
    for name in ['CMakeLists.txt', 'main.cc']}}

def run(name, argv, runenv=env):
    print('START ' + name, flush=True)
    start = time.time()
    with (root / (name + '.log')).open('w') as log:
        code = subprocess.run([str(a) for a in argv], cwd=str(repo), env=runenv,
            stdout=log, stderr=subprocess.STDOUT).returncode
    receipt['steps'].append({'name': name, 'returncode': code, 'seconds': time.time() - start})
    (root / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('END ' + name + ' exit=' + str(code), flush=True)
    if code:
        print('\n'.join((root / (name + '.log')).read_text(errors='replace').splitlines()[-35:]), flush=True)
        raise SystemExit(code)

def executable(build):
    for path in [build / 'apple_parity_consumer', build / 'bin/apple_parity_consumer']:
        if path.is_file():
            return path
    raise RuntimeError('consumer executable missing')

def verify(label, build, producer=None, prefix=None):
    commands = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands',
        'apple_parity_consumer'], text=True, env=env)
    (root / (label + '-commands.log')).write_text(commands)
    if prefix:
        for forbidden in [str(repo / 'include'), str(repo / 'src'), str(producer)]:
            if forbidden in commands:
                raise RuntimeError('installed consumer uses producer path: ' + forbidden)
        libraries = list(prefix.rglob('mlx.metallib'))
        if len(libraries) != 1:
            raise RuntimeError('SDK has unexpected Metal resource count')
        staged = executable(build).parent / 'mlx.metallib'
        if hashlib.sha256(staged.read_bytes()).digest() != hashlib.sha256(libraries[0].read_bytes()).digest():
            raise RuntimeError('staged Metal resource differs')
        licenses = sorted(p.name for p in prefix.rglob('*') if p.is_file() and
            any(part == 'licenses' for part in p.parts))
        for library in ['bitsandbytes_apple', 'awq_apple', 'compressed_tensors_apple', 'ggml_q4k_apple']:
            if library + '-LICENSE' not in licenses or library + '-NOTICE.md' not in licenses:
                raise RuntimeError('missing new borrowed license: ' + library)
        receipt[label + '-sdk'] = {'licenses': licenses,
            'metallib_sha256': hashlib.sha256(staged.read_bytes()).hexdigest()}
    runtime = dict(env)
    runtime['PATH'] = '/usr/bin:/bin'
    for name in list(runtime):
        if name.startswith('DYLD_'):
            runtime.pop(name)
    runtime['DYLD_PRINT_LIBRARIES'] = '1'
    hidden = []
    try:
        if producer:
            paths = [producer / 'bin']
            paths += [p for p in producer.rglob('mlx.metallib') if not any(b == producer / 'bin' for b in p.parents)]
            for original in paths:
                if not original.exists():
                    continue
                renamed = original.with_name(original.name + '.apple-parity-hidden')
                if renamed.exists():
                    raise RuntimeError('stale hidden producer asset')
                original.rename(renamed)
                hidden.append((original, renamed))
        run(label + '-runtime', [executable(build)], runtime)
        if producer and str(producer) in (root / (label + '-runtime.log')).read_text(errors='replace'):
            raise RuntimeError('runtime loaded the hidden producer')
    finally:
        for original, renamed in reversed(hidden):
            renamed.rename(original)

common = ['-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_OSX_DEPLOYMENT_TARGET=14.4',
    '-DCMAKE_CXX_COMPILER=/usr/bin/clang++', '-DCMAKE_C_COMPILER=/usr/bin/clang']
runtime_offline = dict(env)
runtime_offline['HTTPS_PROXY'] = 'http://127.0.0.1:9'
runtime_offline['HTTP_PROXY'] = 'http://127.0.0.1:9'
runtime_offline['ALL_PROXY'] = 'http://127.0.0.1:9'
for requested in (sys.argv[2:] or ['shared', 'source', 'static']):
    label = 'shared' if requested == 'shared-consumer' else requested
    if label == 'shared':
        producer = Path('/Users/kiro/hyle-scratch/d5-graph-825c41f8')
        if requested != 'shared-consumer':
            run(label + '-configure', [cmake, '-S', repo, '-B', producer, '-DHYLE_INSTALL=ON',
                '-DHYLE_BUILD_TESTS=OFF', '-DHYLE_TEST_INSTRUMENTATION=OFF', '-DHYLE_BUILD_BENCH=OFF'])
            run(label + '-build', [cmake, '--build', producer, '--parallel', '2', '--target',
                'hyle_compute', 'hyle_graph'])
    else:
        producer = Path('/Users/kiro/hyle-scratch/d5-apple-delivery/source-build')
        if label == 'source':
            run(label + '-configure', [cmake, '-S', consumer, '-B', producer, *common,
                '-DHYLE_SOURCE_ROOT=' + str(repo), '-DHYLE_STATIC=ON'])
            run(label + '-build', [cmake, '--build', producer, '--parallel', '2', '--target',
                'apple_parity_consumer'])
            verify(label, producer)
            continue
    staging, prefix = root / (label + '-staging'), root / (label + '-relocated')
    if prefix.exists():
        if requested != 'shared-consumer':
            raise RuntimeError('refusing to overwrite prior SDK receipt')
    else:
        run(label + '-install', [cmake, '--install', producer, '--prefix', staging])
        staging.rename(prefix)
    build = root / (label + '-consumer')
    run(label + '-consumer-configure', [cmake, '-S', consumer, '-B', build, *common,
        '-DCMAKE_PREFIX_PATH=' + str(prefix), '-DFETCHCONTENT_FULLY_DISCONNECTED=ON'], runtime_offline)
    run(label + '-consumer-build', [cmake, '--build', build, '--parallel', '2', '--target',
        'apple_parity_consumer'])
    verify(label, build, producer, prefix)
print('PASS source/static/shared public delivery sample', flush=True)
