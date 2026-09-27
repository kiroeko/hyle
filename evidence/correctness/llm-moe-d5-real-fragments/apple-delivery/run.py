"""One fixed Apple batch: ordinary boundaries and three public C++ deliveries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['shared', 'source', 'static', 'all'])
parser.add_argument('--attempt', default='')
args = parser.parse_args()
repo = Path.cwd().resolve()
root = Path.home()/'hyle-scratch/d5-apple-delivery'
root.mkdir(parents=True, exist_ok=True)
candidate = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
    raise SystemExit('Freeze the canonical candidate and sync its complete checkout first.')
if args.attempt and not re.fullmatch(r'[a-z0-9-]+', args.attempt):
    raise SystemExit('Use a simple lowercase attempt name.')
consumer = root/'consumer'
sources = [repo/f'tests/{entry}_route_apple/d5_qwen35_execution_test.cc' for entry in ('compute', 'graph')]
copied, todo = {}, sources[:]
while todo:
    source = todo.pop().resolve()
    name = source.relative_to(repo).as_posix()
    if name in copied:
        continue
    if not name.startswith('tests/'):
        raise SystemExit('Consumer fixture reached private product source: '+name)
    data = source.read_bytes()
    target = consumer/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    copied[name] = hashlib.sha256(data).hexdigest()
    for include in re.findall(rb'^\s*#include\s+"([^"]+)"', data, re.M):
        include = include.decode()
        if not include.startswith('hyle/'):
            todo.append(source.parent/include)
cmake_source = Path(__file__).parent/'consumer/CMakeLists.txt'
shutil.copyfile(cmake_source, consumer/'CMakeLists.txt')
copied['CMakeLists.txt'] = hashlib.sha256(cmake_source.read_bytes()).hexdigest()
identity = {'candidate': candidate, 'consumer': copied, 'product_git_blobs': {}}
listing = subprocess.check_output(['git', 'ls-tree', '-r', candidate, '--',
    'include', 'src', 'cmake', 'third_party', 'CMakeLists.txt'], text=True)
for line in listing.splitlines():
    meta, name = line.split('\t', 1)
    mode, kind, oid = meta.split()
    if kind != 'blob' or mode not in ('100644', '100755'):
        raise SystemExit('Unexpected source entry: '+name)
    data = (repo/name).read_bytes()
    if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() != oid:
        raise SystemExit('Product source differs from the immutable candidate: '+name)
    identity['product_git_blobs'][name] = oid
(root/'inputs.json').write_text(json.dumps(identity, indent=2)+'\n')
cache = Path.home()/'Library/Caches/hyle/dependencies/v1/sources'
deps = ['-DD5_GGML_SOURCE='+str(cache/'ggml-456172ec733a-41b7e10fd424'),
        '-DD5_GTEST_SOURCE='+str(cache/'googletest-063de7e9578f-162765170ae6')]
cmake, ctest = shutil.which('cmake'), shutil.which('ctest')
common = ['-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_OSX_DEPLOYMENT_TARGET=14.4',
          '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']

def rename_inside(source, destination, boundary):
    source, destination, boundary = source.resolve(), destination.resolve(), boundary.resolve()
    if not source.is_relative_to(boundary) or not destination.is_relative_to(boundary):
        raise RuntimeError('Move outside the declared validation directory.')
    if source == boundary or destination.exists():
        raise RuntimeError('Invalid move or existing destination.')
    source.rename(destination)

def stage(label):
    out = root/'receipts'/(label + ('-'+args.attempt if args.attempt else ''))
    if out.exists():
        old = json.loads((out/'receipt.json').read_text())
        if old.get('verdict') == 'PASS' and old.get('candidate') == candidate:
            print(label, 'reusing completed immutable receipt', flush=True)
            return
        raise SystemExit('Inspect the existing attempt before restarting: '+str(out))
    out.mkdir(parents=True)
    receipt = {'candidate': candidate, 'stage': label, 'steps': []}
    shutil.copyfile(root/'inputs.json', out/'inputs.json')

    def run(name, command, env=None):
        command = list(map(str, command))
        started = time.monotonic()
        with (out/(name+'.log')).open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        receipt['steps'].append({'label': name, 'argv': command, 'returncode': result.returncode,
                                 'seconds': round(time.monotonic()-started, 3)})
        (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(label, name, result.returncode, receipt['steps'][-1]['seconds'], flush=True)
        if result.returncode:
            print((out/(name+'.log')).read_text(errors='replace')[-9000:], flush=True)
            raise SystemExit(result.returncode)

    if label == 'shared':
        producer = Path.home()/'hyle-scratch/d5-graph-825c41f8'
        staging, prefix = root/'shared-staging', root/'shared-relocated'
        reuse_sdk = prefix.exists()
        if reuse_sdk:
            previous = root/'receipts/shared'
            prior_inputs = json.loads((previous/'inputs.json').read_text())
            prior = json.loads((previous/'receipt.json').read_text())
            if prior_inputs['product_git_blobs'] != identity['product_git_blobs'] or not any(
                step['label'] == 'install' and step['returncode'] == 0 for step in prior['steps']):
                raise SystemExit('Existing SDK has no matching successful producer/install identity.')
            receipt['reused_sdk'] = {'candidate': prior['candidate'], 'receipt': str(previous/'receipt.json')}
        else:
            shutil.copyfile(producer/'CMakeCache.txt', out/'producer-original-cache.txt')
            run('producer-configure', [cmake, '-S', repo, '-B', producer, '-U', 'HYLE_DEV_FACTORS',
                '-DHYLE_COMPONENTS=compute,graph,simd', '-DHYLE_BUILD_TESTS=OFF', '-DHYLE_BUILD_BENCH=OFF',
                '-DHYLE_TEST_INSTRUMENTATION=OFF', '-DHYLE_INSTALL=ON', '-DHYLE_STATIC=OFF',
                '-DHYLE_COMPUTE_CPU=OFF', '-DHYLE_COMPUTE_ONEDNN=OFF', '-DHYLE_COMPUTE_CUDA=OFF',
                '-DHYLE_COMPUTE_APPLE=ON', '-DHYLE_COMPUTE_VULKAN=OFF', '-DHYLE_WERROR=ON',
                '-DCMAKE_INSTALL_PREFIX='+str(staging)])
            run('producer-build', [cmake, '--build', producer, '-j', '2'])
    else:
        reuse_sdk = False
        producer = root/'source-build'
        staging, prefix = root/'static-staging', root/'static-relocated'
        if label == 'source':
            run('source-configure', [cmake, '-S', consumer, '-B', producer, *common, *deps,
                '-DD5_SOURCE_ROOT='+str(repo), '-DCMAKE_INSTALL_PREFIX='+str(staging), '-DHYLE_WERROR=ON'])
            run('source-build', [cmake, '--build', producer, '-j', '2'])
            (root/'source-candidate.txt').write_text(candidate+'\n')
            build = producer
        elif (root/'source-candidate.txt').read_text().strip() != candidate:
            raise SystemExit('Static producer belongs to another candidate.')
    if label != 'source':
        if not reuse_sdk:
            run('install', [cmake, '--install', producer, '--prefix', staging])
            rename_inside(staging, prefix, root)
        build = root/(label+'-consumer')
        run('consumer-configure', [cmake, '-S', consumer, '-B', build, *common, *deps,
            '-DCMAKE_PREFIX_PATH='+str(prefix)])
        run('consumer-build', [cmake, '--build', build, '--target', 'd5_apple_delivery', '-j', '2'])
    commands = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', 'd5_apple_delivery'])
    (out/'compile-link-commands.log').write_bytes(commands)
    if label != 'source':
        for forbidden in (str(repo/'include'), str(repo/'src'), str(producer)):
            if forbidden.encode() in commands:
                raise SystemExit('Installed consumer used a producer path: '+forbidden)
        assets = list(prefix.rglob('mlx.metallib'))
        if len(assets) != 1:
            raise SystemExit('Expected one SDK Metal resource.')
        sha = hashlib.sha256(assets[0].read_bytes()).hexdigest()
        if hashlib.sha256((build/'mlx.metallib').read_bytes()).hexdigest() != sha:
            raise SystemExit('Consumer Metal resource differs from the relocated SDK.')
        (out/'runtime-resource.json').write_text(json.dumps({'sha256': sha,
            'sdk_path': str(assets[0].relative_to(prefix))}, indent=2)+'\n')
    runtime = dict(os.environ)
    runtime['HYLE_D5_QWEN35_FIXTURE'] = str(Path.home()/'hyle-scratch/d5-qwen35-six-profile-fixture')
    runtime['PATH'] = '/usr/bin:/bin'
    for name in list(runtime):
        if name.startswith('DYLD_') or name == 'GTEST_FILTER':
            runtime.pop(name)
    runtime['DYLD_PRINT_LIBRARIES'] = '1'
    hidden = []
    try:
        if label != 'source':
            for directory in (producer/'bin', producer/'hyle/bin'):
                if directory.exists():
                    destination = directory.with_name(directory.name+'.d5-hidden')
                    rename_inside(directory, destination, producer)
                    hidden.append((destination, directory))
            for asset in producer.rglob('mlx.metallib'):
                if any(asset.is_relative_to(directory) for directory, _ in hidden):
                    continue
                destination = asset.with_name(asset.name+'.d5-hidden')
                rename_inside(asset, destination, producer)
                hidden.append((destination, asset))
        run('consumer-test', [ctest, '--test-dir', build, '--output-on-failure', '--no-tests=error',
            '-j', '1', '--output-junit', out/'tests.xml'], runtime)
    finally:
        for source, destination in reversed(hidden):
            rename_inside(source, destination, producer)
    native = (build/'Testing/Temporary/LastTest.log').read_bytes()
    (out/'native-tests.log').write_bytes(native)
    passed = sum(map(int, re.findall(rb'\[  PASSED  \]\s+(\d+) tests?\.', native)))
    if b'[  SKIPPED ]' in native or passed != 12:
        raise SystemExit('Expected twelve executed passing tests, without skips.')
    if label != 'source' and str(producer).encode() in native:
        raise SystemExit('Installed execution loaded from the hidden producer.')
    receipt.update(verdict='PASS', tests=12, runtime_path=runtime['PATH'])
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(label, 'PASS, 12 tests, no skips', flush=True)

for label in (['shared', 'source', 'static'] if args.stage == 'all' else [args.stage]):
    stage(label)
