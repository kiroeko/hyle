"""D5's fixed CUDA delivery batch; not a general validation framework.

Run from the canonical checkout in a VS x64 environment. Stages are shared,
source, static (in that order). Product and test sources are immutable inputs.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['shared', 'source', 'static'])
parser.add_argument('--resume-shared-consumer', action='store_true')
args = parser.parse_args()
repo = Path.cwd().resolve()
root = Path('C:/hv/d5cd')
out = root / 'receipts' / (args.stage+'-corrected' if args.resume_shared_consumer else args.stage)
out.mkdir(parents=True, exist_ok=True)
flags = {'creationflags': subprocess.CREATE_NO_WINDOW}
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
candidate = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
    raise SystemExit('Freeze the candidate before delivery.')
receipt = {'candidate': candidate, 'stage': args.stage, 'steps': []}
if args.resume_shared_consumer:
    if args.stage != 'shared':
        raise SystemExit('Only the recorded shared consumer correction can resume.')
    old = json.loads((root/'receipts/shared/receipt.json').read_text())
    changed = subprocess.check_output(['git', 'diff', '--name-only', old['candidate'], candidate], text=True).splitlines()
    if any(not p.startswith(('tests/', 'docs/', 'evidence/')) for p in changed):
        raise SystemExit('Product identity changed; existing SDK cannot be reused.')
    receipt['producer_candidate'] = old['candidate']
    receipt['producer_reused_reason'] = 'Only tests/docs/evidence changed; product and vendor inputs identical.'

def run(label, command, env=None):
    command = list(map(str, command))
    start = time.monotonic()
    with (out / (label + '.log')).open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, **flags)
    receipt['steps'].append({'label': label, 'argv': command,
        'returncode': result.returncode, 'seconds': round(time.monotonic()-start, 3)})
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(label, result.returncode, receipt['steps'][-1]['seconds'], flush=True)
    if result.returncode:
        print((out / (label + '.log')).read_text(errors='replace')[-8000:], flush=True)
        raise SystemExit(result.returncode)

consumer = root / 'consumer'
sources = [
    'tests/compute_route_cuda/d5_qwen35_execution_test.cc',
    'tests/compute_route_cuda/d5_qwen38_official_test.cc',
    'tests/graph_route_cuda/d5_qwen35_profile_test.cc',
    'tests/graph_route_cuda/d5_qwen38_official_test.cc',
]
copied = {}
todo = [repo / name for name in sources]
while todo:
    source = todo.pop().resolve()
    relative = source.relative_to(repo).as_posix()
    if relative in copied:
        continue
    if not relative.startswith('tests/'):
        raise SystemExit('External consumer reached non-public product source: '+relative)
    data = source.read_bytes()
    target = consumer / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    copied[relative] = hashlib.sha256(data).hexdigest()
    for name in re.findall(rb'^\s*#include\s+"([^"]+)"', data, re.M):
        name = name.decode()
        if not name.startswith('hyle/'):
            todo.append(source.parent / name)
shutil.copyfile(Path(__file__).parent / 'consumer/CMakeLists.txt', consumer / 'CMakeLists.txt')
(root / 'consumer-inputs.json').write_text(json.dumps({'candidate': candidate, 'files': copied}, indent=2)+'\n')
receipt['consumer_files'] = len(copied)
cache = Path('C:/Users/kiro/AppData/Local/hyle/dependencies/v1/sources')
dep_args = ['-DD5_GGML_SOURCE='+str(cache/'ggml-456172ec733a-41b7e10fd424'),
            '-DD5_GTEST_SOURCE='+str(cache/'googletest-063de7e9578f-162765170ae6')]
cmake, ctest = shutil.which('cmake'), shutil.which('ctest')
common = ['-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=89-real']

if args.stage == 'shared':
    producer = repo / '.scratch/b-cuda-d5'
    staging = root / 'shared-staging'
    prefix = root / 'shared-relocated'
if args.stage == 'shared' and not args.resume_shared_consumer:
    (root / 'shared-original-CMakeCache.txt').write_bytes((producer/'CMakeCache.txt').read_bytes())
    run('producer-configure', [cmake, '-S', repo, '-B', producer, '-U', 'HYLE_DEV_FACTORS',
        '-DHYLE_COMPONENTS=compute,graph,simd', '-DHYLE_BUILD_TESTS=OFF', '-DHYLE_BUILD_BENCH=OFF',
        '-DHYLE_TEST_INSTRUMENTATION=OFF', '-DHYLE_INSTALL=ON', '-DHYLE_STATIC=OFF',
        '-DHYLE_COMPUTE_CPU=OFF', '-DHYLE_COMPUTE_CUDA=ON', '-DHYLE_COMPUTE_APPLE=OFF',
        '-DHYLE_COMPUTE_VULKAN=OFF', '-DHYLE_COMPUTE_ONEDNN=OFF', '-DHYLE_WERROR=ON',
        '-DCMAKE_INSTALL_PREFIX='+str(staging), '-DCMAKE_CUDA_ARCHITECTURES=89-real'])
    run('producer-build', [cmake, '--build', producer, '-j', '2'])
elif args.stage == 'source':
    source_tree = root / 'source-tree'
    if not source_tree.exists():
        archive = root / 'candidate.tar'
        run('archive', ['git', 'archive', '--format=tar', '--output='+str(archive), candidate])
        source_tree.mkdir()
        run('extract', ['tar', '-xf', archive, '-C', source_tree])
        archive.unlink()
        (root/'source-candidate.txt').write_text(candidate+'\n')
    if (root/'source-candidate.txt').read_text().strip() != candidate:
        raise SystemExit('Source tree has a different immutable candidate.')
    producer = root / 'source-build'
    staging = root / 'static-staging'
    run('source-configure', [cmake, '-S', consumer, '-B', producer, *common, *dep_args,
        '-DD5_SOURCE_ROOT='+str(source_tree), '-DCMAKE_INSTALL_PREFIX='+str(staging), '-DHYLE_WERROR=ON'])
    run('source-build', [cmake, '--build', producer, '--target', 'd5_cuda_delivery', '-j', '2'])
    build = producer
elif args.stage == 'static':
    producer = root / 'source-build'
    staging = root / 'static-staging'
    prefix = root / 'static-relocated'
    if (root/'source-candidate.txt').read_text().strip() != candidate:
        raise SystemExit('Static producer is not this candidate.')

if args.stage != 'source' and not args.resume_shared_consumer:
    run('install', [cmake, '--install', producer, '--prefix', staging])
    if prefix.exists():
        raise SystemExit('Relocated prefix already exists; inspect the prior run instead of replacing it.')
    staging.rename(prefix)
if args.stage != 'source':
    build = root / (args.stage+'-consumer')
    run('consumer-configure', [cmake, '-S', consumer, '-B', build, *common, *dep_args,
        '-DCMAKE_PREFIX_PATH='+str(prefix)])
    run('consumer-build', [cmake, '--build', build, '--target', 'd5_cuda_delivery', '-j', '2'])

# The executable must find every non-system DLL next to itself. No producer or
# CUDA toolkit PATH can silently compensate for an incomplete package.
runtime_env = dict(os.environ)
runtime_env['HYLE_D5_QWEN35_FIXTURE'] = str(repo / '.scratch/d5-qwen35-six-profile-fixture')
runtime_env['PATH'] = os.pathsep.join([os.environ['SystemRoot']+'/System32', os.environ['SystemRoot']])
runtime_env.pop('GTEST_FILTER', None)
hidden = []
try:
    if args.stage != 'source':
        for directory in [producer/'bin', producer/'hyle/bin']:
            if directory.exists():
                destination = directory.with_name(directory.name+'.d5-hidden')
                if destination.exists():
                    raise SystemExit('Unexpected existing hidden producer directory.')
                directory.rename(destination)
                hidden.append((destination, directory))
    run('consumer-test', [ctest, '--test-dir', build, '--output-on-failure',
        '--no-tests=error', '-j', '1', '--output-junit', out/'tests.xml'], runtime_env)
finally:
    for source, destination in reversed(hidden):
        source.rename(destination)

native = (build/'Testing/Temporary/LastTest.log').read_bytes()
(out/'native-tests.log').write_bytes(native)
if b'[  SKIPPED ]' in native or b'[  PASSED  ] 8 tests.' not in native:
    raise SystemExit('Expected eight executed, passing tests with no skips.')
receipt['verdict'] = 'PASS'
receipt['tests'] = 8
receipt['product_provider_fallback'] = False
receipt['runtime_path'] = runtime_env['PATH']
(out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(args.stage, 'PASS, 8 tests, no skips', flush=True)
