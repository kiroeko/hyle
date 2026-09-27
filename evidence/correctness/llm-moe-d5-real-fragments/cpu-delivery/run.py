"""Fixed D5 CPU source/static/shared consumer batch; run in a VS x64 environment."""
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

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['shared', 'source', 'static'])
parser.add_argument('--attempt', default='')
args = parser.parse_args()
repo = Path.cwd().resolve()
root = Path('C:/hv/d5cp').resolve()
source_tree = Path('C:/hv/d5cd/source-tree').resolve()
product = '9273bc71017a1febb69db240b6ea8970b88dd77c'
flags = {'creationflags': subprocess.CREATE_NO_WINDOW}
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
os.environ['VSLANG'] = '1033'
if args.attempt and not re.fullmatch(r'[a-z0-9-]+', args.attempt):
    raise SystemExit('Use a simple lowercase attempt name.')
out = root/'receipts'/(args.stage + ('-'+args.attempt if args.attempt else ''))
if out.exists():
    raise SystemExit('Retain previous receipts; inspect before using another attempt.')
out.mkdir(parents=True)
candidate = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, **flags).strip()
if subprocess.check_output(['git', 'status', '--porcelain'], text=True, **flags).strip():
    raise SystemExit('Freeze the candidate before delivery.')
receipt = {'candidate': candidate, 'product_candidate': product, 'stage': args.stage, 'steps': []}

def run(label, command, env=None):
    command = list(map(str, command))
    start = time.monotonic()
    with (out/(label+'.log')).open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, **flags)
    receipt['steps'].append({'label': label, 'argv': command,
        'returncode': result.returncode, 'seconds': round(time.monotonic()-start, 3)})
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(label, result.returncode, receipt['steps'][-1]['seconds'], flush=True)
    if result.returncode:
        print((out/(label+'.log')).read_text(errors='replace')[-8000:], flush=True)
        raise SystemExit(result.returncode)

def rename_inside(source, destination, boundary):
    source, destination, boundary = source.resolve(), destination.resolve(), boundary.resolve()
    if not source.is_relative_to(boundary) or not destination.is_relative_to(boundary):
        raise RuntimeError('Directory move escaped the declared validation root.')
    if source == boundary or destination == boundary or destination.exists():
        raise RuntimeError('Invalid directory move or existing destination.')
    source.rename(destination)

# Reuse the existing immutable product archive, not its CUDA build. Verify both
# committed source identity and the actual archive bytes before compiling CPU.
product_paths = ['include', 'src', 'cmake', 'third_party', 'CMakeLists.txt']
changed = subprocess.check_output(['git', 'diff', '--name-only', product, candidate, '--', *product_paths], **flags)
if changed.strip():
    raise SystemExit('Product changed; do not reuse the old source snapshot: '+changed.decode())
listing = subprocess.check_output(['git', 'ls-tree', '-r', product, '--', *product_paths], text=True, **flags)
verified = {}
for line in listing.splitlines():
    meta, name = line.split('\t', 1)
    mode, kind, oid = meta.split()
    if kind != 'blob' or mode not in ('100644', '100755'):
        raise SystemExit('Unexpected archive entry: '+name)
    data = (source_tree/name).read_bytes()
    actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if actual != oid:
        raise SystemExit('Archive bytes do not match the frozen product: '+name)
    verified[name] = oid
(out/'source-identity.json').write_text(json.dumps({'product': product, 'candidate': candidate,
    'archive': str(source_tree), 'git_blobs': verified}, indent=2)+'\n')

consumer = root/'consumer'
sources = [f'tests/{entry}_route_cpu/{name}.cc' for entry in ('compute', 'graph')
           for name in ('d5_qwen35_profile_test', 'd5_qwen38_official_test', 'd5_qwen_gguf_vocabulary_test')]
copied, todo = {}, [repo/name for name in sources]
while todo:
    source = todo.pop().resolve()
    relative = source.relative_to(repo).as_posix()
    if relative in copied:
        continue
    if not relative.startswith('tests/'):
        raise SystemExit('Fixture reached private product source: '+relative)
    data = source.read_bytes()
    target = consumer/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    copied[relative] = hashlib.sha256(data).hexdigest()
    for name in re.findall(rb'^\s*#include\s+"([^"]+)"', data, re.M):
        name = name.decode()
        if not name.startswith('hyle/'):
            todo.append(source.parent/name)
cmake_input = Path(__file__).parent/'consumer/CMakeLists.txt'
shutil.copyfile(cmake_input, consumer/'CMakeLists.txt')
copied['CMakeLists.txt'] = hashlib.sha256(cmake_input.read_bytes()).hexdigest()
(out/'consumer-inputs.json').write_text(json.dumps(copied, indent=2)+'\n')

cache = Path('C:/Users/kiro/AppData/Local/hyle/dependencies/v1/sources')
dep_args = ['-DD5_GGML_SOURCE='+str(cache/'ggml-456172ec733a-41b7e10fd424'),
            '-DD5_GTEST_SOURCE='+str(cache/'googletest-063de7e9578f-162765170ae6')]
cmake, ctest = shutil.which('cmake'), shutil.which('ctest')
common = ['-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
if args.stage == 'shared':
    producer = repo/'.scratch/build-compute-v2-dev'
    staging, prefix = root/'shared-staging', root/'shared-relocated'
    (out/'producer-original-cache.txt').write_bytes((producer/'CMakeCache.txt').read_bytes())
    run('producer-configure', [cmake, '-S', repo, '-B', producer, '-U', 'HYLE_DEV_FACTORS',
        '-DHYLE_COMPONENTS=compute,graph,simd', '-DHYLE_BUILD_TESTS=OFF', '-DHYLE_BUILD_BENCH=OFF',
        '-DHYLE_TEST_INSTRUMENTATION=OFF', '-DHYLE_INSTALL=ON', '-DHYLE_STATIC=OFF',
        '-DHYLE_COMPUTE_CPU=ON', '-DHYLE_COMPUTE_ONEDNN=ON', '-DHYLE_COMPUTE_CUDA=OFF',
        '-DHYLE_COMPUTE_APPLE=OFF', '-DHYLE_COMPUTE_VULKAN=OFF', '-DHYLE_WERROR=ON',
        '-DCMAKE_INSTALL_PREFIX='+str(staging)])
    run('producer-build', [cmake, '--build', producer, '-j', '2'])
else:
    producer = root/'source-build'
    staging, prefix = root/'static-staging', root/'static-relocated'
    if args.stage == 'source':
        run('source-configure', [cmake, '-S', consumer, '-B', producer, *common, *dep_args,
            '-DD5_SOURCE_ROOT='+str(source_tree), '-DCMAKE_INSTALL_PREFIX='+str(staging), '-DHYLE_WERROR=ON'])
        # The static SDK installs every configured public component, including
        # base libraries not reached by the representative consumer targets.
        run('source-build', [cmake, '--build', producer, '-j', '2'])
        (root/'source-candidate.txt').write_text(candidate+'\n')
        build = producer
    elif (root/'source-candidate.txt').read_text().strip() != candidate:
        raise SystemExit('Static producer is not this consumer candidate.')

if args.stage != 'source':
    run('install', [cmake, '--install', producer, '--prefix', staging])
    rename_inside(staging, prefix, root)
    build = root/(args.stage+'-consumer')
    run('consumer-configure', [cmake, '-S', consumer, '-B', build, *common, *dep_args,
        '-DCMAKE_PREFIX_PATH='+str(prefix)])
    run('consumer-build', [cmake, '--build', build, '--target', 'd5_cpu_delivery', '-j', '2'])

commands = subprocess.check_output([shutil.which('ninja'), '-C', str(build), '-t', 'commands', 'd5_cpu_delivery'], **flags)
(out/'compile-link-commands.log').write_bytes(commands)
if args.stage != 'source':
    normalized = commands.decode(errors='replace').replace('\\', '/').lower()
    for forbidden in (str(repo/'include'), str(repo/'src'), str(producer), str(source_tree)):
        if forbidden.replace('\\', '/').lower() in normalized:
            raise SystemExit('Installed consumer depends on producer source/build: '+forbidden)
    origins = {}
    for path in build.glob('*.dll'):
        matches = list(prefix.rglob(path.name))
        if len(matches) != 1:
            raise SystemExit('DLL has no unique relocated SDK origin: '+path.name)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashlib.sha256(matches[0].read_bytes()).hexdigest() != sha:
            raise SystemExit('Staged DLL differs from SDK: '+path.name)
        origins[path.name] = {'sha256': sha, 'sdk_path': str(matches[0].relative_to(prefix))}
    (out/'dll-origins.json').write_text(json.dumps(origins, indent=2)+'\n')

runtime_env = dict(os.environ)
runtime_env['HYLE_D5_QWEN35_FIXTURE'] = str(repo/'.scratch/d5-qwen35-six-profile-fixture')
runtime_env['PATH'] = os.pathsep.join([os.environ['SystemRoot']+'/System32', os.environ['SystemRoot']])
runtime_env.pop('GTEST_FILTER', None)
hidden = []
try:
    if args.stage != 'source':
        for directory in (producer/'bin', producer/'hyle/bin'):
            if directory.exists():
                destination = directory.with_name(directory.name+'.d5-hidden')
                rename_inside(directory, destination, producer)
                hidden.append((destination, directory))
    run('consumer-test', [ctest, '--test-dir', build, '--output-on-failure', '--no-tests=error',
        '-j', '1', '--output-junit', out/'tests.xml'], runtime_env)
finally:
    for source, destination in reversed(hidden):
        rename_inside(source, destination, producer)
native = (build/'Testing/Temporary/LastTest.log').read_bytes()
(out/'native-tests.log').write_bytes(native)
passed = sum(map(int, re.findall(rb'\[  PASSED  \]\s+(\d+) tests?\.', native)))
if b'[  SKIPPED ]' in native or passed != 8:
    raise SystemExit('Expected eight executed passing tests, without skips.')
receipt.update(verdict='PASS', tests=8, runtime_path=runtime_env['PATH'])
(out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(args.stage, 'PASS, 8 tests, no skips', flush=True)
