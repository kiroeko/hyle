"""V4 fixed public Vulkan consumer batch; reuse one shared producer and one source/static producer."""
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
parser.add_argument('--reuse-shared-sdk', action='store_true')
args = parser.parse_args()
if args.reuse_shared_sdk and args.stage != 'shared':
    parser.error('--reuse-shared-sdk applies only to shared consumers')
repo = Path.cwd().resolve()
root = Path('C:/hv/v4w').resolve()
source_tree = root/'source-tree'
product = '2f78714c8'
flags = {'creationflags': subprocess.CREATE_NO_WINDOW}
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
os.environ['VSLANG'] = '1033'
# Import the supported local MSVC toolchain in a hidden process once.
vs = Path('C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat')
if not vs.is_file(): raise SystemExit('Supported VS toolchain is absent')
# cmd.exe uses its own quoting; Python's list-to-command-line escapes are for
# CRT parsers and turn this known batch filename into literal backslash quotes.
vc_env = subprocess.check_output('"' + os.environ['ComSpec'] +
    '" /d /s /c "call "' + str(vs) + '" -arch=x64 >nul && set"', text=True, **flags)
for line in vc_env.splitlines():
    name, separator, value = line.partition('=')
    if separator and name:
        if name.lower() == 'path' and '\\VC\\Tools\\MSVC\\' not in value: continue
        os.environ[name] = value
os.environ['HYLE_DEP_SOURCE_CACHE'] = 'C:/public/tools/hyle-dependencies/v1'

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
        print('Failure log:', out/(label+'.log'), flush=True)
        raise SystemExit(result.returncode)

def rename_inside(source, destination, boundary):
    source, destination, boundary = source.resolve(), destination.resolve(), boundary.resolve()
    if not source.is_relative_to(boundary) or not destination.is_relative_to(boundary):
        raise RuntimeError('Directory move escaped the declared validation root.')
    if source == boundary or destination == boundary or destination.exists():
        raise RuntimeError('Invalid directory move or existing destination.')
    source.rename(destination)

# Reuse the existing immutable product archive, not its CUDA build. Verify both
# committed source identity and the actual archive bytes before compiling Vulkan.
product_paths = ['include', 'src', 'cmake', 'third_party', 'scripts', 'CMakeLists.txt', 'CMakePresets.json', 'LICENSE', 'THIRD_PARTY_NOTICES.md']
product = subprocess.check_output(['git', 'rev-parse', product], text=True, **flags).strip()
receipt['product_candidate'] = product
changed = subprocess.check_output(['git', 'diff', '--name-only', product, candidate, '--', *product_paths], **flags)
if changed.strip():
    raise SystemExit('Product changed; do not reuse the old source snapshot: '+changed.decode())
if not source_tree.exists():
    source_tree.mkdir(parents=True)
    archive = root/'product.tar'
    subprocess.run(['git', 'archive', '--format=tar', '--output='+str(archive), candidate, '--', *product_paths], check=True, **flags)
    subprocess.run(['tar', '-xf', str(archive), '-C', str(source_tree)], check=True, **flags)
    archive.unlink()
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
sources = [f'tests/{entry}_route_vulkan/d5_qwen35_execution_test.cc' for entry in ('compute', 'graph')]
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
        # The external-consumer preprocessor branch omits this private owner-only assertion header.
        if not name.startswith('hyle/') and name != 'vulkan_program_internal.h':
            todo.append(source.parent/name)
cmake_input = Path(__file__).parent/'consumer/CMakeLists.txt'
shutil.copyfile(cmake_input, consumer/'CMakeLists.txt')
copied['CMakeLists.txt'] = hashlib.sha256(cmake_input.read_bytes()).hexdigest()
# Copy only the original public IQ1 equation/batch; private feature mutation
# and full-bank diagnostics are not installed-consumer requirements here.
iq1_source = repo/'tests/compute_route_vulkan/iq1_test_support.h'
iq1 = iq1_source.read_text(encoding='utf-8').split('template <class Prepare> void domains(Prepare prepare) {', 1)[0]
iq1 = iq1.replace('#include "../../src/compute_route_vulkan/vulkan_iq1.h"\n', '')
iq1 += '} // namespace vulkan_iq1_test\n'
iq1_name = 'tests/compute_route_vulkan/iq1_public_support.h'
(consumer/iq1_name).write_text(iq1, encoding='utf-8', newline='\n')
for name, source in [('tests/compute_route_vulkan/iq1_original_grid.h', repo/'tests/compute_route_vulkan/iq1_original_grid.h'),
                     ('iq1_delta.cc', Path(__file__).parent/'consumer/iq1_delta.cc')]:
    shutil.copyfile(source, consumer/name)
    copied[name] = hashlib.sha256(source.read_bytes()).hexdigest()
copied[iq1_name] = {'origin_sha256': hashlib.sha256(iq1_source.read_bytes()).hexdigest(),
                    'public_prefix_sha256': hashlib.sha256(iq1.encode()).hexdigest()}
(out/'consumer-inputs.json').write_text(json.dumps(copied, indent=2)+'\n')

cache = Path('C:/public/tools/hyle-dependencies/v1/sources')
dep_args = ['-DD5_GGML_SOURCE='+str(cache/'ggml-456172ec733a-41b7e10fd424'),
            '-DD5_GTEST_SOURCE='+str(cache/'googletest-063de7e9578f-162765170ae6')]
cmake, ctest = shutil.which('cmake'), shutil.which('ctest')
common = ['-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
source_options = ['-DHYLE_DEP_SOURCE_CACHE=C:/public/tools/hyle-dependencies/v1',
    '-DHYLE_VULKAN_GLSLANG=C:/public/tools/glslang/glslangValidator.exe',
    '-DVulkan_GLSLANG_VALIDATOR_EXECUTABLE=C:/public/tools/glslang/glslangValidator.exe',
    '-DVulkan_GLSLC_EXECUTABLE=C:/public/tools/hyle-vulkan/1.4.357.0/Bin/glslc.exe',
    '-DVulkan_INCLUDE_DIR=C:/public/tools/hyle-vulkan/1.4.357.0/include',
    '-DVulkan_LIBRARY=C:/public/tools/hyle-vulkan/1.4.357.0/Lib/vulkan-1.lib']
if args.stage == 'shared':
    producer = Path('C:/hv/vp1')
    staging, prefix = root/'shared-staging', root/'shared-relocated'
    if args.reuse_shared_sdk:
        original = json.loads((root/'receipts/shared-w1/receipt.json').read_text())
        assert original['product_candidate'] == product and prefix.is_dir()
        for label in ('producer-configure', 'producer-build', 'install'):
            assert any(step['label'] == label and step['returncode'] == 0
                       for step in original['steps'])
        receipt['reused_sdk_candidate'] = original['candidate']
        receipt['reused_sdk_receipt'] = 'receipts/shared-w1/receipt.json'
if args.stage == 'shared' and not args.reuse_shared_sdk:
    (out/'producer-original-cache.txt').write_bytes((producer/'CMakeCache.txt').read_bytes())
    run('producer-configure', [cmake, '-S', repo, '-B', producer, '-U', 'HYLE_DEV_FACTORS',
        '-DHYLE_COMPONENTS=compute,graph,simd', '-DHYLE_BUILD_TESTS=OFF', '-DHYLE_BUILD_BENCH=OFF',
        '-DHYLE_TEST_INSTRUMENTATION=OFF', '-DHYLE_INSTALL=ON', '-DHYLE_STATIC=OFF',
        '-DHYLE_COMPUTE_CPU=OFF', '-DHYLE_COMPUTE_ONEDNN=OFF', '-DHYLE_COMPUTE_CUDA=OFF',
        '-DHYLE_COMPUTE_APPLE=OFF', '-DHYLE_COMPUTE_VULKAN=ON', '-DHYLE_WERROR=ON',
        '-DCMAKE_INSTALL_PREFIX='+str(staging)])
    run('producer-build', [cmake, '--build', producer, '-j', '2'])
elif args.stage != 'shared':
    producer = root/'source-build'
    staging, prefix = root/'static-staging', root/'static-relocated'
    if args.stage == 'source':
        run('source-configure', [cmake, '-S', consumer, '-B', producer, *common, *dep_args,
            *source_options, '-DD5_SOURCE_ROOT='+str(source_tree), '-DCMAKE_INSTALL_PREFIX='+str(staging), '-DHYLE_WERROR=ON'])
        # The static SDK installs every configured public component, including
        # base libraries not reached by the representative consumer targets.
        run('source-build', [cmake, '--build', producer, '-j', '2'])
        (root/'source-candidate.txt').write_text(candidate+'\n')
        build = producer
    elif (root/'source-candidate.txt').read_text().strip() != candidate:
        raise SystemExit('Static producer is not this consumer candidate.')

if args.stage != 'source':
    if not args.reuse_shared_sdk:
        run('install', [cmake, '--install', producer, '--prefix', staging])
        rename_inside(staging, prefix, root)
    build = root/(args.stage+'-consumer')
    run('consumer-configure', [cmake, '-S', consumer, '-B', build, *common, *dep_args,
        '-DCMAKE_PREFIX_PATH='+str(prefix)])
    run('consumer-build', [cmake, '--build', build, '--target', 'd5_vulkan_delivery', '-j', '2'])

commands = subprocess.check_output([shutil.which('ninja'), '-C', str(build), '-t', 'commands', 'd5_vulkan_delivery'], **flags)
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
runtime_env['HYLE_VULKAN_VALIDATION'] = '1'
runtime_env['VK_LAYER_PATH'] = 'C:/public/tools/hyle-vulkan/1.4.357.0/Bin'
hidden = []
try:
    if args.stage != 'source':
        for directory in (producer/'bin', producer/'hyle/bin'):
            if directory.exists():
                destination = directory.with_name(directory.name+'.v4-hidden')
                rename_inside(directory, destination, producer)
                hidden.append((destination, directory))
    run('consumer-test', [ctest, '--test-dir', build, '--output-on-failure', '--no-tests=error',
        '-j', '1', '--output-junit', out/'tests.xml'], runtime_env)
finally:
    for source, destination in reversed(hidden):
        rename_inside(source, destination, producer)
    native_path = build/'Testing/Temporary/LastTest.log'
    if native_path.exists():
        (out/'native-tests.log').write_bytes(native_path.read_bytes())
native = (build/'Testing/Temporary/LastTest.log').read_bytes()
(out/'native-tests.log').write_bytes(native)
passed = sum(map(int, re.findall(rb'\[  PASSED  \]\s+(\d+) tests?\.', native)))
if b'[  SKIPPED ]' in native or passed != 14:
    raise SystemExit('Expected fourteen executed passing tests, without skips.')
receipt.update(verdict='PASS', tests=14, runtime_path=runtime_env['PATH'])
(out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(args.stage, 'PASS, 14 tests, no skips', flush=True)
