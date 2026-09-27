"""Windows entry for the fixed, serial Vulkan source/static/shared batch."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
flags = {'creationflags': subprocess.CREATE_NO_WINDOW}
root = Path('C:/hv/d5vk')
root.mkdir(parents=True, exist_ok=True)
batch = root/'environment.cmd'
batch.write_text('@echo off\ncall "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\Common7\\Tools\\VsDevCmd.bat" -no_logo -arch=x64\nif errorlevel 1 exit /b 1\nset VSLANG=1033\nset\n')
result = subprocess.run(['cmd', '/d', '/c', str(batch)], capture_output=True, text=True, **flags)
result.check_returncode()
environment = dict(os.environ)
for line in result.stdout.splitlines():
    if '=' in line and not line.startswith('='):
        key, value = line.split('=', 1)
        environment[key] = value
environment.update(VSLANG='1033', PYTHONUNBUFFERED='1',
    VULKAN_SDK='C:/public/tools/hyle-vulkan/1.4.357.0',
    HYLE_VULKAN_VALIDATION='1', VK_LAYER_PATH='C:/public/tools/hyle-vulkan/1.4.357.0/Bin')
device = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
    capture_output=True, text=True, env=environment, **flags)
device.check_returncode()
(root/'environment.json').write_text(json.dumps({'gpu': device.stdout.strip(),
    'vulkan_sdk': environment['VULKAN_SDK'], 'validation_layer': True}, indent=2)+'\n')

# Compile the owner-only assertion branch once. Only the new cold boundary
# and a representative FFN need execution; the existing long-position failures
# are retained separately and cannot be reported as owner-wide PASS.
owner = root/'receipts/owner'
owner.mkdir(parents=True, exist_ok=False)
steps = []
candidate = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, **flags).strip()
owner_filter = '*D5OfficialFullMoeColdBoundary:*D5Qwen35OfficialFfnUsesMaturePreparedCalls'
def owner_run(label, command):
    started = time.monotonic()
    with (owner/(label+'.log')).open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, **flags)
    steps.append({'label': label, 'argv': command, 'returncode': result.returncode,
                  'seconds': round(time.monotonic()-started, 3)})
    (owner/'receipt.json').write_text(json.dumps({'candidate': candidate,
        'gtest_filter': owner_filter, 'steps': steps}, indent=2)+'\n')
    print('owner', label, result.returncode, flush=True)
    if result.returncode:
        print((owner/(label+'.log')).read_text(errors='replace')[-5000:], flush=True)
        raise SystemExit(result.returncode)

build = str(Path('.scratch/build/m2-vulkan-indexed').resolve())
owner_run('inputs', [sys.executable, 'tests/compute_models/inputs_test.py'])
owner_run('configure', ['cmake', '-S', '.', '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    '-DHYLE_DEV_FACTORS=compute_route_vulkan,graph_route_vulkan', '-DHYLE_BUILD_TESTS=ON',
    '-DHYLE_BUILD_BENCH=OFF', '-DHYLE_INSTALL=OFF', '-DHYLE_TEST_INSTRUMENTATION=OFF',
    '-DHYLE_WERROR=ON', '-DHYLE_COMPUTE_VULKAN=ON'])
owner_run('build', ['cmake', '--build', build, '--target',
    'hyle_compute_route_vulkan_tests', 'hyle_graph_route_vulkan_tests', '-j', '2'])
environment['HYLE_D5_QWEN35_FIXTURE'] = str(Path('.scratch/d5-qwen35-six-profile-fixture').resolve())
environment['GTEST_FILTER'] = owner_filter
try:
    owner_run('tests', ['ctest', '--test-dir', build, '--output-on-failure', '--no-tests=error',
        '-L', '^(compute_route_vulkan|graph_route_vulkan)$', '-LE', 'headers', '-j', '1',
        '--output-junit', str(owner/'tests.xml')])
finally:
    native = Path(build)/'Testing/Temporary/LastTest.log'
    if native.exists():
        (owner/'native-tests.log').write_bytes(native.read_bytes())
environment.pop('GTEST_FILTER')
for stage in ('shared', 'source', 'static'):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('run.py')), stage],
        env=environment, **flags)
    if result.returncode:
        raise SystemExit(result.returncode)
