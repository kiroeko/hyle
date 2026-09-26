import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd().resolve()
route = sys.argv[1]
os.environ['VSLANG'] = '1033'
os.environ['HYLE_D5_QWEN35_FIXTURE'] = str(root / '.scratch/d5-qwen35-six-profile-fixture')
os.environ['VULKAN_SDK'] = 'C:/public/tools/hyle-vulkan/1.4.357.0'
if route == 'vulkan':
    os.environ['HYLE_TEST_VULKAN_VALIDATE'] = '1'
    os.environ['VK_LAYER_PATH'] = 'C:/public/tools/hyle-vulkan/1.4.357.0/Bin'
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
build_name, factors = {
 'cpu': ('.scratch/build-compute-v2-dev', ['compute_provider_ggml', 'compute_route_cpu', 'graph_route_cpu']),
 'cuda': ('.scratch/b-cuda-d5', ['compute_provider_ggml_cuda', 'graph_provider_ggml_cuda', 'compute_route_cuda', 'graph_route_cuda']),
 'vulkan': ('.scratch/build/m2-vulkan-indexed', ['compute_provider_ggml_vulkan', 'compute_provider_iree_vulkan', 'compute_route_vulkan', 'graph_route_vulkan'])
}[route]
build = root / build_name
out = root / '.scratch/validation/d5-kimi-owner' / route
out.mkdir(parents=True, exist_ok=True)
receipt = {'candidate': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'route': route, 'factors': factors, 'steps': []}

def run(name, args):
    start = time.monotonic()
    with (out / (name + '.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    receipt['steps'].append({'name': name, 'argv': args, 'returncode': result.returncode,
                              'seconds': round(time.monotonic() - start, 2)})
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(route, name, result.returncode, receipt['steps'][-1]['seconds'], flush=True)
    text = (out / (name + '.log')).read_text(encoding='utf-8', errors='replace')
    print(text[-2500:], flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

vs = Path('C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat')
batch = out / 'build.cmd'
targets = ' '.join('hyle_' + factor + '_tests' for factor in factors)
batch.write_text(f'@echo off\ncall "{vs}" -no_logo -arch=x64 && cmake --build "{build}" --target {targets} -j 2\n', encoding='utf-8')
run('build', ['cmd', '/c', str(batch)])
cache = (build / 'CMakeCache.txt').read_text(encoding='utf-8')
(out / 'cache-identity.txt').write_text('\n'.join(line for line in cache.splitlines() if line and not line.startswith(('#','//')) and any(s in line for s in ('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:', 'CMAKE_CXX_FLAGS', 'CMAKE_GENERATOR:', 'CMAKE_HOME_DIRECTORY:', 'HYLE_', 'BUILD_SHARED_LIBS:', 'CMAKE_CUDA_ARCHITECTURES:'))) + '\n', encoding='utf-8')
run('tests', ['ctest', '--test-dir', str(build), '-C', 'Release', '--output-on-failure', '--no-tests=error', '-L', '^(' + '|'.join(factors) + ')$', '-LE', 'headers', '-j', '1', '--output-junit', str(out / 'tests.xml')])
(out / 'native-tests.log').write_bytes((build / 'Testing/Temporary/LastTest.log').read_bytes())
native = (out / 'native-tests.log').read_text(encoding='utf-8', errors='replace')
if '[  SKIPPED ]' in native:
    print('SKIP detected; inspect before accepting this receipt', flush=True)
    raise SystemExit(2)
