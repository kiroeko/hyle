import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd().resolve()
route = sys.argv[1]
out = root / '.scratch/validation/d5-model-gate-batch'
trees = {'cpu': '.scratch/build-compute-v2-dev', 'cuda': '.scratch/b-cuda-d5',
         'vulkan': '.scratch/build/m2-vulkan-indexed'}
names = {
 'cpu': ['compute_ops', 'compute_provider_ggml', 'compute', 'compute_backend_cpu',
         'compute_provider_onednn', 'compute_provider_native', 'graph',
         'graph_executable', 'graph_provider_onednn', 'compute_route_cpu',
         'graph_route_cpu'],
 'cuda': ['compute_provider_ggml_cuda', 'graph_provider_ggml_cuda',
          'compute_backend_cuda', 'compute_provider_cublas', 'compute_provider_cuda',
          'compute_route_cuda', 'graph_route_cuda'],
 'vulkan': ['compute_provider_ggml_vulkan', 'compute_provider_iree_vulkan',
            'compute_route_vulkan', 'graph_route_vulkan']}
env = os.environ.copy()
env['HYLE_D5_QWEN35_FIXTURE'] = str(root / '.scratch/d5-qwen35-six-profile-fixture')
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
for name in names[route]:
    exe = root / trees[route] / 'bin' / ('hyle_' + name + '_test.exe')
    if not exe.exists():
        raise SystemExit('Missing test artifact: ' + str(exe))
    log_path = out / (route + '-' + name + '.log')
    start = time.monotonic()
    with log_path.open('w', encoding='utf-8') as log:
        args = [str(exe)]
        if len(sys.argv) > 2:
            args += ['--gtest_filter=' + sys.argv[2]]
        result = subprocess.run(args, cwd=exe.parent, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    print(name, 'exit:', result.returncode, 'seconds:', round(time.monotonic()-start, 2), flush=True)
    data = log_path.read_text(encoding='utf-8', errors='replace')
    print('\n'.join(line for line in data.splitlines() if '[  PASSED  ]' in line or '[  FAILED  ]' in line or '[  SKIPPED ]' in line), flush=True)
    if result.returncode:
        print(data[-4000:], flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
