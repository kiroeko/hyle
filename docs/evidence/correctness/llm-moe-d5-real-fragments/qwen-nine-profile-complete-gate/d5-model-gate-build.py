import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd().resolve()
route = sys.argv[1]
out = root / '.scratch/validation/d5-model-gate-batch'
out.mkdir(parents=True, exist_ok=True)
trees = {'cpu': '.scratch/build-compute-v2-dev', 'cuda': '.scratch/b-cuda-d5',
         'vulkan': '.scratch/build/m2-vulkan-indexed'}
targets = {
 'cpu': ['compute', 'compute_backend_cpu', 'compute_ops', 'compute_provider_ggml',
         'compute_provider_onednn', 'compute_provider_native', 'compute_route_cpu',
         'graph', 'graph_provider_onednn', 'graph_route_cpu'],
 'cuda': ['compute_backend_cuda', 'compute_provider_cublas', 'compute_provider_cuda',
          'compute_provider_ggml_cuda', 'graph_provider_ggml_cuda',
          'compute_route_cuda', 'graph_route_cuda'],
 'vulkan': ['compute_provider_ggml_vulkan', 'compute_provider_iree_vulkan',
            'compute_route_vulkan', 'graph_route_vulkan']}
vs = Path('C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat')
command = f'call "{vs}" -no_logo -arch=x64 && cmake --build "{root / trees[route]}" --target '
command += ' '.join('hyle_' + target + '_tests' for target in targets[route]) + ' -j 2'
batch = out / (route + '-build.cmd')
batch.write_text('@echo off\n' + command + '\n', encoding='utf-8')
log_path = out / (route + '-build.log')
with log_path.open('w', encoding='utf-8') as log:
    result = subprocess.run(['cmd', '/c', str(batch)], cwd=root, stdout=log, stderr=subprocess.STDOUT)
print(route, 'build:', result.returncode, flush=True)
print(log_path.read_text(encoding='utf-8', errors='replace')[-7000:])
raise SystemExit(result.returncode)
