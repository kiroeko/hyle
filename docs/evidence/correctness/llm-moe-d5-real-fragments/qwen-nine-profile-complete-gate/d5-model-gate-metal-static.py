from pathlib import Path
import subprocess

out = Path('.scratch/validation/d5-model-gate-batch').resolve()
prefix = 'export PATH="$HOME/hyle-verify-tools/metal-17F109/shim:/opt/homebrew/bin:$PATH"; '
build = '$HOME/hyle-scratch/d5-model-gate-metal-static'
command = prefix + f'cmake -S "$HOME/hyle" -B "{build}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=14.4 -DHYLE_DEV_FACTORS=compute_provider_ggml_metal -DHYLE_STATIC=ON -DHYLE_BUILD_TESTS=ON -DHYLE_WERROR=ON -DHYLE_COMPUTE_APPLE=ON -DHYLE_COMPUTE_CPU=OFF -DHYLE_COMPUTE_CUDA=OFF -DHYLE_COMPUTE_VULKAN=OFF'
command += f' && cmake --build "{build}" --target hyle_compute_provider_ggml_metal_tests -j 2'
command += f' && cd "{build}" && ctest --output-on-failure -j 1 -R "^hyle_compute_provider_ggml_metal_test$"'
path = out / 'apple-static-resource-final.log'
with path.open('w', encoding='utf-8') as log:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             'kiro@172.22.10.198', command],
                            stdout=log, stderr=subprocess.STDOUT)
print('apple static resource', result.returncode, flush=True)
print(path.read_text(encoding='utf-8', errors='replace')[-6500:], flush=True)
raise SystemExit(result.returncode)
