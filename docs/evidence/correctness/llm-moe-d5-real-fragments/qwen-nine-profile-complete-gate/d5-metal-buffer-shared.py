from pathlib import Path
import subprocess

out = Path('.scratch/validation/d5-model-gate-batch').resolve()
prefix = 'export PATH="$HOME/hyle-verify-tools/metal-17F109/shim:/opt/homebrew/bin:$PATH"; '
prefix += 'export HYLE_D5_QWEN35_FIXTURE="$HOME/hyle-scratch/d5-qwen35-six-profile-fixture"; '
build = '$HOME/hyle-scratch/d5-graph-825c41f8'
command = prefix + f'cmake --build "{build}" --target hyle_compute_provider_ggml_metal_tests hyle_compute_route_apple_tests hyle_graph_route_apple_tests -j 2'
command += f' && cd "{build}" && bin/hyle_compute_provider_ggml_metal_test'
for entry in ('compute', 'graph'):
    command += f' && bin/hyle_{entry}_route_apple_test --gtest_filter="*D5NineProfilesUseCompleteGateAndActualStateFeedback"'
path = out / 'apple-shared-resource-regression.log'
with path.open('w', encoding='utf-8') as log:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'kiro@172.22.10.198', command], stdout=log, stderr=subprocess.STDOUT)
print(path.read_text(encoding='utf-8', errors='replace')[-4000:], flush=True)
raise SystemExit(result.returncode)
