import ctypes
import os
from pathlib import Path
import subprocess
import time

root = Path.cwd().resolve()
out = root / '.scratch/validation/d5-model-gate-batch'
bin_path = root / '.scratch/build/m2-vulkan-indexed/bin'
env = os.environ.copy()
env['HYLE_D5_QWEN35_FIXTURE'] = str(root / '.scratch/d5-qwen35-six-profile-fixture')
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
cases = [
    ('compute_provider_ggml_vulkan', 'GgmlVulkanNative.ExplicitValidationUsesItsOwnInstance', 'validation-layer'),
    ('compute_route_vulkan', '*Gdn*:*Recurrent*:*OfficialAllRegionShapes*:*OfficialCausalConv*', 'direct-correction'),
    ('graph_route_vulkan', '*', 'graph-full')]
for name, selection, label in cases:
    case_env = env.copy()
    if label == 'validation-layer':
        case_env['HYLE_TEST_VULKAN_VALIDATE'] = '1'
        case_env['VK_LAYER_PATH'] = 'C:/public/tools/hyle-vulkan/1.4.357.0/Bin'
    path = out / ('vulkan-' + label + '.log')
    start = time.monotonic()
    with path.open('w', encoding='utf-8') as log:
        result = subprocess.run([str(bin_path / ('hyle_' + name + '_test.exe')),
                                 '--gtest_filter=' + selection],
                                cwd=bin_path, env=case_env, stdout=log, stderr=subprocess.STDOUT)
    print(label, 'exit', result.returncode, 'seconds', round(time.monotonic()-start, 2), flush=True)
    text = path.read_text(encoding='utf-8', errors='replace')
    print('\n'.join(line for line in text.splitlines() if '[  PASSED  ]' in line or '[  FAILED  ]' in line or '[  SKIPPED ]' in line),flush=True)
    if result.returncode:
        print(text[-8000:],flush=True)
        raise SystemExit(result.returncode)
