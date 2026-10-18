import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent / 'before'
out.mkdir(exist_ok=False)
candidate = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
assert candidate.startswith('1c5f1df6')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root)
exe = Path('C:/hyle-scratch/active-vk/static/bin/hyle_compute_route_vulkan_test.exe')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
ctypes.windll.kernel32.SetErrorMode(0x8003)
env = dict(os.environ, VK_LAYER_PATH='C:/public/tools/hyle-vulkan/1.4.357.0/Bin',
    VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation', VK_LOADER_LAYERS_DISABLE='~implicit~',
    HYLE_TEST_VULKAN_VALIDATE='1', HYLE_VULKAN_VALIDATION='1', TRACY_ONLY_LOCALHOST='1')
env.pop('PATH', None)
env.pop('GTEST_FILTER', None)
log, report = out / 'direct.log', out / 'direct.json'
command = [str(exe), '--gtest_filter=VulkanTopK.IntegerPaddingKeepsFinalWinnerAtLargePowerBoundaries',
           '--gtest_output=json:' + str(report)]
with log.open('w') as stream:
    done = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
text = log.read_text(encoding='utf-8', errors='replace')
print(text[-7000:])
data = json.loads(report.read_text(encoding='utf-8'))
result = {'candidate': candidate, 'scope': 'Y30 before-fix ordinary Vulkan TopK counterexample, no P1',
    'status': 'FAIL' if done.returncode else 'PASS', 'exit_code': done.returncode,
    'command': command, 'executable_sha256': sha(exe), 'log_sha256': sha(log),
    'gtest_sha256': sha(report), 'tests': data['tests'], 'failures': data['failures'],
    'validation_error': any(m in text for m in ('VUID-', 'Validation Error', 'SYNC-HAZARD'))}
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
assert done.returncode == 1 and data['tests'] == data['failures'] == 1
assert not result['validation_error']
