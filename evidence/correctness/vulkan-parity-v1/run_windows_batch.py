"""One serial current-candidate batch; native DLLs must be beside each EXE."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BUILD = Path('C:/hv/vp1')
OUT = Path(__file__).resolve().parent
SDK = Path('C:/public/tools/hyle-vulkan/1.4.357.0')
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
flags = {'creationflags': subprocess.CREATE_NO_WINDOW}
env = dict(os.environ, VULKAN_SDK=str(SDK), HYLE_VULKAN_VALIDATION='1',
    HYLE_TEST_VULKAN_VALIDATE='1', VK_LAYER_PATH=str(SDK / 'Bin'))
candidate = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, **flags).strip()
if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True, **flags).strip():
    raise RuntimeError('batch requires an immutable clean candidate')
tag = 'native-' + candidate[:8]
receipt = {'candidate': candidate, 'build': str(BUILD), 'steps': []}
jobs = [
    ('provider', 'hyle_compute_provider_ggml_vulkan_test', '*'),
    ('direct', 'hyle_compute_route_vulkan_test', 'VulkanGguf.*'),
    ('graph', 'hyle_graph_route_vulkan_test', 'VulkanGraphGguf.*'),
]
failed = False
for name, stem, selected in jobs:
    executable = BUILD / 'bin' / (stem + '.exe')
    # hyle_stage_runtime_dependencies has copied target-owned DLLs here.
    # Never launch via a remote DLL directory or permit missing-DLL dialogs.
    required = [p for p in (BUILD / 'bin').glob('hyle_*.dll')]
    if not executable.is_file() or not required:
        raise RuntimeError('native executable or adjacent HYLE DLLs missing')
    log, xml = OUT / (tag + '-' + name + '.log'), OUT / (tag + '-' + name + '.xml')
    print('START ' + name, flush=True)
    with log.open('w', encoding='utf-8') as stream:
        code = subprocess.run([str(executable), '--gtest_filter=' + selected,
            '--gtest_output=xml:' + str(xml)], env=env, cwd=executable.parent,
            stdout=stream, stderr=subprocess.STDOUT, **flags).returncode
    print('END ' + name + ' exit=' + str(code), flush=True)
    receipt['steps'].append({'name': name, 'returncode': code, 'filter': selected,
        'binary_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
        'log': log.name, 'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
        'adjacent_dlls': [p.name for p in required]})
    if code:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-35:]), flush=True)
        failed = True
    (OUT / (tag + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
raise SystemExit(1 if failed else 0)
