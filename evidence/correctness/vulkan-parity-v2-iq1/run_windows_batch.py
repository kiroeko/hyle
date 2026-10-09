"""Serial original IQ1 source qualification; hidden children, adjacent DLLs."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
BUILD = Path('C:/hv/vp1/bin');OUT = Path(__file__).resolve().parent
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
FLAGS = {'creationflags': subprocess.CREATE_NO_WINDOW}
ENV = dict(os.environ, HYLE_VULKAN_VALIDATION='1', HYLE_TEST_VULKAN_VALIDATE='1',
           VK_LAYER_PATH='C:/public/tools/hyle-vulkan/1.4.357.0/Bin')
ENV.pop('HYLE_VULKAN_IQ1_DELTA', None)
candidate = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,**FLAGS).strip()
if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True,**FLAGS).strip():
    raise RuntimeError('clean immutable candidate required')
tag = 'native-' + candidate[:8]
jobs = [
    ('provider', 'hyle_compute_provider_iree_vulkan_test.exe', '*'),
    ('direct', 'hyle_compute_route_vulkan_test.exe',
     'VulkanIree.OriginalIq1*'),
    ('graph', 'hyle_graph_route_vulkan_test.exe',
     'VulkanGraph.OriginalIq1*'),
]
receipt = {'candidate': candidate, 'steps': [], 'scope': 'original IQ1 three-format source batch; eight actual cold geometries only, full bank GPU/catalogue/platform/delivery/P1 unproven'}
failed = False
for label, name, selected in jobs:
    exe = BUILD / name;dlls = sorted(BUILD.glob('hyle_*.dll'))
    if not exe.exists() or not dlls: raise RuntimeError('EXE or adjacent DLLs missing')
    log = OUT / (tag + '-' + label + '.log');xml = log.with_suffix('.xml')
    if log.exists() or xml.exists(): raise RuntimeError('candidate already attempted, no overwrite')
    print('START ' + label, flush=True)
    with log.open('w', encoding='utf-8') as stream:
        code = subprocess.run([str(exe),'--gtest_filter=' + selected,'--gtest_output=xml:' + str(xml)],
            cwd=BUILD,env=ENV,stdout=stream,stderr=subprocess.STDOUT,timeout=300,**FLAGS).returncode
    raw = log.read_text(errors='replace');validation = 'Validation Error' in raw or 'ERROR / SPEC' in raw
    stats = ET.parse(xml).getroot().attrib if xml.exists() else {}
    print('END ' + label + ' exit=' + str(code) + ' tests=' + stats.get('tests','?'), flush=True)
    receipt['steps'].append({'name':label,'returncode':code,'filter':selected,'tests':stats,
        'validation_error':validation,'log':log.name,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
        'xml':xml.name,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
        'adjacent_dlls':[p.name for p in dlls]})
    if code or validation or not xml.exists():
        print('\n'.join(raw.splitlines()[-25:]),flush=True);failed=True
    (OUT / (tag + '-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
raise SystemExit(1 if failed else 0)
