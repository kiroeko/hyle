"""Fixed serial ordinary-entry artifact batch; no performance sampling."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
BUILD = Path('C:/hv/vp1')
OUT = Path(__file__).resolve().parent
SDK = Path('C:/public/tools/hyle-vulkan/1.4.357.0')
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
FLAGS = {'creationflags': subprocess.CREATE_NO_WINDOW}
ENV = dict(os.environ, VULKAN_SDK=str(SDK), HYLE_VULKAN_VALIDATION='1',
           HYLE_TEST_VULKAN_VALIDATE='1', VK_LAYER_PATH=str(SDK / 'Bin'))
ENV.pop('HYLE_VULKAN_IQ1_DELTA', None)
candidate = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, **FLAGS).strip()
if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True, **FLAGS).strip():
    raise RuntimeError('batch requires a clean immutable candidate')
tag = 'native-' + candidate[:8]
jobs = [
    ('provider', 'hyle_compute_provider_iree_vulkan_test', '*'),
    ('direct', 'hyle_compute_route_vulkan_test',
     'VulkanIree.*:VulkanTensor.ExactStorageHandlesOddStridesWordsAndEmptyTransfers:'
     'VulkanTensor.UnsignedPayloadViewConcatAndNoArithmetic'),
    ('graph', 'hyle_graph_route_vulkan_test',
     'VulkanGraph.OriginalArtifact*:VulkanGraph.OriginalFloat8*:'
     'VulkanGraph.CheckedScanChainsTwoOriginalProgramsWithIndependentResults'),
]
receipt = {'candidate': candidate, 'build': str(BUILD), 'steps': [],
           'limits': 'Micro source qualification; not all 18 full model geometries, Air or delivery PASS.'}
failed = False
for name, stem, selected in jobs:
    executable = BUILD / 'bin' / (stem + '.exe')
    dlls = sorted((BUILD / 'bin').glob('hyle_*.dll'))
    if not executable.is_file() or not dlls:
        raise RuntimeError('executable or adjacent native DLLs missing')
    log, xml = OUT / (tag + '-' + name + '.log'), OUT / (tag + '-' + name + '.xml')
    if log.exists() or xml.exists():
        raise RuntimeError('immutable candidate batch already attempted; no silent overwrite')
    print('START ' + name, flush=True)
    with log.open('w', encoding='utf-8') as stream:
        code = subprocess.run([str(executable), '--gtest_filter=' + selected,
                               '--gtest_output=xml:' + str(xml)], env=ENV,
                              cwd=executable.parent, stdout=stream,
                              stderr=subprocess.STDOUT, timeout=300, **FLAGS).returncode
    raw = log.read_text(errors='replace')
    validation_error = 'Validation Error' in raw or 'ERROR / SPEC' in raw
    stats = ET.parse(xml).getroot().attrib if xml.exists() else {}
    print('END ' + name + ' exit=' + str(code) + ' tests=' + stats.get('tests', '?'), flush=True)
    receipt['steps'].append({'name': name, 'returncode': code, 'filter': selected,
        'validation_error': validation_error, 'xml': xml.name, 'tests': stats,
        'binary_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
        'log': log.name, 'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
        'adjacent_dlls': [p.name for p in dlls]})
    if code or validation_error or not xml.exists():
        print('\n'.join(raw.splitlines()[-30:]), flush=True)
        failed = True
    (OUT / (tag + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
raise SystemExit(1 if failed else 0)
