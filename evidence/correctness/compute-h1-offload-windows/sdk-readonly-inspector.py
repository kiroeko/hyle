import ctypes
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

repo = Path('C:/public/project-hyle/hyle')
delivery = Path('C:/h7/h1-windows-delivery')
candidate = '7eaf67b8a496ee19b609533980cd69e7efb5aa49'
dumpbin = Path('C:/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/dumpbin.exe')
flags = {'creationflags': subprocess.CREATE_NO_WINDOW}
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
headers = ['compute/resource_budget.h', 'compute/cuda_weights.h', 'graph/cpu_cuda.h', 'graph/executable.h']
rows = {}
for mode in ['shared', 'static']:
    sdk = delivery / (mode + '-sdk')
    package = sdk / 'lib/cmake/Hyle'
    config = (package / 'HyleConfig.cmake').read_text()
    targets = (package / 'HyleTargets.cmake').read_text()
    release = (package / 'HyleTargets-release.cmake').read_text()
    recipe = re.search(r'set\(Hyle_COMPUTE_RECIPE_ID "([0-9a-f]{64})"\)', config).group(1)
    assert 'set(Hyle_AVAILABLE_COMPUTE_ROUTES "cpu;cuda")' in config
    assert 'set(Hyle_STATIC ' + ('ON' if mode == 'static' else 'OFF') + ')' in config
    for facade in ['compute', 'graph']:
        block = re.search(r'set_target_properties\(hyle::' + facade + r' PROPERTIES(.*?)\n\)', targets, re.S).group(1)
        assert 'HYLE_COMPUTE_RECIPE_ID "' + recipe + '"' in block
        assert 'HYLE_COMPUTE_ROUTES "cpu;cuda"' in block
    graph = re.search(r'set_target_properties\(hyle::graph PROPERTIES(.*?)\n\)', targets, re.S).group(1)
    assert 'hyle::_factor_graph_route_cpu_cuda' in graph
    assert 'add_library(hyle::_factor_graph_route_cpu_cuda ' + ('STATIC' if mode == 'static' else 'SHARED') + ' IMPORTED)' in targets
    locations = sorted(set(re.findall(r'"\$\{_IMPORT_PREFIX\}/([^";]+)"', release)))
    missing = [name for name in locations if not (sdk / name).is_file()]
    assert not missing, missing
    hashes = {}
    for header in headers:
        expected = subprocess.check_output(['git', 'show', candidate + ':include/hyle/' + header], cwd=repo, **flags)
        actual = (sdk / 'include/hyle' / header).read_bytes()
        assert actual == expected, (mode, header)
        hashes[header] = hashlib.sha256(actual).hexdigest()
    rows[mode] = {'sdk': str(sdk), 'recipe': recipe, 'routes': 'cpu;cuda',
                  'header_sha256': hashes, 'present_imported_locations': len(locations),
                  'composition_export_and_facade_reference': 'PASS'}

apps = delivery / 'shared-apps'
pending = sorted(apps.glob('*.exe'))
assert {p.name for p in pending} == {'hyle_cpu_cuda_sample.exe', 'hyle_cuda_weights_sample.exe', 'hyle_cpu_cuda_contracts.exe'}
seen = set()
imports = {}
system = Path(os.environ['SystemRoot']) / 'System32'
while pending:
    path = pending.pop()
    if path.name.lower() in seen:
        continue
    seen.add(path.name.lower())
    assert len(seen) <= 100
    output = subprocess.check_output([str(dumpbin), '/NOLOGO', '/DEPENDENTS', str(path)], timeout=15, **flags).decode(errors='replace')
    names = sorted(set(re.findall(r'^\s+([\w.\-]+\.dll)\s*$', output, re.M | re.I)))
    resolution = {}
    for name in names:
        local = apps / name
        if local.is_file():
            resolution[name] = 'exe directory'
            if name.lower().startswith('hyle_'):
                pending.append(local)
        elif name.lower().startswith(('api-ms-', 'ext-ms-')):
            resolution[name] = 'Windows API set (not a colocated physical DLL)'
        elif (system / name).is_file():
            resolution[name] = 'Windows System32'
        else:
            raise RuntimeError('Missing direct PE import: ' + path.name + ' -> ' + name)
    imports[path.name] = resolution

receipt = {'candidate': candidate, 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
           'scope': 'Read-only installed metadata/header equality and PE imports; no compilation, linking, app execution or relocation',
           'sdks': rows, 'shared_pe_imports': imports,
           'import_boundary': 'All three EXEs plus recursively reached HYLE DLLs; third-party DLL internals and runtime-only loads are not inspected',
           'not_run': ['installed static consumers', 'installed standalone header compilation', 'relocation', 'WSL', 'new native overlap trace'],
           'heavy_work': 'Stopped pending explicit user continuation'}
target = repo / 'evidence/correctness/compute-h1-offload-windows/sdk-readonly.json'
target.write_bytes((json.dumps(receipt, indent=2) + '\n').encode('utf-8'))
print(json.dumps({'headers_equal': 8, 'sdk_locations_present': {k: v['present_imported_locations'] for k, v in rows.items()},
                  'pe_files_inspected': len(imports), 'native_apps_run': 0}))
