"""Capture delivery provenance in the existing batch directory, no product edits."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys

stage = sys.argv[1]
root = Path('C:/hv/d5cd')
out = root / 'receipts' / (stage+'-provider-runtime')
build = root / ('source-build' if stage == 'source' else stage+'-consumer')
tool = Path('C:/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/dumpbin.exe')
sdk = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64')
audit = []
for path in sorted([*build.glob('*.exe'), *build.glob('*.dll')]):
    run = subprocess.run([str(tool), '/dependents', str(path)], capture_output=True,
                         text=True, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    imports = re.findall(r'^\s+([A-Za-z0-9_.-]+\.dll)\s*$', run.stdout, re.M | re.I)
    missing = [name for name in imports if not name.lower().startswith(('api-ms-', 'ext-ms-'))
               and not (build/name).exists() and not (Path('C:/Windows/System32')/name).exists()]
    audit.append({'file': path.name, 'imports': imports, 'missing_near_exe_or_system': missing})
(out/'dll-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
assert not any(row['missing_near_exe_or_system'] for row in audit), audit
origins = []
for name in ['cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll']:
    origin = sdk/name
    staged = build/name
    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    expected, actual = digest(origin), digest(staged)
    assert expected == actual, name
    origins.append({'name': name, 'source': str(origin), 'staged': str(staged),
                    'bytes': staged.stat().st_size, 'sha256': actual})
(out/'cuda-dll-origins.json').write_text(json.dumps(origins, indent=2)+'\n')
for target in ['d5_cuda_direct', 'd5_cuda_graph']:
    shutil.copyfile(build/'hyle-runtime'/(target+'-Release.cmake'), out/(target+'-runtime.cmake'))
print(stage, 'PE closure and all three original SDK DLL hashes verified')
