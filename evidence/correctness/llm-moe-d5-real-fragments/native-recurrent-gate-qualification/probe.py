import ctypes
import os
from pathlib import Path
import shutil
import subprocess

root = Path.cwd().resolve()
out = root / '.scratch/validation/d5-model-gate-native'
out.mkdir(parents=True, exist_ok=True)
build = root / '.scratch/b-cuda-d5'
source = Path(__file__).with_suffix('.cc')
vendor = build / 'src/compute_ggml_runtime/source-4e1a7b2f39afe15d'
cuda = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3')
vs = Path('C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat')
for name in ('cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll'):
    src = build / 'bin' / name
    target = out / name
    if not target.exists():
        try:
            os.link(src, target)
        except OSError:
            shutil.copy2(src, target)
cmd = f'call "{vs}" -no_logo -arch=x64 && cl /nologo /EHsc /std:c++20 /O2 /MD /W4 /WX /I"{vendor / "include"}" "{source}" /Fe:probe.exe /Fo:probe.obj /link /LIBPATH:"{build / "bin"}" /LIBPATH:"{cuda / "lib/x64"}" ggml-base.lib ggml-cuda.lib cuda.lib cudart.lib cublas.lib cublasLt.lib advapi32.lib'
(out / 'build.cmd').write_text('@echo off\n' + cmd + '\n', encoding='utf-8')
with (out / 'build.log').open('w') as log:
    p = subprocess.run(['cmd', '/c', str(out / 'build.cmd')], cwd=out, stdout=log, stderr=subprocess.STDOUT)
print('build', p.returncode, flush=True)
if p.returncode:
    print((out / 'build.log').read_text()[-6000:]); raise SystemExit(p.returncode)
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
with (out / 'run.log').open('w') as log:
    p = subprocess.run([str(out / 'probe.exe'), str(root / '.scratch/d5-qwen35-six-profile-fixture')], cwd=out, stdout=log, stderr=subprocess.STDOUT, timeout=180)
print((out / 'run.log').read_text(), flush=True)
raise SystemExit(p.returncode)
