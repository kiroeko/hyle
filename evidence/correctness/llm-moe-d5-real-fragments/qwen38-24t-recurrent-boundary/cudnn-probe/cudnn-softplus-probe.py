import ctypes
import shutil
from pathlib import Path
import subprocess

root = Path.cwd().resolve()
out = root / '.scratch/validation/d5-q24-recurrent-boundary/cudnn-probe'
out.mkdir(parents=True, exist_ok=True)
sdk = Path('C:/Program Files/NVIDIA/CUDNN/v9.24')
cuda = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3')
deps = list((sdk / 'bin/13.3/x64').glob('*.dll'))
for name in ('cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll', 'nvrtc64_130_0.dll', 'nvrtc-builtins64_133.dll', 'nvJitLink_130_0.dll'):
    deps.append(cuda / 'bin/x64' / name)
for source in deps:
    dest = out / source.name
    if not dest.exists():
        shutil.copy2(source, dest)
vs = Path('C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat')
source = Path(__file__).with_suffix('.cc')
cmd = f'"{vs}" -no_logo -arch=x64 && cl /nologo /EHsc /std:c++17 /O2 /W3 /wd4996 /I"{sdk / "include/13.3"}" /I"{cuda / "include"}" "{source}" /Fe:probe.exe /Fo:probe.obj /link /LIBPATH:"{sdk / "lib/13.3/x64"}" /LIBPATH:"{cuda / "lib/x64"}" cudnn.lib cudart.lib'
(out / 'build.cmd').write_text('@echo off\ncall ' + cmd + '\n', encoding='utf-8')
with (out / 'build.log').open('w') as log:
    p = subprocess.run(['cmd', '/c', str(out / 'build.cmd')], cwd=out, stdout=log, stderr=subprocess.STDOUT)
print('build', p.returncode, flush=True)
if p.returncode:
    print((out / 'build.log').read_text()); raise SystemExit(p.returncode)
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
with (out / 'run.log').open('w') as log:
    p = subprocess.run([str(out / 'probe.exe')], cwd=out, stdout=log, stderr=subprocess.STDOUT, timeout=120)
print((out / 'run.log').read_text(), flush=True)
print('run', p.returncode, flush=True)
raise SystemExit(p.returncode)
