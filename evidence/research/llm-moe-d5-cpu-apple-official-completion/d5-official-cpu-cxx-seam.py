import ctypes
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ctypes.windll.kernel32.SetErrorMode(1 | 2 | 0x8000)
sys.stdout.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parents[1]
out = root / '.scratch/validation/d5-cpu-apple-official-sources/cpu-carrier-probe'
sdk = root / '.scratch/validation/d5-cpu-apple-official-sources/cpu-sdk-subset'
headers = Path('C:/Program Files/Python312/Lib/site-packages/torch/include')
dlls = ['torch_cpu.dll', 'c10.dll', 'fbgemm.dll', 'asmjit.dll', 'libiomp5md.dll']
for name in dlls:
    shutil.copy2(sdk / 'lib' / name, out / name)
shutil.copy2(root / '.scratch/validation/d5-mature-native-q1/all_finite_bf16.bin', out)
batch = out / 'build.cmd'
batch.write_text('@echo off\ncall "C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat" -no_logo -arch=x64 -host_arch=x64 >nul\nif errorlevel 1 exit /b %errorlevel%\nset VSLANG=1033\ncl /nologo /std:c++20 /O2 /fp:strict /EHsc /MD /I"'+str(headers)+'" cxx.cc /Fe:cxx.exe /link /LIBPATH:"'+str(sdk/'lib')+'" torch_cpu.lib c10.lib\nif errorlevel 1 exit /b %errorlevel%\ndumpbin /dependents torch_cpu.dll\ndumpbin /dependents cxx.exe\nexit /b %errorlevel%\n',encoding='utf-8')
steps = []
for name, command in [('build',['cmd.exe','/d','/c',str(batch)]), ('run',[str(out/'cxx.exe')])]:
    begin = time.monotonic()
    with (out / (name+'.log')).open('wb') as log:
        result = subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW,timeout=180)
    steps.append(dict(name=name,argv=command,exit_code=result.returncode,seconds=time.monotonic()-begin))
    print(name,result.returncode,(out/(name+'.log')).read_text(encoding='utf-8',errors='replace')[-3500:],flush=True)
    if result.returncode: break
receipt = dict(steps=steps,source_sha256=hashlib.sha256((out/'cxx.cc').read_bytes()).hexdigest(),
               private_cpu_runtime=[dict(name=name,bytes=(out/name).stat().st_size,
                   sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name in dlls],
               scope='CPU-only C++ ABI/DLL seam; not an installed HYLE SDK or full fragment qualification')
(out/'cxx-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
raise SystemExit(result.returncode)
