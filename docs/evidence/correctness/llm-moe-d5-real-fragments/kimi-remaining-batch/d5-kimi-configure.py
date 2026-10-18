import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd().resolve()
route = sys.argv[1]
out = root / '.scratch/validation/d5-kimi-owner' / route
tree = root / {'cpu':'.scratch/build-compute-v2-dev','cuda':'.scratch/b-cuda-d5','vulkan':'.scratch/build/m2-vulkan-indexed'}[route]
os.environ['VSLANG']='1033'
ctypes.windll.kernel32.SetErrorMode(0x0001|0x0002|0x8000)
batch = out / 'configure.cmd'
vs = 'C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat'
batch.write_text(f'@echo off\ncall "{vs}" -no_logo -arch=x64 && cmake -S "{root}" -B "{tree}"\n',encoding='utf-8')
start = time.monotonic()
with (out/'configure.log').open('w',encoding='utf-8') as log:
    result = subprocess.run(['cmd','/c',str(batch)],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
record={'candidate':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'route':route,'returncode':result.returncode,'seconds':round(time.monotonic()-start,2)}
(out/'configure-receipt.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(record)
if result.returncode: print((out/'configure.log').read_text(encoding='utf-8',errors='replace')[-3000:])
sys.exit(result.returncode)
