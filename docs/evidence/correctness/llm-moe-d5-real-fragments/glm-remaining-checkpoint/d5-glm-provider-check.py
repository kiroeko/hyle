import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['VSLANG'] = '1033'
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
route = sys.argv[1]
tree = Path('.scratch/build-compute-v2-dev' if route == 'cpu' else '.scratch/b-cuda-d5').resolve()
owner = 'ggml' if route == 'cpu' else 'ggml_cuda'
filter = '*AbsorbedMla*:GgmlCpuProvider.*Attention*' if route == 'cpu' else 'GgmlCudaTensor.*Concat*:GgmlCudaTensor.UnsupportedNumericDomains*'
out = Path('.scratch/validation/d5-glm-admission') / route
out.mkdir(parents=True, exist_ok=True)
vs = Path('C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat')
batch = out / 'build.cmd'
batch.write_text(f'@echo off\ncall "{vs}" -no_logo -arch=x64 && cmake --build "{tree}" --target hyle_compute_provider_{owner}_tests -j 2\n')
steps = []
for name, args in [('build', ['cmd', '/c', str(batch)]),
                   ('tests', [str(tree / 'bin' / f'hyle_compute_provider_{owner}_test.exe'), '--gtest_filter='+filter])]:
    with (out/(name+'.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    steps.append({'step':name,'argv':args,'returncode':result.returncode})
    (out/'receipt.json').write_text(json.dumps({'candidate':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'steps':steps}, indent=2)+'\n')
    output = (out/(name+'.log')).read_text(encoding='utf-8',errors='replace')
    print(route, name, result.returncode, output[-900:], flush=True)
    if result.returncode or '[  SKIPPED ]' in output: raise SystemExit(1)
