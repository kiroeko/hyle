"""Small native Windows SDK probe; no Hyle build or SDK copying/download."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / '.scratch/perf-review-2026-10-18/bf16-sdpa'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts'))
from hyle_environment import windows_compiler_environment

CPU = Path('C:/public/tools/hyle-dependencies/v1/sdks/torch-cpu/Windows-bf0079104e068dcbd60b2f8de589dc725282cbe865a85016cc5b8aa46cab519d/libtorch')
CUDA = Path('C:/public/tools/hyle-dependencies/v1/sdks/torch-cuda-leaf/Windows-2bc729898e422b9f3da54349eed98f2f0b5dd415434508ee2ab2a13fb021815d/torch')
TOOLKIT = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3')
COMPILER = 'C:/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/cl.exe'
env = windows_compiler_environment({'CMAKE_CXX_COMPILER': COMPILER, 'CMAKE_C_COMPILER': COMPILER})
env['PATH'] = os.pathsep.join([str(CPU/'lib'), str(CUDA/'lib'), str(TOOLKIT/'bin'), env['PATH']])
env['VSLANG'] = '1033'
exe = OUT/'sdpa_probe.exe'
if '--compile' in sys.argv:
    command = [COMPILER, '/nologo', '/std:c++20', '/O2', '/EHsc', '/DNDEBUG', '/utf-8', '/MD', '/wd4702',
               '/I'+str(CPU/'include'), '/I'+str(CPU/'include/torch/csrc/api/include'),
               '/I'+str(CUDA/'include'), '/I'+str(TOOLKIT/'include'), str(HERE/'sdpa_probe.cc'),
               '/Fo'+str(OUT/'sdpa_probe.obj'), '/Fe'+str(exe), '/link',
               str(CPU/'lib/torch.lib'), str(CPU/'lib/torch_cpu.lib'), str(CPU/'lib/c10.lib'),
               str(CUDA/'lib/torch_cuda.lib'), str(CUDA/'lib/c10_cuda.lib')]
    (OUT/'compile-command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    result = subprocess.run(command, cwd=OUT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT/'compile.log').write_bytes(result.stdout)
    print(result.stdout.decode('utf-8', errors='replace'))
    if result.returncode:
        raise SystemExit(result.returncode)
if '--run' in sys.argv:
    result = subprocess.run([str(exe)], cwd=OUT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (OUT/'results.jsonl').write_bytes(result.stdout)
    (OUT/'run.stderr.log').write_bytes(result.stderr)
    print(result.stdout.decode('utf-8', errors='replace'))
    print(result.stderr.decode('utf-8', errors='replace'), file=sys.stderr)
    raise SystemExit(result.returncode)
