"""Run one real Hyle Direct space check in a fresh process for each shape."""
from pathlib import Path
import datetime
import json
import os
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parent
HERE = ROOT/'.scratch/perf-review-2026-10-18/bf16-sdpa'
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/'scripts'))
from hyle_environment import windows_compiler_environment

COMPILER = 'C:/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/cl.exe'
CPU = Path('C:/public/tools/hyle-dependencies/v1/sdks/torch-cpu/Windows-bf0079104e068dcbd60b2f8de589dc725282cbe865a85016cc5b8aa46cab519d/libtorch')
CUDA = Path('C:/public/tools/hyle-dependencies/v1/sdks/torch-cuda-leaf/Windows-2bc729898e422b9f3da54349eed98f2f0b5dd415434508ee2ab2a13fb021815d/torch')
TOOLKIT = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3')
env = windows_compiler_environment({'CMAKE_CXX_COMPILER':COMPILER,'CMAKE_C_COMPILER':COMPILER})
env['VSLANG'] = '1033'
env['PATH'] = os.pathsep.join([str(CPU/'lib'),str(CUDA/'lib'),str(TOOLKIT/'bin/x64'),str(TOOLKIT/'bin'),env['PATH']])

if '--compile' in sys.argv:
    for phase in ('compile','link'):
        command = json.loads((SOURCE/f'hyle-space-{phase}-command.json').read_text())
        if phase == 'compile':
            command = [str(SOURCE/'hyle_space_probe.cc') if arg.endswith('hyle_space_probe.cc') else arg for arg in command]
        result = subprocess.run(command, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (HERE/f'hyle-space-{phase}.log').write_bytes(result.stdout)
        print(result.stdout.decode('utf-8',errors='replace'),flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

if '--run' in sys.argv:
    rows = []
    for length in (512,1024,2048,8192):
        command = [str(HERE/'hyle_space_probe.exe'),str(length)]
        result = subprocess.run(command, cwd=HERE, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=120)
        (HERE/f'hyle-space-{length}.jsonl').write_bytes(result.stdout)
        (HERE/f'hyle-space-{length}.stderr.log').write_bytes(result.stderr)
        print(result.stdout.decode('utf-8',errors='replace'),flush=True)
        if result.stderr:
            print(result.stderr.decode('utf-8',errors='replace'),file=sys.stderr,flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        parsed = [json.loads(line) for line in result.stdout.decode('utf-8').splitlines() if line.strip()]
        if len(parsed)!=1 or not parsed[0]['correct'] or not parsed[0]['provider_match']:
            raise RuntimeError(f'Unexpected Hyle space result for {length}')
        rows.extend(parsed)
    (HERE/'hyle-space-results.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows),encoding='utf-8')
    metadata = {
        'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'platform':platform.platform(),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
        'shape_processes':4,'all_correct':all(r['correct'] for r in rows),
        'peak_extra_algorithm':'resetPeakStats immediately before submit; after_wait.allocated_peak - before_submit.allocated_current',
        'allocator_scope':'Torch allocated/reserved bytes only; Hyle CUDA ports are separate requested payload bytes; driver/runtime/whole process VRAM unmeasured',
        'performance_timing':False,'probe_sdk_copy_or_download':False,
        'probe_scope':'Only the space probe compilation/linking and four fresh-process executions; prerequisite Hyle library/test builds and their separate SDK runtime staging are outside this field',
    }
    (HERE/'hyle-space-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
