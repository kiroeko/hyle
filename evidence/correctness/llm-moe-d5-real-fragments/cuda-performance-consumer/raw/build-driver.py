import ctypes
import os
from pathlib import Path
import subprocess
import sys

ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
root = Path(__file__).resolve().parents[1]
build = Path('C:/hv/d5a6')
build.mkdir(parents=True, exist_ok=True)
flags = {'creationflags': subprocess.CREATE_NO_WINDOW}
batch = root / '.scratch/d5-cuda-delivery-env.cmd'
result = subprocess.run(['cmd', '/c', str(batch)], capture_output=True, text=True, **flags)
result.check_returncode()
environment = dict(os.environ)
for line in result.stdout.splitlines():
    if '=' in line and not line.startswith('='):
        key, value = line.split('=', 1)
        environment[key] = value
environment['VSLANG'] = '1033'
environment['HYLE_D5_QWEN35_FIXTURE'] = str(root / '.scratch/d5-qwen35-six-profile-fixture')
environment['HYLE_BENCH_VERIFY_ONLY'] = '1'
environment['HYLE_PERF_CALLER_CPU'] = '2'
cache = Path(os.environ['LOCALAPPDATA']) / 'hyle/dependencies/v1/sources'
commands = [
    ['cmake', '-S', str(root / 'bench/graph_route_cuda/external'), '-B', str(build), '-G', 'Ninja',
     '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_MAKE_PROGRAM=C:/PROGRA~1/PYTHON~1/Scripts/ninja.exe',
     '-DCMAKE_PREFIX_PATH=C:/hv/d5cd/static-relocated',
     '-DD5_GGML_SOURCE=' + str(cache / 'ggml-456172ec733a-41b7e10fd424'),
     '-DD5_GTEST_SOURCE=' + str(cache / 'googletest-063de7e9578f-162765170ae6')],
    ['cmake', '--build', str(build), '--target', 'hyle_graph_route_cuda_bench', '--parallel', '2'],
    ['ctest', '--test-dir', str(build), '--output-on-failure', '-R', '^d5_cuda_fragment_verify$'],
]
for index, command in enumerate(commands):
    print(f'stage {index}: {command[0]}', flush=True)
    log = build / f'stage-{index}.log'
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, env=environment, cwd=root, stdout=stream,
                                stderr=subprocess.STDOUT, text=True, **flags)
    print(f'exit={result.returncode} log={log}', flush=True)
    if result.returncode:
        print(log.read_text(encoding='utf-8', errors='replace')[-14000:])
        raise SystemExit(result.returncode)
