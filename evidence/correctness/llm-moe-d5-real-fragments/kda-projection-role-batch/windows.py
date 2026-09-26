import ctypes
from pathlib import Path
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
root = Path.cwd().resolve()
route = sys.argv[1]
out = root / '.scratch/validation/d5-kda-projection' / route
out.mkdir(parents=True, exist_ok=True)
tree = root / {'cuda': '.scratch/b-cuda-d5', 'vulkan': '.scratch/build/m2-vulkan-indexed', 'cpu': '.scratch/build-compute-v2-dev'}[route]
vs = Path('C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat')
batch = out / 'build.cmd'
batch.write_text(f'@echo off\ncall "{vs}" -no_logo -arch=x64 && cmake --build "{tree}" --target hyle_compute_route_{route}_tests hyle_graph_route_{route}_tests -j 2\n', encoding='utf-8')
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
with (out / 'build.log').open('w', encoding='utf-8') as log:
    result = subprocess.run(['cmd', '/c', str(batch)], stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
print(route, 'batch build', result.returncode, flush=True)
if result.returncode:
    print((out/'build.log').read_text(encoding='utf-8', errors='replace')[-6000:], flush=True)
    raise SystemExit(result.returncode)
failed = False
for entry in ('compute', 'graph'):
    exe = tree / 'bin' / f'hyle_{entry}_route_{route}_test.exe'
    path = out / f'{entry}.log'
    start = time.monotonic()
    with path.open('w', encoding='utf-8') as log:
        result = subprocess.run([str(exe), '--gtest_filter=*MoeFragments.D5KdaProjectionRoleBatch'], cwd=exe.parent, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    output = path.read_text(encoding='utf-8', errors='replace')
    print(entry, result.returncode, 'seconds', round(time.monotonic()-start, 2), flush=True)
    print('\n'.join(line for line in output.splitlines() if '[  PASSED  ]' in line or '[  FAILED  ]' in line or '[  SKIPPED ]' in line), flush=True)
    if result.returncode:
        print(output[-8000:], flush=True)
        failed = True
raise SystemExit(1 if failed else 0)
