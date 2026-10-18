import ctypes
import os
from pathlib import Path
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['VSLANG'] = '1033'
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
root = Path.cwd()
tree = root / '.scratch/b-cuda-d5'
out = root / '.scratch/validation/d5-f25/extra'
out.mkdir(parents=True, exist_ok=True)
vs = Path('C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat')
batch = out / 'build.cmd'
batch.write_text(f'@echo off\ncall "{vs}" -no_logo -arch=x64 && cmake --build "{tree}" --target hyle_compute_ops_tests hyle_compute_provider_cuda_tests -j 2 && ctest --test-dir "{tree}" -R "^hyle_compute_(ops|provider_cuda)_header_self_contained_test$" --output-on-failure\n', encoding='utf-8')
with (out / 'build.log').open('w', encoding='utf-8') as log:
    result = subprocess.run(['cmd', '/c', str(batch)], stdout=log, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW)
print('extra build', result.returncode, flush=True)
if result.returncode:
    print((out / 'build.log').read_text(encoding='utf-8', errors='replace')[-5000:])
    raise SystemExit(result.returncode)
for binary, selection in [('hyle_compute_ops_test.exe', '*'),
                          ('hyle_compute_provider_cuda_test.exe', 'SglangPrimitives.*:CudaKernels.PreciseSglangVectorsUseIndependentOracleAndSlots:*.PendingCompletionRetainsProgramAndSlotAfterClose:*.ColdMetadataFailuresRollbackAndReplayUsesReservedState')]:
    exe = tree / 'bin' / binary
    with (out / (binary + '.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run([str(exe), '--gtest_filter=' + selection], cwd=exe.parent,
                                stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    print(binary, result.returncode, flush=True)
    if result.returncode:
        print((out / (binary + '.log')).read_text(encoding='utf-8', errors='replace')[-6000:])
        raise SystemExit(result.returncode)
