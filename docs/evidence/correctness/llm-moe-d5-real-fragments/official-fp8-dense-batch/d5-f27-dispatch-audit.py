import base64
import ctypes
import os
from pathlib import Path
import subprocess

os.environ['VSLANG'] = '1033'
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
root = Path.cwd().resolve()
out = root / '.scratch/validation/d5-f27/dispatch-audit'
out.mkdir(parents=True, exist_ok=True)
tree = root / '.scratch/build/m2-vulkan-indexed'
vs = Path('C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat')
batch = out / 'build.cmd'
batch.write_text(f'@echo off\ncall "{vs}" -no_logo -arch=x64 && cmake --build "{tree}" --target iree-dump-module -j 2\n', encoding='utf-8')
with (out / 'build.log').open('w', encoding='utf-8') as log:
    result = subprocess.run(['cmd', '/c', str(batch)], stdout=log, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW)
if result.returncode:
    print((out / 'build.log').read_text(encoding='utf-8', errors='replace')[-3000:])
    raise SystemExit(result.returncode)
binary = out / 'fp8.vmfb'
binary.write_bytes(base64.b64decode((root / 'third_party/iree_vulkan_runtime/programs/fp8.vmfb.b64').read_bytes()))
tool = tree / 'bin/iree-dump-module.exe'
for mode in ('disassembly', 'flatbuffer-json'):
    with (out / (mode + '.txt')).open('w', encoding='utf-8') as log:
        subprocess.run([str(tool), '--output=' + mode, str(binary)], stdout=log,
                        stderr=subprocess.STDOUT, check=True, cwd=tool.parent,
                        creationflags=subprocess.CREATE_NO_WINDOW)
print('locked bytecode dumped', out)
