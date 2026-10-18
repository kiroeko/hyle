import ctypes
import subprocess
from pathlib import Path

ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
repo = Path('C:/public/project-hyle/hyle')
prefix = Path('C:/h7/w-cuda/selected-graph/shared-sdk-relocated')
assert 'set(Hyle_AVAILABLE_COMPUTE_ROUTES "cuda")' in (prefix/'lib/cmake/Hyle/HyleConfig.cmake').read_text()
result = subprocess.run(['cmake','-S',str(repo/'samples/package/graph/cpu_cuda'),
    '-B','C:/h7/h1-missing-composition-consumer','-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_PREFIX_PATH='+str(prefix),'-DHyle_DIR='+str(prefix/'lib/cmake/Hyle')],
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
body = result.stdout.decode(errors='replace')
assert result.returncode != 0
assert "does not contain requested Compute route 'cpu'" in ' '.join(body.split()), body[-2000:]
print('PASS expected rejection: genuine CUDA-only SDK cannot satisfy CPU+CUDA consumer; configure exit='+str(result.returncode))
print(body[-2000:])
