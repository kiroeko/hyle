import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
repo = Path.cwd().resolve()
build = repo/'.scratch/b-cuda-d5'
out = Path('C:/hv/d5cd/receipts/selected-cuda-correction')
out.mkdir(parents=True, exist_ok=True)
flags = {'creationflags': subprocess.CREATE_NO_WINDOW}
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, **flags).strip()
assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True, **flags).strip()
changed = subprocess.check_output(['git', 'diff', '--name-only', '9273bc71', head], text=True, **flags).splitlines()
assert all(name.startswith('docs/') or name == 'tests/compute_route_cuda/context_test.cc' for name in changed)
assert not (out/'receipt.json').exists()
receipt = {'candidate': head, 'product_candidate': '9273bc71017a1febb69db240b6ea8970b88dd77c',
           'changed_files': changed, 'steps': [], 'corrected_test': 'CudaDefaultCompute.DenseFfnUsesFixedRulesAndIndependentSlots'}
def run(label, argv, env=None):
    argv = list(map(str, argv))
    start = time.monotonic()
    with (out/(label+'.log')).open('w', encoding='utf-8') as stream:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, **flags)
    receipt['steps'].append({'label': label, 'argv': argv, 'returncode': result.returncode,
                            'seconds': round(time.monotonic()-start, 3)})
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(label, result.returncode, flush=True)
    if result.returncode:
        print((out/(label+'.log')).read_text(errors='replace')[-6000:])
        raise SystemExit(result.returncode)
run('build', ['cmake', '--build', build, '--target', 'hyle_compute_route_cuda_tests', '-j', '2'])
env = dict(os.environ)
env['HYLE_D5_QWEN35_FIXTURE'] = str(repo/'.scratch/d5-qwen35-six-profile-fixture')
env['GTEST_FILTER'] = receipt['corrected_test']
run('corrected-test', ['ctest', '--test-dir', build, '--output-on-failure', '--no-tests=error',
                       '-R', '^hyle_compute_route_cuda_test$', '-j', '1', '--output-junit', out/'correction.xml'], env)
native = (build/'Testing/Temporary/LastTest.log').read_bytes()
(out/'native-tests.log').write_bytes(native)
assert b'[  SKIPPED ]' not in native and re.search(rb'\[  PASSED  \]\s+1 test\.', native)
env.pop('GTEST_FILTER', None)
run('headers', ['ctest', '--test-dir', build, '--output-on-failure', '--no-tests=error',
                '-L', '^(compute_backend_cuda|compute_provider_cublas|compute_provider_ggml_cuda)$',
                '-R', '_header_self_contained_test$', '-j', '1', '--output-junit', out/'headers.xml'], env)
shutil.copyfile(build/'Testing/Temporary/LastTest.log', out/'native-headers.log')
run('inputs-tool', ['python', 'tests/compute_models/inputs_test.py'])
run('closure', ['python', 'scripts/check-module-closure.py', '--previous-ref', '9401a0c3'])
run('docs', ['python', 'scripts/check-docs.py'])
receipt['verdict'] = 'PASS'
(out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print('targeted assertion correction and remaining gates PASS', flush=True)
