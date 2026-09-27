"""Run the selected CUDA factor/header union once against the reusable tree."""
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
out = Path('C:/hv/d5cd/receipts/selected-cuda-gates')
out.mkdir(parents=True, exist_ok=True)
plan_path = repo/'.scratch/validation/d5-cuda-sdk-final-plan.json'
plan = json.loads(plan_path.read_text())
flags = {'creationflags': subprocess.CREATE_NO_WINDOW}
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, **flags).strip()
assert head == plan['candidate_commit']
assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True, **flags).strip()
assert not (out/'receipt.json').exists(), 'Preserve the previous batch'
shutil.copyfile(plan_path, out/'selection-plan.json')
rows = plan['selected_gates']
targets = []
factors = []
headers = []
for row in rows:
    if row['id'].startswith('factor:'):
        command = row['commands'][0]
        targets.append(command[command.index('--target')+1])
        factors.extend(row['factors'])
    elif row['id'].startswith('headers:'):
        headers.extend(row['factors'])
receipt = {'candidate': head, 'selector': str(plan_path), 'steps': [],
           'delivery_receipts': ['shared-provider-runtime', 'source-provider-runtime', 'static-provider-runtime'],
           'note': 'Actual selected factor/header commands are batched by target/label union; delivery uses the fixed CUDA card consumers.'}

def run(label, argv, env=None):
    argv = list(map(str, argv))
    start = time.monotonic()
    with (out/(label+'.log')).open('w', encoding='utf-8') as stream:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, **flags)
    receipt['steps'].append({'label': label, 'argv': argv, 'returncode': result.returncode,
                             'seconds': round(time.monotonic()-start, 3)})
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(label, result.returncode, receipt['steps'][-1]['seconds'], flush=True)
    if result.returncode:
        print((out/(label+'.log')).read_text(errors='replace')[-9000:])
        raise SystemExit(result.returncode)

configure = next(row['commands'][0] for row in rows if row['id'] == 'configure')
configure = [part.replace('{build}', str(build)) for part in configure]
configure += ['-U', 'HYLE_COMPONENTS', '-DHYLE_STATIC=OFF', '-DHYLE_COMPUTE_CPU=OFF',
              '-DHYLE_COMPUTE_ONEDNN=OFF', '-DHYLE_COMPUTE_APPLE=OFF', '-DHYLE_COMPUTE_VULKAN=OFF',
              '-DCMAKE_CUDA_ARCHITECTURES=89-real']
run('configure', configure)
run('build', ['cmake', '--build', build, '--config', 'Release', '--target', *targets, '--parallel', '2'])
env = dict(os.environ)
env.pop('GTEST_FILTER', None)
env['HYLE_D5_QWEN35_FIXTURE'] = str(repo/'.scratch/d5-qwen35-six-profile-fixture')
labels = '^('+'|'.join(sorted(set(factors)))+')$'
run('factors', ['ctest', '--test-dir', build, '-C', 'Release', '--output-on-failure',
                '--no-tests=error', '-L', labels, '-LE', 'headers', '-j', '1',
                '--output-junit', out/'factors.xml'], env)
native = (build/'Testing/Temporary/LastTest.log').read_bytes()
(out/'native-tests.log').write_bytes(native)
receipt['gtest_passed'] = sum(map(int, re.findall(rb'\[  PASSED  \]\s+(\d+) tests?\.', native)))
receipt['gtest_skip_markers'] = native.count(b'[  SKIPPED ]')
if receipt['gtest_skip_markers']:
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    raise SystemExit('Inspect skipped tests before signing this batch')
labels = '^('+'|'.join(sorted(set(headers)))+')$'
run('headers', ['ctest', '--test-dir', build, '-C', 'Release', '--output-on-failure',
                '--no-tests=error', '-L', labels, '-R', '_header_self_contained_test$', '-j', '1',
                '--output-junit', out/'headers.xml'], env)
run('inputs-tool', ['python', 'tests/compute_models/inputs_test.py'])
run('closure', ['python', 'scripts/check-module-closure.py', '--previous-ref', '9401a0c3'])
run('docs', ['python', 'scripts/check-docs.py'])
receipt['verdict'] = 'PASS'
(out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print('selected CUDA factor/header union PASS', flush=True)
