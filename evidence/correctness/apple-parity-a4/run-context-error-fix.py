import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

repo = Path('/Users/kiro/hyle')
out = Path('/Users/kiro/hyle-scratch/apple-parity-delivery')
shared = Path('/Users/kiro/hyle-scratch/d5-graph-825c41f8')
source = Path('/Users/kiro/hyle-scratch/d5-apple-delivery/source-build')
candidate = sys.argv[1]
if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
    raise RuntimeError('unexpected changes in validation copy')
subprocess.run(['git', '-C', str(repo), 'fetch', '-q',
    '/Users/kiro/hyle-scratch/apple-parity/context-error.bundle', 'HEAD'], check=True)
subprocess.run(['git', '-C', str(repo), 'reset', '--hard', '-q', candidate], check=True)
env = dict(os.environ)
env['PATH'] = '/Users/kiro/hyle-verify-tools/metal-17F109/shim:/opt/homebrew/bin:' + env['PATH']
name = 'context-error-' + candidate[:8]
receipt = {'candidate': candidate, 'steps': []}
def run(label, argv, runenv=env):
    print('START ' + label, flush=True)
    with (out / (name + '-' + label + '.log')).open('w') as log:
        code = subprocess.run(argv, cwd=str(repo), env=runenv, stdout=log, stderr=subprocess.STDOUT).returncode
    receipt['steps'].append({'name': label, 'returncode': code})
    (out / (name + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print('END ' + label + ' exit=' + str(code), flush=True)
    if code:
        print('\n'.join((out / (name + '-' + label + '.log')).read_text(errors='replace').splitlines()[-25:]), flush=True)
        raise SystemExit(code)

cmake = '/opt/homebrew/bin/cmake'
run('context-build', [cmake, '--build', str(shared), '--parallel', '2', '--target',
    'hyle_compute_route_apple_test'])
run('context-tests', [str(shared / 'bin/hyle_compute_route_apple_test'),
    '--gtest_filter=AppleRegion.*:AppleLifetime.*', '--gtest_output=xml:' + str(out / (name + '-context.xml'))])
run('shared-configure', [cmake, '-S', str(repo), '-B', str(shared), '-DHYLE_BUILD_TESTS=OFF',
    '-DHYLE_INSTALL=ON', '-DHYLE_TEST_INSTRUMENTATION=OFF'])
run('shared-build', [cmake, '--build', str(shared), '--parallel', '2', '--target', 'hyle_compute', 'hyle_graph'])
run('shared-install', [cmake, '--install', str(shared), '--prefix', str(out / 'shared-relocated')])
run('source-build', [cmake, '--build', str(source), '--parallel', '2', '--target', 'apple_parity_consumer'])
run('static-install', [cmake, '--install', str(source), '--prefix', str(out / 'static-relocated')])
runtime = dict(env)
runtime['PATH'] = '/usr/bin:/bin'
for key in list(runtime):
    if key.startswith('DYLD_'):
        runtime.pop(key)
runtime['DYLD_PRINT_LIBRARIES'] = '1'
for label in ['shared', 'static']:
    build = out / (label + '-consumer')
    run(label + '-consumer-build', [cmake, '--build', str(build), '--parallel', '2', '--target', 'apple_parity_consumer'])
for label, binary in [('source', source / 'apple_parity_consumer'),
    ('static', out / 'static-consumer/apple_parity_consumer'),
    ('shared', out / 'shared-consumer/apple_parity_consumer')]:
    run(label + '-positive', [str(binary)], runtime)
for label in ['source', 'static', 'shared']:
    shutil.copyfile(out / ('missing-resource-' + label + '.log'), out / ('missing-resource-first-' + label + '.log'))
shutil.copyfile(out / 'missing-resource-receipt.json', out / 'missing-resource-first-receipt.json')
run('resource-negatives', ['/usr/bin/python3', '/Users/kiro/hyle-scratch/apple-parity/run-missing-resource.py'])
