import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

repo = Path('/Users/kiro/hyle')
out = Path('/Users/kiro/hyle-scratch/apple-parity')
build = out / 'mixed-core'
apple_only = Path('/Users/kiro/hyle-scratch/d5-graph-825c41f8')
candidate = sys.argv[1]
if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
    raise RuntimeError('validation copy has unexpected changes')
subprocess.run(['git', '-C', str(repo), 'fetch', '-q', str(out / 'mixed-core.bundle'), 'HEAD'], check=True)
subprocess.run(['git', '-C', str(repo), 'reset', '--hard', '-q', candidate], check=True)
env = dict(os.environ)
env['PATH'] = '/Users/kiro/hyle-verify-tools/metal-17F109/shim:/opt/homebrew/bin:' + env['PATH']
name = 'mixed-core-' + candidate[:8]
receipt = {'candidate': candidate, 'steps': []}
def run(label, argv):
    print('START ' + label, flush=True)
    with (out / (name + '-' + label + '.log')).open('w') as log:
        code = subprocess.run(argv, cwd=str(repo), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    receipt['steps'].append({'name': label, 'returncode': code})
    (out / (name + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print('END ' + label + ' exit=' + str(code), flush=True)
    if code:
        print('\n'.join((out / (name + '-' + label + '.log')).read_text(errors='replace').splitlines()[-30:]), flush=True)
        sys.exit(code)

run('configure', ['/opt/homebrew/bin/cmake', '-S', str(repo), '-B', str(build), '-G', 'Ninja',
    '-DHYLE_DEV_FACTORS=compute_ggml_runtime,compute_provider_ggml_metal',
    '-DHYLE_COMPUTE_CPU=ON', '-DHYLE_COMPUTE_APPLE=ON', '-DHYLE_COMPUTE_CUDA=OFF',
    '-DHYLE_COMPUTE_VULKAN=OFF', '-DHYLE_COMPUTE_ONEDNN=OFF', '-DHYLE_BUILD_TESTS=ON',
    '-DHYLE_INSTALL=OFF', '-DHYLE_STATIC=OFF', '-DHYLE_TEST_INSTRUMENTATION=OFF',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_OSX_DEPLOYMENT_TARGET=14.4',
    '-DCMAKE_C_COMPILER=/usr/bin/clang', '-DCMAKE_CXX_COMPILER=/usr/bin/clang++'])
sources = []
relative_files = ['include/ggml.h', 'src/ggml-common.h', 'src/ggml-quants.c',
    'src/ggml-quants.h', 'src/ggml-metal/kernels/mul_mv.metal', 'src/ggml-metal/ggml-metal-device.cpp']
for producer in [apple_only, build]:
    compile_commands = subprocess.check_output(['ninja', '-C', str(producer), '-t', 'commands', 'ggml-base'], text=True, env=env)
    roots = [Path(p) for p in set(re.findall(r'/[^\s]+/src/compute_ggml_runtime/source-[^/\s]+', compile_commands))]
    if len(roots) != 1:
        raise RuntimeError('unexpected materialized source roots')
    sources.append(roots[0])
    for path in relative_files:
        if not (roots[0] / path).is_file():
            raise RuntimeError('expected source file missing: ' + path)
receipt['source_equal'] = []
for path in relative_files:
    digests = [hashlib.sha256((source / path).read_bytes()).hexdigest() for source in sources]
    if digests[0] != digests[1]:
        raise RuntimeError('mixed core source drift: ' + path)
    receipt['source_equal'].append({'path': path, 'sha256': digests[0]})
(out / (name + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
run('build', ['/opt/homebrew/bin/cmake', '--build', str(build), '--parallel', '2', '--target',
    'hyle_compute_ggml_runtime_test', 'hyle_compute_provider_ggml_metal_test'])
commands = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands'], text=True, env=env)
(out / (name + '-commands.log')).write_text(commands)
for label, target in [('cpu', 'hyle_compute_ggml_runtime_test'), ('metal', 'hyle_compute_provider_ggml_metal_test')]:
    run(label, [str(build / 'bin' / target), '--gtest_output=xml:' + str(out / (name + '-' + label + '.xml'))])
print('PASS directed CPU/Metal shared source check', flush=True)
