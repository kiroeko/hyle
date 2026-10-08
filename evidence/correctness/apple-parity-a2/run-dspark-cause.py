import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

repo = Path('/Users/kiro/hyle')
out = Path('/Users/kiro/hyle-scratch/apple-parity')
delivery = Path('/Users/kiro/hyle-scratch/apple-parity-delivery')
build = Path('/Users/kiro/hyle-scratch/d5-graph-825c41f8')
candidate = sys.argv[1]
if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
    raise RuntimeError('validation copy has unexpected changes')
subprocess.run(['git', '-C', str(repo), 'fetch', '-q', str(out / 'dspark-cause.bundle'), 'HEAD'], check=True)
subprocess.run(['git', '-C', str(repo), 'reset', '--hard', '-q', candidate], check=True)
env = dict(os.environ)
env['PATH'] = '/Users/kiro/hyle-verify-tools/metal-17F109/shim:/opt/homebrew/bin:' + env['PATH']
env['HYLE_D5_QWEN35_FIXTURE'] = str(out / 'qwen35-real-fixture')
def run(name, argv, runenv=env):
    print('START ' + name, flush=True)
    with (out / (name + '.log')).open('w') as log:
        code = subprocess.run(argv, cwd=str(repo), env=runenv,
            stdout=log, stderr=subprocess.STDOUT).returncode
    print('END ' + name + ' exit=' + str(code), flush=True)
    if code:
        print('\n'.join((out / (name + '.log')).read_text(errors='replace').splitlines()[-15:]), flush=True)
    return code

name = 'dspark-cause-' + candidate[:8]
headers = ['hyle/compute/apple.h', 'hyle/compute/detail/apple_plan.h', 'hyle/graph/apple.h']
header_receipt = []
for label in ['source', 'static', 'shared']:
    commands = (delivery / (label + '-commands.log')).read_text()
    line = next(s for s in commands.splitlines() if ' -c ' in s and '/consumer/main.cc' in s)
    tokens = shlex.split(line)
    flags, i = [], 0
    while i < len(tokens):
        token = tokens[i]
        if token in ['-isystem', '-I', '-arch', '-isysroot']:
            flags += tokens[i:i+2]
            i += 2
            continue
        if token.startswith(('-D', '-I', '-mmacosx-version-min=')):
            flags.append(token)
        i += 1
    if '-DHYLE_COMPUTE_APPLE=1' not in flags:
        raise RuntimeError('Apple enabled header flags missing')
    for index, header in enumerate(headers):
        source = out / (name + '-' + label + '-header' + str(index) + '.cc')
        source.write_text('#include <' + header + '>\n')
        code = run(name + '-' + label + '-header' + str(index),
            ['/usr/bin/clang++', '-std=c++23', '-fsyntax-only', *flags, str(source)])
        header_receipt.append({'delivery': label, 'header': header, 'returncode': code, 'flags': flags})
        if code:
            sys.exit(code)
(out / (name + '-headers.json')).write_text(json.dumps(header_receipt, indent=2) + '\n')

code = run(name + '-configure', ['/opt/homebrew/bin/cmake', '-S', str(repo), '-B', str(build),
    '-DHYLE_BUILD_TESTS=ON', '-DHYLE_INSTALL=OFF', '-DHYLE_TEST_INSTRUMENTATION=OFF'])
if code:
    sys.exit(code)
code = run(name + '-build', ['/opt/homebrew/bin/cmake', '--build', str(build), '--parallel', '2',
    '--target', 'hyle_compute_route_apple_test', 'hyle_graph_route_apple_test'])
if code:
    sys.exit(code)
capture_env = dict(env)
capture_env['HYLE_APPLE_DSPARK_CAUSE_CAPTURE'] = '1'
code = run(name + '-direct', [str(build / 'bin/hyle_compute_route_apple_test'),
    '--gtest_filter=*AppleParityDirectBranches.DsparkTargetPrefillAndFiveTokenAttentionCache*',
    '--gtest_output=xml:' + str(out / (name + '-direct.xml'))], capture_env)
print('cause_capture_verdict=' + str(code), flush=True)
for entry in ['Direct', 'Graph']:
    target = 'compute' if entry == 'Direct' else 'graph'
    code = run(name + '-' + entry.lower() + '-target-hidden',
        [str(build / ('bin/hyle_' + target + '_route_apple_test')),
         '--gtest_filter=*AppleParity' + entry + 'Branches.DsparkTargetHiddenAndFp32Confidence*',
         '--gtest_output=xml:' + str(out / (name + '-' + entry.lower() + '-target-hidden.xml'))])
    if code:
        sys.exit(code)
