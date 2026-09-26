import json
import os
from pathlib import Path
import subprocess
import time

root = Path.home() / 'hyle'
build = Path.home() / 'hyle-scratch/d5-graph-825c41f8'
out = Path.home() / 'hyle-scratch/d5-q24-semantic-batch'
os.environ['PATH'] = str(Path.home() / 'hyle-verify-tools/metal-17F109/shim') + ':/opt/homebrew/bin:' + os.environ['PATH']
os.environ['HYLE_D5_QWEN35_FIXTURE'] = str(Path.home() / 'hyle-scratch/d5-qwen35-six-profile-fixture')
receipt = {'candidate': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(), 'steps': []}
steps = [
    ('owner-configure', ['cmake', '-S', str(root), '-B', str(build), '-DHYLE_WERROR=ON']),
    ('owner-build', ['cmake', '--build', str(build), '--target', 'hyle_compute_provider_ggml_metal_tests', 'hyle_compute_route_apple_tests', 'hyle_graph_route_apple_tests', '-j', '2']),
    ('owner-tests', ['ctest', '--test-dir', str(build), '-C', 'Release', '--output-on-failure', '--no-tests=error', '-L', '^(compute_provider_ggml_metal|compute_route_apple|graph_route_apple)$', '-LE', 'headers', '--output-junit', str(out / 'owner.xml')]),
]
for name, argv in steps:
    print(name, flush=True)
    start = time.monotonic()
    with (out / (name + '.log')).open('w') as log:
        result = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    item = {'name': name, 'argv': argv, 'returncode': result.returncode, 'elapsed_seconds': round(time.monotonic() - start, 3)}
    receipt['steps'].append(item)
    (out / 'owner-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(item), flush=True)
    if result.returncode:
        print((out / (name + '.log')).read_text()[-10000:], flush=True)
        raise SystemExit(result.returncode)
cache = (build / 'CMakeCache.txt').read_text()
(out / 'cache-identity.txt').write_text('\n'.join(line for line in cache.splitlines() if line and not line.startswith(('#', '//')) and any(x in line for x in ('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:', 'CMAKE_CXX_FLAGS', 'CMAKE_GENERATOR:', 'CMAKE_HOME_DIRECTORY:', 'HYLE_', 'BUILD_SHARED_LIBS:'))) + '\n')
(out / 'owner-native.log').write_bytes((build / 'Testing/Temporary/LastTest.log').read_bytes())
