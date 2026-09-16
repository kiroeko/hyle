from pathlib import Path
import hashlib
import json
import os
import subprocess

root = Path('/Users/kiro/hyle-verify/r5-apple-20260917').resolve()
assert root == Path('/Users/kiro/hyle-verify/r5-apple-20260917')
source = root / 'source'
prefix = root / 'relocated-static-v1'
original = root / 'install-static-v1'
logs = root / 'logs' / 'delivery-static-v1'
logs.mkdir(exist_ok=True)
env = os.environ.copy()
env['PATH'] = '/Users/kiro/hyle-verify-tools/metal-17F109/shim:/opt/homebrew/bin:' + env['PATH']
for name in ('DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH', 'MLX_METAL_PATH'):
    env.pop(name, None)

def run(label, args, cwd=None, success=True):
    print(label, flush=True)
    proc = subprocess.run([str(a) for a in args], cwd=cwd, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (logs / (label + '.log')).write_text(proc.stdout)
    print(proc.stdout[-5000:], flush=True)
    if success:
        proc.check_returncode()
    return proc

if original.exists():
    assert not prefix.exists()
    original.rename(prefix)
assert prefix.is_dir() and not original.exists()
executables = []
for product in ('compute', 'graph'):
    build = root / ('consumer-static-' + product)
    run('configure-' + product,
        ['/opt/homebrew/bin/cmake', '-S', source / 'samples/package' / product / 'cpp',
         '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_PREFIX_PATH=' + str(prefix), '-DCMAKE_OSX_DEPLOYMENT_TARGET=14.4'])
    run('build-' + product, ['/opt/homebrew/bin/cmake', '--build', build, '--parallel', '2'])
    exe = build / ('hyle_package_' + product + '_cpp')
    assert exe.is_file() and (build / 'mlx.metallib').is_file()
    assert hashlib.sha256((build / 'mlx.metallib').read_bytes()).digest() == hashlib.sha256((prefix / 'lib/mlx.metallib').read_bytes()).digest()
    executables.append((product, build, exe))

fallbacks = list((root / 'sdk-static').rglob('mlx.metallib'))
assert len(fallbacks) == 1, fallbacks
fallback = fallbacks[0]
hidden = fallback.with_name('mlx.metallib.delivery-hidden')
assert not hidden.exists()
fallback.rename(hidden)
try:
    product, build, exe = executables[0]
    resource = build / 'mlx.metallib'
    absent = build / 'mlx.metallib.delivery-hidden'
    assert not absent.exists()
    resource.rename(absent)
    try:
        negative = run('missing-resource-negative-control', [exe], success=False)
        # MLX is_available catches initialization/library-load failures; the
        # public Context reports that failed native availability check.
        assert negative.returncode != 0 and 'MLX Metal GPU is unavailable' in negative.stdout, negative.stdout
    finally:
        absent.rename(resource)
    for product, build, exe in executables:
        run('relocated-test-' + product,
            ['/opt/homebrew/bin/ctest', '--test-dir', build, '--output-on-failure'])
        run('native-links-' + product, ['/usr/bin/otool', '-L', exe])
finally:
    hidden.rename(fallback)
receipt = {
    'journey': 'relocated installed static SDK; Direct + Graph',
    'build_fallback_hidden': str(fallback),
    'missing_resource_negative_control': 'expected nonzero: MLX Metal GPU is unavailable',
    'actual_compute': 'native RMSNorm + Q4 FFN, rows 1 and 4, three changing inputs',
    'cpu_cuda': 'OFF',
    'resource_sha256': hashlib.sha256((prefix / 'lib/mlx.metallib').read_bytes()).hexdigest(),
    'result': 'PASS'
}
(logs / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
