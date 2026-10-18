from pathlib import Path
import hashlib
import json
import os
import subprocess
import zipfile

root = Path('/Users/kiro/hyle-verify/r5-apple-20260917').resolve()
commit = 'cbedb837791da15392019299f2652d9d3d70ec9d'
archive = root / 'hyle-r5-delivery-source.zip'
sha = hashlib.sha256(archive.read_bytes()).hexdigest()
assert archive.with_suffix('.zip.sha256').read_text().split()[0] == sha
extracted = root / 'source-delivery-cbedb837'
source = extracted / 'hyle-0.1.0-source'
if not source.exists():
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        z.extractall(extracted)
assert commit in (source / 'SOURCE-MANIFEST.txt').read_text()
parent = root / 'source-consumer-v1'
parent.mkdir(exist_ok=True)
(parent / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.25)
project(apple_source_delivery LANGUAGES CXX)
add_subdirectory("''' + str(source) + '''" hyle)
enable_testing()
foreach(product IN ITEMS compute graph)
  add_executable(apple_source_${product} "''' + str(source) + '''/samples/package/${product}/cpp/main.cc")
  target_compile_features(apple_source_${product} PRIVATE cxx_std_23)
  target_link_libraries(apple_source_${product} PRIVATE hyle::${product})
  hyle_stage_apple_runtime(apple_source_${product})
  add_test(NAME apple_source_${product} COMMAND apple_source_${product})
endforeach()
''')
logs = root / 'logs' / 'delivery-shared-source-v1'
logs.mkdir(exist_ok=True)
env = os.environ.copy()
env['PATH'] = '/Users/kiro/hyle-verify-tools/metal-17F109/shim:/opt/homebrew/bin:' + env['PATH']
for name in ('DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH', 'MLX_METAL_PATH'):
    env.pop(name, None)

def run(label, args, cwd=None, extra_env=None, success=True):
    print(label, flush=True)
    native_env = dict(env, **(extra_env or {}))
    proc = subprocess.run([str(a) for a in args], cwd=cwd, env=native_env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (logs / (label + '.log')).write_text(proc.stdout)
    print(proc.stdout[-6500:], flush=True)
    if success:
        proc.check_returncode()
    return proc

run('all-vendor-source-audit', ['/opt/homebrew/bin/cmake', '-P', source / 'cmake/HyleDependencyVersions.cmake'])
build = root / 'sdk-shared'
original = root / 'install-shared-v1'
prefix = root / 'relocated-shared-v1'
run('source-configure', ['/opt/homebrew/bin/cmake', '-S', parent, '-B', build, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_OSX_DEPLOYMENT_TARGET=14.4',
    '-DHYLE_COMPONENTS=compute;graph', '-DHYLE_COMPUTE_CPU=OFF', '-DHYLE_COMPUTE_CUDA=OFF',
    '-DHYLE_COMPUTE_APPLE=ON', '-DHYLE_STATIC=OFF', '-DHYLE_INSTALL=ON',
    '-DHYLE_BUILD_TESTS=OFF', '-DHYLE_BUILD_BENCH=OFF', '-DHYLE_WERROR=ON',
    '-DHYLE_SANITIZER=off', '-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
    '-DCMAKE_INSTALL_PREFIX=' + str(original)])
run('source-build', ['/opt/homebrew/bin/cmake', '--build', build, '--parallel', '2'])
raw = build / 'hyle/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib'
assert raw.is_file(), raw
hidden = raw.with_name('mlx.metallib.delivery-hidden')
assert not hidden.exists()
raw.rename(hidden)
try:
    run('source-test-without-build-fallback', ['/opt/homebrew/bin/ctest', '--test-dir', build, '--output-on-failure'])
finally:
    hidden.rename(raw)
run('install', ['/opt/homebrew/bin/cmake', '--install', build])
assert not prefix.exists()
original.rename(prefix)
consumers = []
for product in ('compute', 'graph'):
    consumer = root / ('consumer-shared-' + product)
    run('configure-' + product,
        ['/opt/homebrew/bin/cmake', '-S', source / 'samples/package' / product / 'cpp',
         '-B', consumer, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_PREFIX_PATH=' + str(prefix), '-DCMAKE_OSX_DEPLOYMENT_TARGET=14.4'])
    run('build-' + product, ['/opt/homebrew/bin/cmake', '--build', consumer, '--parallel', '2'])
    consumers.append((product, consumer, consumer / ('hyle_package_' + product + '_cpp')))

# Hide every producer resource, including the copy beside its native dylib.
producer_resources = list(build.rglob('mlx.metallib'))
hidden_resources = []
try:
    for resource in producer_resources:
        destination = resource.with_name('mlx.metallib.delivery-hidden')
        assert not destination.exists()
        resource.rename(destination)
        hidden_resources.append((resource, destination))
    for product, consumer, exe in consumers:
        run('relocated-test-' + product,
            ['/opt/homebrew/bin/ctest', '--test-dir', consumer, '--output-on-failure'])
        actual = run('native-loaded-' + product, [exe], extra_env={'DYLD_PRINT_LIBRARIES': '1'})
        loaded = [line for line in actual.stdout.splitlines() if 'libmlx' in line]
        assert len(loaded) == 1 and str(prefix) in loaded[0], loaded
        run('native-links-' + product, ['/usr/bin/otool', '-L', exe])
finally:
    for resource, destination in reversed(hidden_resources):
        destination.rename(resource)

cmake_payload = '\n'.join(p.read_text() for p in (prefix / 'lib/cmake/Hyle').glob('*.cmake'))
for producer_path in (str(source), str(build), str(original), '/Library/Developer/'):
    assert producer_path not in cmake_payload, producer_path
lock = json.loads((source / 'cmake/HyleDependencyLock.json').read_text())
names = {'mlx': 'mlx-LICENSE', 'metal_cpp': 'metal-cpp-LICENSE', 'json': 'mlx-json-LICENSE', 'fmt': 'mlx-fmt-LICENSE'}
for dep in lock['dependencies']:
    if dep['name'] in names:
        expected = dep['license']['files'][0]['sha256']
        data = (prefix / 'share/licenses/Hyle' / names[dep['name']]).read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        assert hashlib.sha256(data).hexdigest() == expected
assert not list((prefix / 'include/mlx').rglob('*.cpp'))
assert not list((prefix / 'include/mlx').rglob('*.metal'))
assert not (prefix / 'include/mlx/backend/metal/kernels.h').exists()
(logs / 'installed-files.txt').write_text('\n'.join(sorted(p.relative_to(prefix).as_posix() for p in prefix.rglob('*') if p.is_file())) + '\n')
receipt = {'candidate': commit, 'source_archive_sha256': sha,
    'journeys': ['complete-source add_subdirectory', 'relocated installed shared SDK'],
    'offline': True, 'cpu_cuda': 'OFF', 'real_compute': 'RMSNorm + Q4 FFN decode/prefill replay',
    'producer_resources_hidden_during_installed_execution': [str(p) for p in producer_resources],
    'single_mlx_dylib_loaded_from_relocated_prefix': True,
    'sdk_paths_headers_licenses': 'PASS', 'result': 'PASS'}
(logs / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
