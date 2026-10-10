"""Original C3 build/install steps only; unavailable Vulkan GPU runs are BLOCKED."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
routes = sys.argv[1]
assert routes in ('vulkan', 'cpu;vulkan')
root = Path('/root/hyle-scratch/compute-unified')
source = root / 'source-7a7b4c3b'
batch = root / ('l-' + routes.replace(';', '-'))
assert not (batch / 'selected-graph').exists(), 'Resume the existing batch.'
batch.mkdir(exist_ok=True)
driver = root / 'wsl-vulkan-build-only.cmake'
facts = json.loads((root / 'wsl-vulkan-current-device-facts.json').read_text())
assert not any(d['device_type'] in (1, 2, 3) for d in facts['devices'])
argv = [str(source / 'scripts/hyle-unix.sh'), 'cmake',
    '-DHYLE_SOURCE_DIR=' + str(source), '-DHYLE_SMOKE_ROOT=' + str(batch),
    '-DHYLE_SMOKE_COMPONENT=graph', '-DHYLE_SMOKE_GENERATOR=Ninja',
    '-DHYLE_SMOKE_COMPUTE_ROUTES=' + routes, '-P', str(driver)]
identity = {'scope': 'Original source/shared/static configure/build/install/relocation/assets and consumer link steps; physical Vulkan GPU math not executed',
    'product_candidate': 'a2476b62cfe49c1663998645c362392685c06eb1',
    'validation_source_commit': 'a2476b62cfe49c1663998645c362392685c06eb1',
    'source_archive_sha256': 'd0c2377d14be5752f6e1876a54d8c568cb1cc32a7066391bd784699a1ad14c86',
    'routes': routes, 'argv': argv, 'runtime_verdict': 'BLOCKED',
    'reason': 'Only software Vulkan device; ordinary route requires physical GPU',
    'driver_sha256': hashlib.sha256(driver.read_bytes()).hexdigest()}
(batch / 'launch.json').write_text(json.dumps(identity, indent=2) + '\n')
log = batch / 'batch.log'
print('C3 build/install only: ' + routes + ' -> ' + str(log), flush=True)
with log.open('w') as output:
    result = subprocess.run(argv, cwd=source, stdout=output, stderr=subprocess.STDOUT)
(batch / 'build-terminal.json').write_text(json.dumps({'build_install_exit': result.returncode,
    'runtime_verdict': 'BLOCKED', 'math_executed': False}, indent=2) + '\n')
for line in log.read_text(errors='replace').splitlines()[-12:]:
    print(line[:180], flush=True)
print('Build/install exit=' + str(result.returncode) + '; runtime remains BLOCKED', flush=True)
raise SystemExit(result.returncode)
