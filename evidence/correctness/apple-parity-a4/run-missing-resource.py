import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path('/Users/kiro/hyle-scratch/apple-parity-delivery')
source = Path('/Users/kiro/hyle-scratch/d5-apple-delivery/source-build')
shared = Path('/Users/kiro/hyle-scratch/d5-graph-825c41f8')
assets = list(source.rglob('mlx.metallib')) + list(shared.rglob('mlx.metallib'))
for label in ['static', 'shared']:
    assets += list((root / (label + '-relocated')).rglob('mlx.metallib'))
    assets += list((root / (label + '-consumer')).rglob('mlx.metallib'))
assets = sorted(set(p.resolve() for p in assets))
records = []
for path in assets:
    path.relative_to('/Users/kiro/hyle-scratch')
    records.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
env = dict(os.environ)
env['PATH'] = '/usr/bin:/bin'
for key in list(env):
    if key.startswith('DYLD_'):
        env.pop(key)
env['DYLD_PRINT_LIBRARIES'] = '1'
receipt = {'assets': records, 'results': []}
hidden = []
try:
    for path in assets:
        destination = path.with_name(path.name + '.apple-parity-missing-test')
        if destination.exists():
            raise RuntimeError('stale hidden resource: ' + str(destination))
        path.rename(destination)
        hidden.append((path, destination))
    for label, binary in [('source', source / 'apple_parity_consumer'),
        ('static', root / 'static-consumer/apple_parity_consumer'),
        ('shared', root / 'shared-consumer/apple_parity_consumer')]:
        print('START missing-resource ' + label, flush=True)
        with (root / ('missing-resource-' + label + '.log')).open('w') as log:
            result = subprocess.run([str(binary)], env=env, stdout=log, stderr=subprocess.STDOUT)
        output = (root / ('missing-resource-' + label + '.log')).read_text(errors='replace')
        valid = result.returncode == 1 and 'FAIL ' in output and 'metallib' in output
        receipt['results'].append({'delivery': label, 'returncode': result.returncode,
            'clear_missing_resource': valid})
        print(label + ' exit=' + str(result.returncode) + ' clear_missing=' + str(valid), flush=True)
finally:
    for original, destination in reversed(hidden):
        destination.rename(original)
    receipt['restored'] = all(Path(p['path']).is_file() and
        hashlib.sha256(Path(p['path']).read_bytes()).hexdigest() == p['sha256'] for p in records)
    (root / 'missing-resource-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
if not receipt['restored'] or len(receipt['results']) != 3 or not all(r['clear_missing_resource'] for r in receipt['results']):
    raise SystemExit(1)
print('PASS three explicit resource negatives and exact restoration', flush=True)
