import json
from pathlib import Path
import shutil

# The first rejected run has complete raw archives in Windows canonical
# evidence. The successful producer, SDKs, fixtures and source cache stay.
root = Path('/Users/kiro/hyle-scratch').resolve(strict=True)
target = root / 'apple-parity-p1-326a40ee'
if target.is_symlink() or target.resolve() != root / 'apple-parity-p1-326a40ee':
    raise RuntimeError('failed producer resolved outside its exact owned path')
if not (root / 'apple-parity/overlap-first-326a40ee.tar.gz').is_file():
    raise RuntimeError('rejected raw archive missing')
size = sum(p.stat().st_size for p in target.rglob('*') if p.is_file() and not p.is_symlink())
shutil.rmtree(target)
print(json.dumps({'scope': str(target), 'removed_bytes': size}))
