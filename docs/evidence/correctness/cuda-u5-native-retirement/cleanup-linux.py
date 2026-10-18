import json
import shutil
from pathlib import Path

task_root = Path('/root/hyle-scratch/active-cpu-z').resolve()
names = ('cuda-u4-static-relocated', 'cuda-u4-static-consumer',
         'cuda-u4-shared-relocated', 'cuda-u4-shared-consumer')
rows = []
for name in names:
    path = task_root / name
    if not path.exists():
        continue
    resolved = path.resolve()
    if resolved.parent != task_root or resolved.name != name or path.is_symlink():
        raise RuntimeError('Owned validation target mismatch: ' + str(path))
    size = sum(p.stat().st_size for p in path.rglob('*')
               if p.is_file() and not p.is_symlink())
    shutil.rmtree(path)
    rows.append({'path': str(path), 'bytes': size, 'removed': not path.exists()})
out = Path('/mnt/c/public/project-hyle/hyle/.scratch/validation/cuda-u5-native/cleanup-linux.json')
out.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
print(json.dumps(rows, indent=2))
