import json
from pathlib import Path
import shutil

base=Path('/root/hyle-scratch/active-cpu-z')
rows=[]
for name in ['cuda-c3-static-consumer','cuda-c3-shared-consumer','cuda-c3-core-checks']:
    target=base/name
    if not target.exists():continue
    if target.resolve()!=target or target.parent.resolve()!=base:
        raise RuntimeError('Unexpected cleanup path: '+str(target))
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:exe=(proc/'exe').resolve(strict=True)
        except (OSError,RuntimeError):continue
        if exe.is_relative_to(target):raise RuntimeError('Live process in '+str(target))
    size=sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
    shutil.rmtree(target)
    rows.append({'path':str(target),'bytes':size,'removed':not target.exists()})
output=Path('/mnt/c/public/project-hyle/hyle/evidence/correctness/cuda-best-of-c3/cleanup-linux.json')
output.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows))
