"""Repair only object files whose MSVC header dependency record is empty."""
from pathlib import Path
import re
import subprocess
import sys

root = Path.cwd().resolve()
route = sys.argv[1]
tree = (root / {'cuda': '.scratch/b-cuda-d5',
                'cpu': '.scratch/build-compute-v2-dev',
                'vulkan': '.scratch/build/m2-vulkan-indexed'}[route]).resolve()
tree.relative_to(root / '.scratch')
raw = subprocess.check_output(['ninja', '-C', str(tree), '-t', 'deps'])
items = []
for match in re.finditer(rb'^(.+\.(?:cc|cpp|c)\.obj): #deps 0,', raw, re.MULTILINE):
    relative = match[1].decode('utf-8')
    target = (tree / relative).resolve()
    target.relative_to(tree)
    if target.is_file():
        items.append(target)
print(route, len(items), 'C/C++ objects have empty header dependencies', flush=True)
if '--apply' in sys.argv:
    receipt = root / '.scratch/validation/d5-f25' / ('dependency-repair-' + route + '.txt')
    receipt.write_text('\n'.join(str(p) for p in items) + '\n', encoding='utf-8')
    for target in items:
        target.unlink()
    print('Removed only these build-local objects; source and CUDA objects untouched.', flush=True)
