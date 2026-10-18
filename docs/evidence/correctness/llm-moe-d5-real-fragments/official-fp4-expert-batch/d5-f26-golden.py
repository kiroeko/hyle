import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

repo = Path('C:/public/project-hyle/llama.cpp')
pin = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
assert pin == '4c9233c034fc450dcf34c7c0988aebe6da5cdf1a'
source = (repo / 'conversion/base.py').read_text(encoding='utf-8')
fn = next(n for n in ast.walk(ast.parse(source))
          if isinstance(n, ast.FunctionDef) and n.name == 'repack_mxfp4_blocks')
segment = ast.get_source_segment(source, fn)
fn.decorator_list = []
namespace = {'torch': torch, 'Tensor': torch.Tensor, 'np': np}
exec(compile(ast.Module(body=[fn], type_ignores=[]), 'pinned-converter', 'exec'), namespace)
packed = torch.tensor([(i * 37 + 11) % 256 for i in range(64)], dtype=torch.uint8).reshape(2, 32)
scale = torch.tensor([[0, 127], [254, 255]], dtype=torch.uint8)
output = namespace['repack_mxfp4_blocks'](packed, scale)
out = Path('.scratch/validation/d5-f26')
out.mkdir(parents=True, exist_ok=True)
record = {'pin': pin, 'source': 'conversion/base.py::ModelBase.repack_mxfp4_blocks',
          'function_sha256': hashlib.sha256(segment.encode()).hexdigest(),
          'torch': torch.__version__, 'numpy': np.__version__,
          'packed': packed.flatten().tolist(), 'scale': scale.flatten().tolist(),
          'result': output.flatten().tolist()}
(out / 'converter-golden.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print('golden', record['result'])
