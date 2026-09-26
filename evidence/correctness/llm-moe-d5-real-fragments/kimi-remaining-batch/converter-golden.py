"""Bounded source-format check; executes the pinned original methods unchanged."""
import ast
import hashlib
import json
import logging
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch

torch.set_num_threads(1)
repo = Path('C:/public/project-hyle/llama.cpp')
pin = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
assert pin == '4c9233c034fc450dcf34c7c0988aebe6da5cdf1a'
source = (repo / 'conversion/kimi_k3.py').read_text(encoding='utf-8')
tree = ast.parse(source)
original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'KimiK3Model')
names = ('_try_fuse_res', 'modify_tensors')
methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]
node = ast.ClassDef(name='KimiK3Model', bases=[ast.Name(id='IdentitySink', ctx=ast.Load())],
                    keywords=[], body=methods, decorator_list=[], type_params=[])
module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[]))

class IdentitySink:
    def modify_tensors(self, data, name, bid):
        yield name, data

namespace = {'torch': torch, 'IdentitySink': IdentitySink,
             'gguf': SimpleNamespace(TENSOR_NAMES={'output': 'output_res_score'}),
             'logger': logging.getLogger('original-kimi-converter')}
exec(compile(module, 'pinned-kimi-methods', 'exec'), namespace)
converter = namespace['KimiK3Model']()
converter.hparams = {'num_key_value_heads': 96, 'num_attention_heads': 96,
                     'v_head_dim': 128, 'qk_nope_head_dim': 128}
converter._res_parts = {}
converter._RES_FUSIONS = {'self_attention_res': ('attn', True), 'mlp_res': ('ffn', True), 'output_attn_res': ('output', False)}
converter.format_tensor_name = lambda role, bid: f'{role}.{bid}.weight'
record = {'pin': pin, 'source': 'conversion/kimi_k3.py', 'torch': torch.__version__,
          'method_sha256': {n.name: hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest() for n in methods},
          'checks': []}

def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

# Same source-coordinate sparse payload as the C++ official BF16 consumer,
# at full original [24576,512] width; compare independent destination indexing.
rows = torch.arange(96 * 256)
columns = (rows * 17 + 199) % 512
codes = (0.5 + (rows % 13).float() / 32).bfloat16()
kv = torch.zeros((96 * 256, 512), dtype=torch.bfloat16)
kv[rows, columns] = codes
results = list(converter.modify_tensors(kv, 'model.layers.3.self_attn.kv_b_proj.weight', 3))
assert len(results) == 2
for key, (name, tensor) in zip((True, False), results):
    n, k = (512, 128) if key else (128, 512)
    expected = torch.zeros((96, n, k), dtype=torch.bfloat16)
    for head in range(96):
        for depth in range(128):
            src_row = head * 256 + (depth if key else 128 + depth)
            src_col = (src_row * 17 + 199) % 512
            expected[head, src_col if key else depth, depth if key else src_col] = codes[src_row]
    assert tensor.dtype == torch.bfloat16 and torch.equal(tensor, expected)
    record['checks'].append({'kind': 'kv_b_split_transpose', 'name': name, 'shape': list(tensor.shape),
                             'storage': 'BF16', 'sha256': digest(tensor), 'verdict': 'EXACT_PASS'})

# Original folded residual scores must keep F32 multiplication of BF16 params.
for role, prefix in enumerate(converter._RES_FUSIONS):
    index = torch.arange(7168)
    norm = (0.875 + ((index * 13 + 5) % 29).float() / 128 + (role + 1) * 0.0037).bfloat16()
    projection = (((index * 7 + role * 13) % 31 - 15).float() / 2048).bfloat16().reshape(1, 7168)
    expected = norm.float() * projection.float().flatten()
    first = list(converter.modify_tensors(norm, prefix + '_norm.weight', 0))
    assert first == []
    fused = list(converter.modify_tensors(projection, prefix + '_proj.weight', 0))
    assert len(fused) == 1 and fused[0][1].dtype == torch.float32
    assert torch.equal(fused[0][1], expected)
    record['checks'].append({'kind': 'residual_parameter_fusion', 'role': prefix,
                             'shape': [7168], 'storage': 'F32', 'sha256': digest(fused[0][1]), 'verdict': 'EXACT_PASS'})
assert not converter._res_parts
out = Path('.scratch/validation/d5-kimi-corrected/converter-golden.json')
out.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print('original converter', len(record['checks']), 'EXACT_PASS')
