"""Maintainer-only AOT regeneration; never invoked by product CMake or runtime."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--triton-path', type=Path)
args = parser.parse_args()
if args.triton_path:
    sys.path.insert(0, str(args.triton_path))
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

assert triton.__version__ == '3.8.0', 'Use the exact wheel in compiler-wheel.json'
root = Path(__file__).resolve().parent
identity = json.loads((root / 'source.json').read_text())
original = root / 'original_kernels.py'
assert hashlib.sha256(original.read_bytes()).hexdigest() == identity['module_sha256']
spec = importlib.util.spec_from_file_location('original_kernels', original)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
job = next(x for x in json.loads((root / 'compilation.json').read_text())['kernels']
           if x['label'] == 'matmul')
compiled = triton.compile(ASTSource(fn=module._w8a8_block_fp8_matmul,
    signature=job['signature'], constexprs=job['constexprs']),
    target=GPUTarget('cuda', 89, 32), options=job['options'])
blob = compiled.asm['cubin']
args.output.mkdir(parents=True, exist_ok=True)
(args.output / 'matmul.cubin').write_bytes(blob)
(args.output / 'matmul.ptx').write_text(compiled.asm['ptx'], encoding='utf-8', newline='\n')
lines = ['// Generated from unchanged SGLang 06992c92 with Triton 3.8.0. See source.json.',
         '// SHA256 cubin: ' + hashlib.sha256(blob).hexdigest(), '#pragma once',
         'namespace hyle::compute::cuda_native::fp8 {',
         'alignas(16) inline constexpr unsigned char kSglangFp8Sm89[] = {']
lines += ['  ' + ','.join(f'0x{x:02x}' for x in blob[i:i+32]) + ','
          for i in range(0, len(blob), 32)]
lines += ['};', '} // namespace hyle::compute::cuda_native::fp8', '']
(args.output / 'sglang_fp8_sm89.h').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
print('cubin SHA256:', hashlib.sha256(blob).hexdigest())
