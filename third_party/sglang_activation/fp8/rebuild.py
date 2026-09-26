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
args.output.mkdir(parents=True, exist_ok=True)
jobs = json.loads((root/'compilation.json').read_text())['kernels']
jobs += json.loads((root/'dynamic-compilation.json').read_text())['kernels']
jobs = [job for job in jobs if job['label'] not in ('quantize', 'quantize_bf16')]
# Ordinary quantization uses the original CUDA kernel after F38 qualification;
# its separate maintainer generator is rebuild_cuda_quant.py.
names = {'matmul':'kSglangFp8Sm89', 'quantize':'kSglangFp8QuantizeF32',
         'matmul_bf16':'kSglangFp8MatmulBf16', 'quantize_bf16':'kSglangFp8QuantizeBf16',
         'quantize_bf16_pow2':'kSglangFp8QuantizeBf16Pow2'}
headers = {}
for job in jobs:
    compiled = triton.compile(ASTSource(fn=getattr(module,job['metadata']['name']),
        signature=job['signature'], constexprs=job['constexprs']),
        target=GPUTarget('cuda',89,32),options=job['options'])
    blob = compiled.asm['cubin']
    label = job['label']
    (args.output/(label+'.cubin')).write_bytes(blob)
    (args.output/(label+'.ptx')).write_text(compiled.asm['ptx'],encoding='utf-8',newline='\n')
    header = 'sglang_fp8_sm89.h' if label == 'matmul' else 'sglang_fp8_dynamic_sm89.h'
    lines = headers.setdefault(header, [
        '// Generated from unchanged SGLang 06992c92 with Triton 3.8.0. See source.json.',
        '#pragma once','namespace hyle::compute::cuda_native::fp8 {'])
    lines += ['// SHA256 cubin: '+hashlib.sha256(blob).hexdigest(),
              f'alignas(16) inline constexpr unsigned char {names[label]}[] = {{']
    lines += ['  '+','.join(f'0x{x:02x}' for x in blob[i:i+32])+',' for i in range(0,len(blob),32)]
    lines += ['};']
    print(label,hashlib.sha256(blob).hexdigest())
for name,lines in headers.items():
    lines += ['} // namespace hyle::compute::cuda_native::fp8','']
    (args.output/name).write_text('\n'.join(lines),encoding='utf-8',newline='\n')
