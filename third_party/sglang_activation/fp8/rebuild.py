"""Maintainer-only AOT regeneration; never invoked by product CMake or runtime."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--triton-path', type=Path)
parser.add_argument('--gemm-only', action='store_true',
                    help='Regenerate only the two GEMMs; preserve the packaged power-of-two quantizer bytes')
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
raw = original.read_bytes()
assert hashlib.sha256(raw).hexdigest() == identity['module_sha256']
adaptation = identity['adaptation']
source = raw.decode('utf-8')
assert adaptation['occurrences'] == 1
assert source.count(adaptation['before']) == 1, 'Expected exactly one vendor dot expression'
adapted = source.replace(adaptation['before'], adaptation['after'], 1)
adapted_sha256 = hashlib.sha256(adapted.encode('utf-8')).hexdigest()
assert adapted_sha256 == adaptation['adapted_module_sha256']
if args.output.resolve() == root:
    parser.error('--output must differ from the source directory; copy regenerated headers after review')
args.output.mkdir(parents=True, exist_ok=True)
adapted_path = args.output / 'adapted_kernels.py'
adapted_path.write_text(adapted, encoding='utf-8', newline='\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = load_module('hyle_sglang_fp8_adapted', adapted_path)
jobs = json.loads((root/'compilation.json').read_text())['kernels']
jobs += json.loads((root/'dynamic-compilation.json').read_text())['kernels']
jobs = [job for job in jobs if job['label'] not in ('quantize', 'quantize_bf16')]
# Ordinary quantization uses the original CUDA kernel after F38 qualification;
# its separate maintainer generator is rebuild_cuda_quant.py. Its header is
# neither read nor written here; all original quantizer policies remain intact.
names = {'matmul':'kSglangFp8Sm89', 'matmul_bf16':'kSglangFp8MatmulBf16',
         'quantize_bf16_pow2':'kSglangFp8QuantizeBf16Pow2'}
assert {job['label'] for job in jobs} == set(names)
assert len(jobs) == len(names)
records = []
preserved_quantizer = None
if args.gemm_only:
    packaged_path = root/'sglang_fp8_dynamic_sm89.h'
    packaged_raw = packaged_path.read_bytes()
    packaged = packaged_raw.decode('utf-8')
    arrays = list(re.finditer(r'alignas\(16\) inline constexpr unsigned char '
        r'kSglangFp8QuantizeBf16Pow2\[\] = \{.*?\n\};', packaged, re.S))
    assert len(arrays) == 1, 'Expected exactly one packaged power-of-two quantizer'
    array = arrays[0]
    blob = bytes(int(value, 16) for value in re.findall(r'0x([0-9a-fA-F]{2})', array[0]))
    sha256 = hashlib.sha256(blob).hexdigest()
    prefix = packaged[:array.start()]
    hashes = re.findall(r'// SHA256 cubin: ([0-9a-f]{64})', prefix)
    assert hashes and hashes[-1] == sha256, 'Packaged power-of-two cubin identity differs'
    preserved_quantizer = ['// Preserved packaged power-of-two quantizer; no recompilation.',
                           '// SHA256 cubin: '+sha256, array[0]]
    (args.output/'quantize_bf16_pow2.cubin').write_bytes(blob)
    job = next(job for job in jobs if job['label'] == 'quantize_bf16_pow2')
    records.append({'label': job['label'], 'sha256': sha256, 'bytes': len(blob),
                    'preserved': True, 'source_header_sha256': hashlib.sha256(packaged_raw).hexdigest(),
                    'signature': job['signature'], 'constexprs': job['constexprs'],
                    'options': job['options'], 'metadata': job['metadata'],
                    'metadata_origin': 'packaged dynamic-compilation.json; not recompiled'})
    jobs = [job for job in jobs if job['label'] != 'quantize_bf16_pow2']
original_module = None
if not args.gemm_only:
    original_module = load_module('hyle_sglang_fp8_original', original)
headers = {}
for job in jobs:
    label = job['label']
    kernel_module = original_module if label == 'quantize_bf16_pow2' else module
    compiled = triton.compile(ASTSource(fn=getattr(kernel_module,job['metadata']['name']),
        signature=job['signature'], constexprs=job['constexprs']),
        target=GPUTarget('cuda',89,32),options=job['options'])
    blob = compiled.asm['cubin']
    (args.output/(label+'.cubin')).write_bytes(blob)
    (args.output/(label+'.ptx')).write_text(compiled.asm['ptx'],encoding='utf-8',newline='\n')
    metadata = compiled.metadata._asdict()
    records.append({'label': label, 'sha256': hashlib.sha256(blob).hexdigest(),
                    'bytes': len(blob), 'signature': job['signature'],
                    'constexprs': job['constexprs'], 'options': job['options'],
                    'metadata': metadata})
    header = 'sglang_fp8_sm89.h' if label == 'matmul' else 'sglang_fp8_dynamic_sm89.h'
    lines = headers.setdefault(header, [
        '// Generated from SGLang 06992c92 with Triton 3.8.0. See source.json.',
        '// GEMM adapts exact E4-to-BF16 register tiles before its F32 dot; quantizers are unchanged.',
        '#pragma once','namespace hyle::compute::cuda_native::fp8 {'])
    if label in ('matmul', 'matmul_bf16'):
        shared = int(metadata['shared'])
        assert 0 <= shared <= 0xffffffff
        lines += [f'inline constexpr unsigned int {names[label]}SharedBytes = {shared};']
    lines += ['// SHA256 cubin: '+hashlib.sha256(blob).hexdigest(),
              f'alignas(16) inline constexpr unsigned char {names[label]}[] = {{']
    lines += ['  '+','.join(f'0x{x:02x}' for x in blob[i:i+32])+',' for i in range(0,len(blob),32)]
    lines += ['};']
    print(label, hashlib.sha256(blob).hexdigest(), 'shared='+str(metadata['shared']))
if preserved_quantizer:
    headers['sglang_fp8_dynamic_sm89.h'] += preserved_quantizer
for name,lines in headers.items():
    lines += ['} // namespace hyle::compute::cuda_native::fp8','']
    (args.output/name).write_text('\n'.join(lines),encoding='utf-8',newline='\n')
record = {'source_sha256': identity['module_sha256'],
          'adapted_module_sha256': adapted_sha256, 'adaptation': adaptation,
          'compiler': triton.__version__, 'target': {'backend': 'cuda', 'arch': 89, 'warp_size': 32},
          'gemm_only': args.gemm_only, 'kernels': records}
(args.output/'compilation.json').write_text(json.dumps(record, indent=2, default=str)+'\n',
                                            encoding='utf-8', newline='\n')
