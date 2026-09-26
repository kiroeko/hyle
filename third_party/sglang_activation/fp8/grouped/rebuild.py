"""Maintainer-only AOT. Normal builds/use never run Python or Triton.

Run --stage gemm on the pinned Triton 3.8 host and --stage align with CUDA
13.3 nvcc, collecting cubins in one --output directory; --stage header
checks their identities and emits the native header from that directory.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['gemm', 'align', 'header'], required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--triton-path', type=Path)
parser.add_argument('--nvcc', default='nvcc')
args = parser.parse_args()
root = Path(__file__).resolve().parent
args.output.mkdir(parents=True, exist_ok=True)
source = json.loads((root/'source.json').read_bytes())
align = json.loads((root/'align-compilation.json').read_bytes())
jobs = json.loads((root/'compilation.json').read_bytes())['kernels']
if args.stage == 'gemm':
    if args.triton_path:
        sys.path.insert(0, str(args.triton_path))
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    assert triton.__version__ == '3.8.0'
    original = root/'original_kernels.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source['module_sha256']
    spec = importlib.util.spec_from_file_location('grouped_original', original)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for job in jobs:
        constants = dict(job['constexprs'])
        constants['compute_type'] = tl.float32 if constants['compute_type'] == 'fp32' else tl.bfloat16
        compiled = triton.compile(ASTSource(fn=module.fused_moe_kernel,
            signature=job['signature'], constexprs=constants), target=GPUTarget('cuda',89,32),
            options=job['options'])
        blob = compiled.asm['cubin']
        assert hashlib.sha256(blob).hexdigest() == job['cubin_sha256']
        (args.output/(job['label']+'.cubin')).write_bytes(blob)
elif args.stage == 'align':
    original = root/'checked_original_align.cu'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == align['adapted_sha256']
    command = [args.nvcc, '-std=c++20', '-arch=sm_89', str(original), '--cubin',
               '-o', str(args.output/'checked_original_align.cubin')]
    if os.name == 'nt': command += ['-Xcompiler=/utf-8']
    subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
else:
    names = {'grouped_fp32_top1':'kGroupedF32Top1', 'grouped_fp32_top8':'kGroupedF32Top8',
             'grouped_bf16_top1':'kGroupedBf16Top1', 'grouped_bf16_top8':'kGroupedBf16Top8',
             'checked_original_align':'kCheckedAlign'}
    hashes = {job['label']:job['cubin_sha256'] for job in jobs}
    hashes['checked_original_align'] = align['cubin_sha256']
    lines = ['// Offline SGLang 06992c92 / Triton 3.8.0 / CUDA 13.3. See compilation records.',
             '#pragma once', 'namespace hyle::compute::cuda_native::grouped_fp8 {']
    for label,name in names.items():
        blob = (args.output/(label+'.cubin')).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        assert digest == hashes[label], label
        lines += ['// SHA256 cubin: '+digest, f'alignas(16) inline constexpr unsigned char {name}[] = {{']
        lines += ['  '+','.join(f'0x{x:02x}' for x in blob[i:i+32])+',' for i in range(0,len(blob),32)]
        lines += ['};']
    for symbol in align['symbols']:
        name = 'kAlignSmallName' if 'small_batch' in symbol else 'kAlignSortName' if 'count_and_sort' in symbol else 'kAlignName'
        lines += [f'inline constexpr char {name}[] = "{symbol}";']
    lines += ['} // namespace hyle::compute::cuda_native::grouped_fp8', '']
    (args.output/'sglang_grouped_fp8_sm89.h').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
