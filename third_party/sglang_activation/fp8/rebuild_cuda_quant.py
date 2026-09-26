"""Maintainer-only CUDA 13.3 SM89 AOT; never part of CMake or execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--nvcc', default='nvcc')
args = parser.parse_args()
root = Path(__file__).resolve().parent
metadata = json.loads((root/'cuda-quant-compilation.json').read_text())
source = root/'original_cuda_quant.cu'
assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata['source_sha256']
assert hashlib.sha256((root/'flashinfer/vec_dtypes.cuh').read_bytes()).hexdigest() == metadata['vec_header_sha256']
args.output.mkdir(parents=True, exist_ok=True)
target = args.output/'original_quant.cubin'
command = [args.nvcc, '-std=c++20', '-arch=sm_89', '-DFLASHINFER_ENABLE_BF16',
           '-DFLASHINFER_ENABLE_F16', '-I'+str(root), str(source), '--cubin', '-o', str(target)]
if os.name == 'nt':
    command += ['-Xcompiler=/utf-8']
subprocess.run(command, check=True,
               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
blob = target.read_bytes()
lines = ['// Generated from original SGLang CUDA group quantization. See cuda-quant-compilation.json.',
         '#pragma once', 'namespace hyle::compute::cuda_native::fp8 {',
         '// SHA256 cubin: '+hashlib.sha256(blob).hexdigest(),
         'alignas(16) inline constexpr unsigned char kSglangFp8QuantizeCuda[] = {']
lines += ['  '+','.join(f'0x{x:02x}' for x in blob[i:i+32])+',' for i in range(0,len(blob),32)]
lines += ['};']
for symbol in metadata['symbols']:
    label = 'Bf16' if '__nv_bfloat16' in symbol else 'F32'
    lines += [f'inline constexpr char kSglangFp8Quantize{label}Name[] = "{symbol}";']
lines += ['} // namespace hyle::compute::cuda_native::fp8', '']
(args.output/'sglang_fp8_quantize_cuda_sm89.h').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
print(hashlib.sha256(blob).hexdigest())
