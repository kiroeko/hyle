"""Rebuild original DeepSeek quantization programs; offline development only."""
from pathlib import Path
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ['TVM_FFI_DISABLE_TORCH_C_DLPACK'] = '1'
if os.name == 'nt':
    ctypes.windll.kernel32.SetErrorMode(1 | 2 | 0x8000)
    class HiddenPopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            kwargs['creationflags'] = kwargs.get('creationflags', 0) | subprocess.CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)
    subprocess.Popen = HiddenPopen

import tilelang

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--nvcc', default='nvcc')
args = parser.parse_args()
assert tilelang.__version__ == '0.1.14'
root = Path(__file__).resolve().parent
args.output.mkdir(parents=True, exist_ok=True)
provenance = json.loads((root/'provenance.json').read_text(encoding='utf-8'))
source = (root/'upstream_kernel.py').read_text(encoding='utf-8')
assert hashlib.sha256(source.encode()).hexdigest() == provenance['upstream_sha256']
# Preserve the original helpers and two kernels byte-for-byte; later GEMM,
# attention and torch launch wrappers are not part of the native program.
module_path = args.output/'original_quant.py'
module_path.write_text(source[:source.index('@tilelang.jit(pass_configs=pass_configs)\ndef fp8_gemm_kernel')], encoding='utf-8', newline='\n')
spec = importlib.util.spec_from_file_location('deepseek_original_quant', module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
compiler = Path(tilelang.__file__).resolve().parent
lines = ['#pragma once', '// Original DeepSeek V4; TileLang0.1.14 + CUDA13.3 SM89 AOT.']
receipt = []
for job in provenance['programs']:
    name, group = job['kernel'], job['group']
    artifact = job.get('artifact', name)
    kwargs = {'block_size': group, 'scale_dtype': 'float32', 'inplace': True}
    if name == 'act_quant_kernel':
        kwargs['round_scale'] = True
    prim = getattr(module, name).get_tir(group, **kwargs)
    target = tilelang.tvm.target.Target({'kind': 'cuda', 'arch': 'sm_89'})
    with target, tilelang.tvm.transform.PassContext(config=module.pass_configs):
        result = tilelang.lower(prim, target=target, enable_host_codegen=False, enable_device_compile=False)
    cu = args.output/(artifact+'.cu')
    cu.write_text(result.kernel_source, encoding='utf-8', newline='\n')
    blob_path = args.output/(artifact+'.cubin')
    command = [args.nvcc, '--cubin', '-arch=sm_89', '-std=c++17', '-DENABLE_BF16',
               '-I'+str(compiler/'src'), '-I'+str(compiler/'3rdparty/cutlass/include'),
               str(cu), '-o', str(blob_path)]
    if os.name == 'nt':
        command.insert(4, '-Xcompiler=/Zc:preprocessor,/utf-8')
    with (args.output/(artifact+'-compile.log')).open('wb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    blob = blob_path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    receipt.append({'kernel': name, 'group': group, 'sha256': digest, 'matches_pinned': digest == job['sha256']})
    lines.append('alignas(16) inline constexpr unsigned char '+job['symbol']+'[] = {')
    for offset in range(0, len(blob), 24):
        lines.append('  '+','.join(f'0x{x:02x}' for x in blob[offset:offset+24])+',')
    lines.append('};')
(args.output/'deepseek_quantization_sm89.h').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
(args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
assert all(row['matches_pinned'] for row in receipt), receipt
