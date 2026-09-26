from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode = True

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
assert triton.__version__=='3.8.0'
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('deepseek_original',root/'original_kernel.py')
module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
args.output.mkdir(parents=True,exist_ok=True)
lines=['#pragma once','#include <cstddef>','// Original DeepSeek block128 decode; reproducible offline Triton3.8 SM89 artifacts.']
receipt=[]
for job,label in zip(json.loads((root/'compilation.json').read_text())['kernels'],
                     ['kDeepSeekDecodeBf16','kDeepSeekDecodeF32']):
    program=triton.compile(ASTSource(fn=module.weight_dequant_kernel,
        signature=job['signature'],constexprs=job['constexprs']),
        target=GPUTarget('cuda',89,32),options=job['options'])
    blob=program.asm['cubin']
    (args.output/(job['label']+'.cubin')).write_bytes(blob)
    digest=hashlib.sha256(blob).hexdigest()
    receipt.append({'label':label,'sha256':digest,'matches_pinned':digest==job['sha256']})
    lines.append(f'alignas(16) inline constexpr unsigned char {label}[] = {{')
    for offset in range(0,len(blob),24):
        lines.append('  '+','.join(f'0x{x:02x}' for x in blob[offset:offset+24])+',')
    lines.append('};')
(args.output/'deepseek_fp8_sm89.h').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
(args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
assert all(item['matches_pinned'] for item in receipt),receipt
