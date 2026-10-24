"""Maintainer-only CUDA 13.3 SM89 AOT; never part of CMake or execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--nvcc', default='nvcc')
parser.add_argument('--ccbin')
parser.add_argument('--cuobjdump')
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
if args.ccbin:
    command += ['-ccbin', args.ccbin]
if os.name == 'nt':
    command += ['-Xcompiler=/utf-8']
flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
record = {'command': command, 'source_sha256': metadata['source_sha256']}
(args.output/'cuda-quant-build.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
compiled = subprocess.run(command, capture_output=True, text=True, creationflags=flags)
(args.output/'nvcc.stdout.log').write_text(compiled.stdout, encoding='utf-8')
(args.output/'nvcc.stderr.log').write_text(compiled.stderr, encoding='utf-8')
record['nvcc_exit_code'] = compiled.returncode
(args.output/'cuda-quant-build.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
compiled.check_returncode()
blob = target.read_bytes()
cuobjdump = args.cuobjdump or str(Path(args.nvcc).with_name('cuobjdump.exe' if os.name == 'nt' else 'cuobjdump'))
symbol_command = [cuobjdump, '--dump-elf-symbols', str(target)]
symbols = subprocess.run(symbol_command, capture_output=True, text=True, creationflags=flags)
(args.output/'cuda-quant-symbols.txt').write_text(symbols.stdout, encoding='utf-8')
record['symbol_command'] = symbol_command
record['symbols_exit_code'] = symbols.returncode
symbols.check_returncode()
entries = []
for symbol in re.findall(r'STT_FUNC\s+STB_GLOBAL\s+STO_ENTRY\s+(\S+)', symbols.stdout):
    assert symbol.startswith('_Z33per_token_group_quant_8bit_kernelI'), symbol
    match = re.search(r'Lb([01])EEv', symbol)
    assert match, symbol
    policy = 'multiply_reciprocal' if match[1] == '1' else 'divide'
    dtype = 'bf16' if '__nv_bfloat16' in symbol else 'f32'
    entries.append({'policy': policy, 'dtype': dtype, 'symbol': symbol})
keys = [entry['policy']+':'+entry['dtype'] for entry in entries]
assert len(entries) == 4 and len(set(keys)) == 4
assert set(keys) == set(metadata['expected_entry_keys'])
entries.sort(key=lambda entry: (entry['policy'], entry['dtype']))
labels = {('divide', 'bf16'): 'kSglangFp8QuantizeBf16Name',
          ('divide', 'f32'): 'kSglangFp8QuantizeF32Name',
          ('multiply_reciprocal', 'bf16'): 'kSglangFp8QuantizeMultiplyReciprocalBf16Name',
          ('multiply_reciprocal', 'f32'): 'kSglangFp8QuantizeMultiplyReciprocalF32Name'}
lines = ['// Generated from SGLang CUDA group quantization and the explicit Hyle scale policy.',
         '// See cuda-quant-compilation.json for adapted source and exact entry identities.',
         '#pragma once', 'namespace hyle::compute::cuda_native::fp8 {',
         '// SHA256 cubin: '+hashlib.sha256(blob).hexdigest(),
         'alignas(16) inline constexpr unsigned char kSglangFp8QuantizeCuda[] = {']
lines += ['  '+','.join(f'0x{x:02x}' for x in blob[i:i+32])+',' for i in range(0,len(blob),32)]
lines += ['};']
for entry in entries:
    label = labels[entry['policy'], entry['dtype']]
    lines += [f'inline constexpr char {label}[] = "{entry["symbol"]}";']
lines += ['} // namespace hyle::compute::cuda_native::fp8', '']
(args.output/'sglang_fp8_quantize_cuda_sm89.h').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
metadata['symbols'] = [entry['symbol'] for entry in entries]
metadata['entry_points'] = entries
metadata['launch_argument_count'] = 11
metadata['cubin_sha256'] = hashlib.sha256(blob).hexdigest()
(args.output/'cuda-quant-compilation.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8', newline='\n')
record['entry_points'] = entries
record['cubin_sha256'] = metadata['cubin_sha256']
(args.output/'cuda-quant-build.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(hashlib.sha256(blob).hexdigest())
