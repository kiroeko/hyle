"""Compile the original pinned ggml full-F32 flash and its charged adapters."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

PIN = '456172ec733a135778adcd32d00e576a58232e45'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--peer', type=Path, required=True)
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.peer, text=True).strip()
    if revision != PIN or subprocess.check_output(['git', 'status', '--porcelain'], cwd=args.peer):
        raise RuntimeError('reference requires the clean declared ggml pin')
    source = args.peer / 'src/ggml-vulkan/vulkan-shaders'
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def check(name):
        if name in inputs:
            return
        blob = subprocess.check_output(['git', 'show', PIN + ':src/ggml-vulkan/vulkan-shaders/' + name], cwd=args.peer)
        if (source / name).read_bytes().replace(b'\r\n', b'\n') != blob.replace(b'\r\n', b'\n'):
            raise RuntimeError('reference shader differs: ' + name)
        inputs[name] = hashlib.sha256(blob).hexdigest()
        for include in re.findall(r'^\s*#include\s+"([^"]+)"', blob.decode(), re.M):
            check(include)

    float_math = {'FLOAT_TYPE': 'float', 'FLOAT_TYPEV2': 'vec2', 'FLOAT_TYPEV4': 'vec4',
                  'ACC_TYPE': 'float', 'ACC_TYPEV2': 'vec2', 'ACC_TYPEV4': 'vec4',
                  'DATA_A_IQ4_NL': '1', 'Q_TYPE': 'float', 'D_TYPE': 'float', 'D_TYPEV4': 'vec4'}
    variants = [('flash', 'flash_attn.comp', float_math),
                ('mask_opt', 'flash_attn_mask_opt.comp', {}),
                ('split_reduce', 'flash_attn_split_k_reduce.comp', {}),
                ('fill', 'fill.comp', {'D_TYPE': 'float'}),
                ('mask', 'diag_mask_inf.comp', {'A_TYPE': 'float', 'D_TYPE': 'float'}),
                ('half', 'copy.comp', {'A_TYPE': 'float', 'D_TYPE': 'float16_t'}),
                ('query_float', 'copy.comp', {'A_TYPE': 'float16_t', 'D_TYPE': 'float'})]
    header = ['#pragma once', '#include <cstdint>', 'namespace hyle::bench::resident::spirv {']
    outputs = []
    for name, filename, definitions in variants:
        check(filename)
        path = args.output / (name + '.spv')
        command = [args.compiler, '-V', '--target-env', 'vulkan1.2',
                   '--preamble-text', '#extension GL_GOOGLE_include_directive : require\n',
                   '-I' + str(source), '-S', 'comp']
        command += ['-D' + key + '=' + value for key, value in definitions.items()]
        command += ['-o', str(path), str(source / filename)]
        subprocess.run(command, check=True)
        raw = path.read_bytes()
        words = struct.unpack('<' + 'I' * (len(raw) // 4), raw)
        header += ['inline constexpr std::uint32_t ' + name + '[] = {']
        header += [','.join('0x%08xu' % word for word in words[offset:offset + 8]) + ','
                   for offset in range(0, len(words), 8)]
        header += ['};']
        outputs.append({'name': name, 'command': command, 'sha256': hashlib.sha256(raw).hexdigest()})
    header += ['}']
    (args.output / 'reference_spirv.h').write_bytes(('\n'.join(header) + '\n').encode())
    (args.output / 'identity.json').write_bytes((json.dumps(
        {'revision': revision, 'source_blobs_sha256': inputs, 'outputs': outputs}, indent=2) + '\n').encode())


if __name__ == '__main__':
    main()
