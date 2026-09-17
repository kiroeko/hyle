"""Compile unchanged, independently pinned ggml inputs for the R6 comparator."""
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
        raise RuntimeError('ggml reference must be a clean checkout at the declared pin')
    source = args.peer / 'src/ggml-vulkan/vulkan-shaders'
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def check(name):
        if name in inputs:
            return
        blob = subprocess.check_output(['git', 'show', PIN + ':src/ggml-vulkan/vulkan-shaders/' + name], cwd=args.peer)
        local = (source / name).read_bytes()
        if local.replace(b'\r\n', b'\n') != blob.replace(b'\r\n', b'\n'):
            raise RuntimeError('reference shader input changed: ' + name)
        inputs[name] = hashlib.sha256(blob).hexdigest()
        for include in re.findall(r'^\s*#include\s+"([^"]+)"', blob.decode(), re.M):
            check(include)

    # ggml shader generator: non-cooperative, no FLOAT16, F32 storage/reduction.
    common = {'DATA_A_F32': '1', 'FLOAT_TYPE': 'float', 'FLOAT_TYPEV2': 'vec2',
              'FLOAT_TYPEV4': 'vec4', 'FLOAT_TYPEV8': 'mat2x4', 'B_TYPE': 'float',
              'B_TYPEV2': 'vec2', 'B_TYPEV4': 'vec4', 'D_TYPE': 'float'}
    variants = [
        ('matrix', 'mul_mm.comp', dict(common, LOAD_VEC_A='4', LOAD_VEC_B='4',
         B_TYPE='vec4', B_TYPE_SCALAR='float', ACC_TYPE='float', ACC_TYPEV2='vec2')),
        ('vector_shared', 'mul_mat_vec.comp', common),
        ('vector_subgroup', 'mul_mat_vec.comp', dict(common, USE_SUBGROUP_ADD_NO_SHMEM='1')),
        ('vector_hybrid', 'mul_mat_vec.comp', dict(common, USE_SUBGROUP_ADD='1')),
        ('gate', 'swiglu.comp', {'A_TYPE': 'float', 'D_TYPE': 'float'}),
    ]
    header = ['#pragma once', '#include <cstdint>',
              'namespace hyle::bench::vulkan::spirv {']
    outputs = []
    for name, filename, definitions in variants:
        check(filename)
        path = args.output / (name + '.spv')
        command = [args.compiler, '-V', '--target-env', 'vulkan1.2',
                   '--preamble-text', '#extension GL_GOOGLE_include_directive : require\n',
                   '-I' + str(source), '-S', 'comp']
        command += ['-D' + key + '=' + value for key, value in definitions.items()]
        command += ['-o', str(path), str(source / filename)]
        subprocess.run(command, check=True, timeout=60)
        raw = path.read_bytes()
        words = struct.unpack('<' + 'I' * (len(raw) // 4), raw)
        header.append('inline constexpr std::uint32_t ' + name + '[] = {')
        for offset in range(0, len(words), 8):
            header.append(','.join('0x%08xu' % word for word in words[offset:offset + 8]) + ',')
        header.append('};')
        outputs.append({'name': name, 'command': command, 'sha256': hashlib.sha256(raw).hexdigest()})
    header.append('}')
    (args.output / 'reference_spirv.h').write_bytes(('\n'.join(header) + '\n').encode())
    (args.output / 'identity.json').write_bytes((json.dumps(
        {'revision': revision, 'source_blobs_sha256': inputs, 'outputs': outputs}, indent=2) + '\n').encode())


if __name__ == '__main__':
    main()
