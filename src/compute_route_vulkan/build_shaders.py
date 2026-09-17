"""Compile admitted ggml F32 and HYLE tensor shaders into the native artifact.

This build step has no runtime cache, device probe or algorithm selection.
The dependency lock authenticates inputs before CMake reaches this command.
"""

import argparse
from pathlib import Path
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tensor-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    common = dict(DATA_A_F32='1', FLOAT_TYPE='float', FLOAT_TYPEV2='vec2',
                  FLOAT_TYPEV4='vec4', FLOAT_TYPEV8='mat2x4', B_TYPE='float',
                  B_TYPEV2='vec2', B_TYPEV4='vec4', D_TYPE='float')
    matrix = dict(common, LOAD_VEC_A='4', LOAD_VEC_B='4', B_TYPE='vec4',
                  B_TYPE_SCALAR='float', ACC_TYPE='float', ACC_TYPEV2='vec2')
    variants = [
        ('mm_f32', 'mul_mm.comp', matrix, {1}),
        ('mv_f32', 'mul_mat_vec.comp', common, {1}),
        ('mv_f32_hybrid', 'mul_mat_vec.comp', dict(common, USE_SUBGROUP_ADD='1'),
         {1, 61, 63}),
        ('mv_f32_subgroup', 'mul_mat_vec.comp',
         dict(common, USE_SUBGROUP_ADD_NO_SHMEM='1'), {1, 61, 63}),
        ('swiglu_f32', 'swiglu.comp', dict(A_TYPE='float', D_TYPE='float'), {1}),
        ('copy_f32', 'copy.comp', dict(A_TYPE='float', D_TYPE='float', DATA_A_F32='1'), {1}),
        ('tensor', str((args.tensor_source / 'tensor.comp').resolve()), {}, {1}),
    ]
    header = ['#pragma once', '#include <cstdint>', '#include <span>',
              'namespace hyle::compute::vulkan::detail::shaders {']
    source = ['#include "vulkan_shaders.h"',
              'namespace hyle::compute::vulkan::detail::shaders {']
    for name, filename, defines, allowed_capabilities in variants:
        output = args.output / (name + '.spv')
        command = [args.compiler, '-V', '--target-env', 'vulkan1.2',
                   '--preamble-text', '#extension GL_GOOGLE_include_directive : require\n',
                   '-I' + str(args.source), '-S', 'comp']
        command.extend('-D' + key + '=' + value for key, value in defines.items())
        command.extend(['-o', str(output), str(args.source / filename)])
        subprocess.run(command, check=True, timeout=60)
        raw = output.read_bytes()
        if len(raw) < 20 or len(raw) % 4:
            raise ValueError(name + ': malformed SPIR-V word stream')
        words = struct.unpack('<' + 'I' * (len(raw) // 4), raw)
        if words[0] != 0x07230203 or words[1] > 0x00010500:
            raise ValueError(name + ': expected SPIR-V <= 1.5')
        at = 5
        while at < len(words):
            size, opcode = words[at] >> 16, words[at] & 65535
            if not size or at + size > len(words):
                raise ValueError(name + ': invalid instruction length')
            if opcode == 17 and (size != 2 or words[at + 1] not in allowed_capabilities):
                raise ValueError(name + ': shader requires an unadmitted capability')
            at += size
        header.append('std::span<const std::uint32_t> ' + name + '() noexcept;')
        source.append('namespace { constexpr std::uint32_t ' + name + '_words[] = {')
        for offset in range(0, len(words), 8):
            source.append('  ' + ', '.join(f'0x{word:08x}u' for word in words[offset:offset + 8]) + ',')
        source.append('}; }')
        source.append('std::span<const std::uint32_t> ' + name + '() noexcept { return ' + name + '_words; }')
    header.append('} // namespace hyle::compute::vulkan::detail::shaders')
    source.append('} // namespace hyle::compute::vulkan::detail::shaders')
    for filename, lines in [('vulkan_shaders.h', header), ('vulkan_shaders.cc', source)]:
        (args.output / filename).write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
