"""Compile the residual HYLE tensor shaders into the native artifact.

This build step has no runtime cache, device probe or algorithm selection.
The dependency lock authenticates inputs before CMake reaches this command.
"""

import argparse
from pathlib import Path
import re
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tensor-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', action='append', help='Compile only these named variants')
    parser.add_argument('--output-stem', default='vulkan_shaders')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.output_stem):
        parser.error('output stem must be a C++ header basename')
    args.output.mkdir(parents=True, exist_ok=True)
    variants = [
        ('cache_encode_e4', str((args.tensor_source / 'cache_encode_e4.comp').resolve()), {}, {1}),
        ('blockwise_w4a8', str((args.tensor_source / 'blockwise_w4a8.comp').resolve()), {}, {1}),
        ('blockwise_w4a8_grouped', str((args.tensor_source / 'blockwise_w4a8.comp').resolve()),
         dict(HYLE_BLOCKWISE_GROUPED='1'), {1}),
        ('static_fp8', str((args.tensor_source / 'static_fp8.comp').resolve()), {}, {1}),
        ('ct_packed', str((args.tensor_source / 'ct_packed.comp').resolve()), {}, {1}),
        ('ct_packed_grouped', str((args.tensor_source / 'ct_packed.comp').resolve()),
         dict(HYLE_CT_GROUPED='1'), {1}),
        ('gptq_v1', str((args.tensor_source / 'ct_packed.comp').resolve()),
         dict(HYLE_GPTQ_V1='1'), {1}),
        ('nvfp4', str((args.tensor_source / 'nvfp4.comp').resolve()), {}, {1}),
        ('nvfp4_grouped', str((args.tensor_source / 'nvfp4.comp').resolve()),
         dict(HYLE_NVFP4_GROUPED='1'), {1}),
        ('nvfp4_local', str((args.tensor_source / 'nvfp4.comp').resolve()),
         dict(HYLE_NVFP4_LOCAL='1'), {1}),
        ('nvfp4_grouped_local', str((args.tensor_source / 'nvfp4.comp').resolve()),
         dict(HYLE_NVFP4_GROUPED='1', HYLE_NVFP4_LOCAL='1'), {1}),
        ('mxfp4', str((args.tensor_source / 'nvfp4.comp').resolve()),
         dict(HYLE_MXFP4='1'), {1}),
        ('mxfp4_grouped', str((args.tensor_source / 'nvfp4.comp').resolve()),
         dict(HYLE_NVFP4_GROUPED='1', HYLE_MXFP4='1'), {1}),
        ('mxfp4_weight_only', str((args.tensor_source / 'nvfp4.comp').resolve()),
         dict(HYLE_MXFP4='1', HYLE_MXFP4_WEIGHT_ONLY='1'), {1}),
        ('mxfp4_grouped_weight_only', str((args.tensor_source / 'nvfp4.comp').resolve()),
         dict(HYLE_NVFP4_GROUPED='1', HYLE_MXFP4='1', HYLE_MXFP4_WEIGHT_ONLY='1'), {1}),
        ('row_cache_padded_map', str((args.tensor_source / 'row_cache_padded_map.comp').resolve()), {}, {1}),
        ('row_cache_indexer', str((args.tensor_source / 'row_cache_indexer.comp').resolve()), {}, {1}),
        ('row_cache_paged_indexer', str((args.tensor_source / 'row_cache_indexer.comp').resolve()),
         dict(HYLE_ROW_INDEXER_PAGED='1'), {1}),
        ('row_cache', str((args.tensor_source / 'row_cache.comp').resolve()), {}, {1}),
        ('sampling', str((args.tensor_source / 'sampling.comp').resolve()), {}, {1}),
        ('tensor', str((args.tensor_source / 'tensor.comp').resolve()), {}, {1}),
        ('checked_tensor', str((args.tensor_source / 'tensor.comp').resolve()),
         dict(HYLE_TENSOR_CHECKED='1'), {1}),
        ('integer_scan', str((args.tensor_source / 'integer_scan.comp').resolve()),
         {}, {1}),
        ('scatter', str((args.tensor_source / 'tensor.comp').resolve()),
         dict(HYLE_TENSOR_SCATTER='1'), {1}),
        ('normalization', str((args.tensor_source / 'tensor.comp').resolve()),
         dict(HYLE_TENSOR_NORMALIZATION='1'), {1}),
        ('concat', str((args.tensor_source / 'tensor.comp').resolve()),
         dict(HYLE_TENSOR_CONCAT='1'), {1}),
        ('reduction', str((args.tensor_source / 'tensor.comp').resolve()),
         dict(HYLE_TENSOR_REDUCTION='1'), {1}),
        ('raw16_gather', str((args.tensor_source / 'raw16_gather.comp').resolve()), {}, {1, 4433}),
    ]
    if args.variant:
        unknown = set(args.variant) - {variant[0] for variant in variants}
        if unknown:
            parser.error('unknown shader variants: ' + ', '.join(sorted(unknown)))
        variants = [variant for variant in variants if variant[0] in args.variant]
    header = ['#pragma once', '#include <cstdint>', '#include <span>',
              'namespace hyle::compute::vulkan::detail::shaders {']
    source = ['#include "' + args.output_stem + '.h"',
              'namespace hyle::compute::vulkan::detail::shaders {']
    for name, filename, defines, allowed_capabilities in variants:
        if name in ('cache_encode_e4', 'nvfp4', 'nvfp4_grouped', 'nvfp4_local', 'nvfp4_grouped_local',
                    'mxfp4', 'mxfp4_grouped', 'mxfp4_weight_only', 'mxfp4_grouped_weight_only',
                    'ct_packed', 'ct_packed_grouped', 'gptq_v1', 'static_fp8',
                    'blockwise_w4a8', 'blockwise_w4a8_grouped'):
            numerical_header = Path(__file__).resolve().parent.parent / 'compute_shader_shared' / 'nvfp4_f32.h'
            numerical = numerical_header.read_text(encoding='utf-8')
            start, finish = 'R"HYLE_SHADER(', ')HYLE_SHADER"'
            if numerical.count(start) != 1 or numerical.count(finish) != 1:
                raise ValueError('NVFP4 shared numerical source has invalid delimiters')
            numerical = numerical.split(start, 1)[1].split(finish, 1)[0]
            if name in ('cache_encode_e4', 'ct_packed', 'ct_packed_grouped', 'gptq_v1', 'static_fp8',
                        'blockwise_w4a8', 'blockwise_w4a8_grouped'):
                typed_header = numerical_header.with_name('ct_packed.h')
                typed = typed_header.read_text(encoding='utf-8')
                if typed.count(start) != 1 or typed.count(finish) != 1:
                    raise ValueError('CT shared numerical source has invalid delimiters')
                numerical += '\n' + typed.split(start, 1)[1].split(finish, 1)[0]
            shader = Path(filename).read_text(encoding='utf-8')
            marker = ('// HYLE_CACHE_E4_F32_SOURCE' if name == 'cache_encode_e4' else
                      '// HYLE_BLOCKWISE_W4A8_F32_SOURCE' if name.startswith('blockwise_w4a8') else
                      '// HYLE_STATIC_FP8_F32_SOURCE' if name == 'static_fp8' else
                      '// HYLE_CT_F32_SOURCE' if name in ('ct_packed', 'ct_packed_grouped', 'gptq_v1') else '// HYLE_NVFP4_F32_SOURCE')
            if shader.count(marker) != 1:
                raise ValueError('NVFP4 shader requires one shared numerical marker')
            generated = args.output / (name + '.generated.comp')
            generated.write_bytes(shader.replace(marker, numerical).encode('utf-8'))
            filename = str(generated.resolve())
        output = args.output / (name + '.spv')
        command = [args.compiler, '-V', '--target-env', 'vulkan1.2',
                   '--preamble-text', '#extension GL_GOOGLE_include_directive : require\n',
                   '-I' + str(args.source), '-I' + str(args.tensor_source), '-S', 'comp']
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
    for filename, lines in [(args.output_stem + '.h', header), (args.output_stem + '.cc', source)]:
        (args.output / filename).write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
