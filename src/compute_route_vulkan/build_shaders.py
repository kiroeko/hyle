"""Compile the residual HYLE tensor shaders into the native artifact.

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
    variants = [
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
