"""Materialize the pinned original general Q8 Metal peer, never product code.

The embedding order follows the original ggml Metal CMake recipe. General
attention arithmetic, padding and mask-block kernels stay byte-for-byte equal
after newline normalization. Only explicit exports are restricted; 64/128 is
a disclosed additional instance of the original independent-DK/DV template.
This does not compile, execute or qualify that instance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

REVISION = '456172ec733a135778adcd32d00e576a58232e45'
INPUTS = {
    'src/ggml-common.h': '0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c',
    'src/ggml-metal/kernels/common.h': '525d4d7f9dbdaef3702195be92235ba2cc7cc86d5c97101d374a52b06117215e',
    'src/ggml-metal/kernels/dequantize.h': '2dd5d69e65deb3718dd01a7e04d6ce529f008955fecfb1d1d8abab3245e43d25',
    'src/ggml-metal/ggml-metal-impl.h': '53cb45725c225093fdfce2a2011495ff9c7bd97c82eaf57364b5cc174ce1e1f8',
    'src/ggml-metal/kernels/fa.metal': '88939f93ec5a38bb6a860dffee60d01885de65ca4464d87c14c31bbc511fda64',
    'src/ggml-metal/CMakeLists.txt': '0f042e911c804fbae3dabefb5f5446bbeb39ac3148ad1e5f8b0310990a4941f8',
}
EXPORTS = (
    'kernel_flash_attn_ext_q8_0_dk64_dv64',
    'kernel_flash_attn_ext_q8_0_dk128_dv128',
    'kernel_flash_attn_ext_q8_0_dk256_dv256',
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def materialize(original, output):
    sources = {}
    for name, expected in INPUTS.items():
        raw = (original / name).read_bytes()
        if digest(raw) != expected:
            raise ValueError('original peer identity mismatch: ' + name)
        sources[name] = raw.decode('utf-8').replace('\r\n', '\n')

    fa = sources['src/ggml-metal/kernels/fa.metal']
    # Do not include the vector route, its split/reduce, or its dispatch policy.
    marker = '#undef FA_TYPES_F32\n'
    if fa.count(marker) != 2:
        raise ValueError('original general/vector boundary changed')
    general = fa[:fa.index(marker) + len(marker)]
    declarations = re.findall(r'^template \[\[host_name\("([^"]+)"[^\n]*$',
                              general, flags=re.MULTILINE)
    if any(declarations.count(name) != 1 for name in EXPORTS):
        raise ValueError('original selected exports missing or duplicated')
    lines = general.splitlines(keepends=True)
    original_exports = [line for line in lines if line.startswith('template [[host_name(')]
    selected = [line for line in original_exports if any('"' + name + '"' in line for name in EXPORTS)]
    base = next(line for line in selected if '"' + EXPORTS[0] + '"' in line)
    rectangular = base.replace(EXPORTS[0], 'kernel_flash_attn_ext_q8_0_dk64_dv128')
    rectangular, count = re.subn(r', 64,\s+64>;\n$', ', 64, 128>;\n', rectangular)
    if count != 1:
        raise ValueError('original independent-DK/DV instantiation changed')

    prefix = ''.join(line for line in lines if not line.startswith('template [[host_name('))
    # Insert before the original FA_TYPES macros are undefined.
    at = prefix.index('#undef FA_TYPES\n')
    general = prefix[:at] + ''.join(selected) + rectangular + prefix[at:]
    # The retained templates/pad/mask kernels are the exact original text.
    stripped = ''.join(line for line in general.splitlines(keepends=True)
                       if not line.startswith('template [[host_name('))
    if stripped != prefix:
        raise ValueError('general implementation changed')

    combined = (sources['src/ggml-metal/kernels/common.h'] +
                sources['src/ggml-metal/kernels/dequantize.h'] + general)
    embedded = []
    common_inlines = impl_inlines = 0
    for line in combined.splitlines(keepends=True):
        if line.strip() in ('#pragma once', '#include "common.h"',
                            '#include "dequantize.h"', '#include "quantize.h"'):
            continue
        if '__embed_ggml-common.h__' in line:
            embedded.append(sources['src/ggml-common.h'])
            common_inlines += 1
        elif '#include "ggml-metal-impl.h"' in line:
            embedded.append(sources['src/ggml-metal/ggml-metal-impl.h'])
            impl_inlines += 1
        else:
            embedded.append(line)
    if (common_inlines, impl_inlines) != (1, 1):
        raise ValueError('original embedding sentinel count changed')
    raw = ''.join(embedded).encode('utf-8')
    if b')HYLE_Q8_SOURCE"' in raw:
        raise ValueError('embedded C++ raw string delimiter collision')
    header = (b'#pragma once\nnamespace hyle::bench::resident {\n'
              b'inline constexpr char kOriginalMetalQ8Source[] = R"HYLE_Q8_SOURCE('
              + raw + b')HYLE_Q8_SOURCE";\n}\n')
    proof = {
        'kind': 'PINNED_ORIGINAL_GGML_METAL_GENERAL_Q8_SOURCE_ONLY',
        'revision': REVISION, 'input_sha256': INPUTS,
        'embedded_source_sha256': digest(raw),
        'embedded_header_sha256': digest(header),
        'original_general_without_export_lines_sha256': digest(prefix.encode('utf-8')),
        'original_exports': list(EXPORTS),
        'added_template_instance': rectangular.strip(),
        'embedding_recipe': 'Original Metal CMake lines 88-95; common/dequant/fa, then exact sentinels',
        'route': 'Original pad -> blk -> general Q8; no vector/split/reduce route',
        'limits': ['SOURCE_ONLY; no compiler, numerical, completion or P1 verdict',
                   '64/128 is a new explicit instance, not an original exported pipeline',
                   'Explicit general selection is not a best-of original host claim',
                   'Query conversion, gather, padding, mask, scratch and completion remain charged by the adapter'],
    }
    files = {'general-q8.metal': raw, 'general-q8-source.h': header,
             'source-provenance.json': (json.dumps(proof, indent=2) + '\n').encode('utf-8')}
    if output.exists():
        if any(not (output / name).is_file() or (output / name).read_bytes() != data
               for name, data in files.items() if name != 'source-provenance.json') or \
                not (output / 'source-provenance.json').is_file() or \
                json.loads((output / 'source-provenance.json').read_text(encoding='utf-8')) != proof:
            raise ValueError('retained original peer source identity changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name, data in files.items():
            (output / name).write_bytes(data)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.original.resolve(strict=True), args.output)))
