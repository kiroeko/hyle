"""Materialize the page shader's shared numerical source, without a second codec."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(output):
    source = Path(__file__).resolve().parent.parent
    numerical = (source / 'compute_shader_shared/nvfp4_f32.h').read_text(encoding='utf-8')
    start, end = 'R"HYLE_SHADER(', ')HYLE_SHADER"'
    if numerical.count(start) != 1 or numerical.count(end) != 1:
        raise ValueError('shared numerical source delimiters differ')
    numerical = numerical.split(start, 1)[1].split(end, 1)[0]
    codec = (source / 'compute_route_vulkan/shaders/fp8_storage.glsl').read_text(encoding='utf-8')
    content = ('// Generated from HYLE shared IEEE arithmetic and E4 storage codec.\n'
               'uint hyle_mul_hi(uint a,uint b) { uint high,low; umulExtended(a,b,high,low); return high; }\n'
               'uint hyle_clz(uint v) { return v==0u ? 32u : 31u-uint(findMSB(v)); }\n' + numerical + '\n' + codec)
    patch = {'path': 'src/ggml-vulkan/vulkan-shaders/hyle_e4_cache.glsl',
             'before_sha256': None, 'after_sha256': hashlib.sha256(content.encode()).hexdigest(),
             'content': content}
    output.write_text(json.dumps({'files': [patch]}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
