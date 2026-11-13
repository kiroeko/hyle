"""Local Qwen3.5 image recipe and immutable vision weight-range preparation.

No network or model execution. The target identity binds this particular
preparation to an already prepared language model, including quantized targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from . import gguf_header


def vision_shapes(config):
    h, f, p, t = (config[k] for k in ('hidden_size', 'intermediate_size', 'patch_size', 'temporal_patch_size'))
    result = {'patch_embed.proj.weight': [h, 3 * t * p * p], 'patch_embed.proj.bias': [h],
              'pos_embed.weight': [config['num_position_embeddings'], h],
              'merger.norm.weight': [h], 'merger.norm.bias': [h],
              'merger.linear_fc1.weight': [h * config['spatial_merge_size'] ** 2] * 2,
              'merger.linear_fc1.bias': [h * config['spatial_merge_size'] ** 2],
              'merger.linear_fc2.weight': [config['out_hidden_size'], h * config['spatial_merge_size'] ** 2],
              'merger.linear_fc2.bias': [config['out_hidden_size']]}
    for i in range(config['depth']):
        for name, shape in {'attn.qkv.weight': [3*h, h], 'attn.qkv.bias': [3*h],
                            'attn.proj.weight': [h, h], 'attn.proj.bias': [h],
                            'mlp.linear_fc1.weight': [f, h], 'mlp.linear_fc1.bias': [f],
                            'mlp.linear_fc2.weight': [h, f], 'mlp.linear_fc2.bias': [h],
                            'norm1.weight': [h], 'norm1.bias': [h],
                            'norm2.weight': [h], 'norm2.bias': [h]}.items():
            result[f'blocks.{i}.{name}'] = shape
    return {'model.visual.' + name: shape for name, shape in result.items()}


def read_recipe(model: Path):
    config = json.loads((model / 'config.json').read_text(encoding='utf-8'))
    vision = config.get('vision_config', {})
    if config.get('model_type') != 'qwen3_5' or vision.get('deepstack_visual_indexes'):
        raise ValueError('this image recipe requires Qwen3.5 without deepstack conditioning')
    keys = ('depth', 'hidden_size', 'intermediate_size', 'num_heads', 'patch_size',
            'temporal_patch_size', 'spatial_merge_size', 'num_position_embeddings', 'out_hidden_size')
    if any(type(vision.get(key)) is not int or vision[key] <= 0 for key in keys):
        raise ValueError('invalid Qwen vision geometry')
    if (vision['hidden_size'] % vision['num_heads'] or vision['hidden_size'] // vision['num_heads'] % 4 or
            vision['in_channels'] != 3 or
            vision['hidden_act'] != 'gelu_pytorch_tanh' or
            math.isqrt(vision['num_position_embeddings']) ** 2 != vision['num_position_embeddings'] or
            vision['out_hidden_size'] != config['text_config']['hidden_size']):
        raise ValueError('unsupported Qwen image mathematical recipe')
    processor = json.loads((model / 'preprocessor_config.json').read_text(encoding='utf-8'))
    if (processor.get('patch_size') != vision['patch_size'] or processor.get('merge_size') != vision['spatial_merge_size'] or
            processor.get('temporal_patch_size') != vision['temporal_patch_size'] or
            processor.get('image_mean') != [0.5] * 3 or processor.get('image_std') != [0.5] * 3):
        raise ValueError('unsupported Qwen image normalization or patch layout')
    if (any(processor.get(key,True) is not True for key in ('do_resize','do_rescale','do_normalize','do_convert_rgb')) or
            processor.get('resample',3)!=3 or not math.isclose(processor.get('rescale_factor',1/255),1/255) or
            any(type(processor.get('size',{}).get(key)) is not int or processor['size'][key]<=0
                for key in ('longest_edge','shortest_edge')) or
            processor['size']['shortest_edge']>processor['size']['longest_edge']):
        raise ValueError('unsupported Qwen image resize or rescale recipe')
    rope = config['text_config'].get('rope_parameters', {})
    if not rope.get('mrope_interleaved') or sum(rope.get('mrope_section', [])) * 2 != int(
            config['text_config']['head_dim'] * rope['partial_rotary_factor']):
        raise ValueError('unsupported Qwen language multimodal position recipe')
    return config, vision, processor


def _hf_roles(model, shapes):
    result = {}
    files = sorted(model.glob('*.safetensors'))
    for file in files:
        size = file.stat().st_size
        with file.open('rb') as stream:
            first = stream.read(8)
            if len(first) != 8: raise ValueError('short safetensors header')
            length = struct.unpack('<Q', first)[0]
            if length > min(size - 8, 64 << 20): raise ValueError('invalid safetensors header')
            header = json.loads(stream.read(length))
        for name, row in header.items():
            if name not in shapes: continue
            shape = row['shape']
            if name.endswith('patch_embed.proj.weight'): shape = [shape[0], math.prod(shape[1:])]
            if shape != shapes[name] or name in result or row['dtype'] not in ('F32', 'BF16', 'F16'):
                raise ValueError('invalid or duplicate visual tensor: ' + name)
            begin, end = row['data_offsets']
            width = 4 if row['dtype'] == 'F32' else 2
            if not 0 <= begin < end <= size - 8 - length or end - begin != math.prod(shape) * width:
                raise ValueError('invalid visual tensor extent: ' + name)
            result[name] = (row['dtype'], shape, file.resolve(), 8 + length + begin, end - begin)
    if result.keys() != shapes.keys():
        raise ValueError('missing visual tensors: ' + ', '.join(sorted(shapes.keys() - result.keys())))
    return result


def _mmproj_roles(path, shapes, vision, output):
    import numpy as np
    if vision['temporal_patch_size']!=2: raise ValueError('this mmproj layout requires two temporal patch kernels')
    keys = ('general.architecture', 'clip.projector_type', 'clip.has_vision_encoder',
            'clip.vision.block_count', 'clip.vision.embedding_length', 'clip.vision.feed_forward_length',
            'clip.vision.attention.head_count', 'clip.vision.patch_size', 'clip.vision.spatial_merge_size')
    info, _ = gguf_header.read_local_header(path, retained_metadata=keys)
    meta = info['metadata']
    if meta.get('general.architecture') != 'clip' or meta.get('clip.projector_type') != 'qwen3vl_merger':
        raise ValueError('mmproj does not declare the Qwen3-VL merger recipe')
    for key, field in zip(keys[3:], ('depth', 'hidden_size', 'intermediate_size', 'num_heads', 'patch_size', 'spatial_merge_size')):
        if meta.get(key) != vision[field]: raise ValueError('mmproj geometry differs: ' + key)
    mapping = {'v.patch_embd.bias': 'patch_embed.proj.bias', 'v.position_embd.weight': 'pos_embed.weight',
               'v.post_ln.weight': 'merger.norm.weight', 'v.post_ln.bias': 'merger.norm.bias',
               'mm.0.weight': 'merger.linear_fc1.weight', 'mm.0.bias': 'merger.linear_fc1.bias',
               'mm.2.weight': 'merger.linear_fc2.weight', 'mm.2.bias': 'merger.linear_fc2.bias'}
    for i in range(vision['depth']):
        for original, role in (('attn_qkv', 'attn.qkv'), ('attn_out', 'attn.proj'), ('ln1', 'norm1'),
                                ('ln2', 'norm2'), ('ffn_up', 'mlp.linear_fc1'), ('ffn_down', 'mlp.linear_fc2')):
            for tail in ('weight', 'bias'): mapping[f'v.blk.{i}.{original}.{tail}'] = f'blocks.{i}.{role}.{tail}'
    encodings = {0: ('F32', 1, 4), 1: ('F16', 1, 2), 30: ('BF16', 1, 2),
                 8: ('Q8_0', 32, 34), 12: ('Q4_K', 256, 144), 13: ('Q5_K', 256, 176), 14: ('Q6_K', 256, 210)}
    result, patches = {}, {}
    with path.open('rb') as stream:
        for tensor in info['tensors']:
            name, code = tensor['name'], tensor['dtype_code']
            if code not in encodings: raise ValueError('unsupported mmproj tensor encoding: ' + str(code))
            dtype, block, width = encodings[code]
            shape = list(reversed(tensor['shape']))
            if tensor['shape'][0] % block: raise ValueError('mmproj encoded row is incomplete')
            size = math.prod(shape) // block * width
            offset = info['tensor_data_offset'] + tensor['offset']
            if offset + size > path.stat().st_size: raise ValueError('mmproj tensor exceeds its source')
            if name in ('v.patch_embd.weight', 'v.patch_embd.weight.1'):
                if block != 1 or shape != [vision['hidden_size'], 3, vision['patch_size'], vision['patch_size']]:
                    raise ValueError('mmproj patch kernel differs from the source recipe')
                stream.seek(offset); data = stream.read(size)
                patches[name] = (dtype, np.frombuffer(data, dtype='<u4' if width == 4 else '<u2').reshape(shape))
                continue
            if name not in mapping: raise ValueError('unsupported mmproj tensor: ' + name)
            logical = 'model.visual.' + mapping[name]
            if shape != shapes[logical] or logical in result:
                raise ValueError('mmproj tensor shape differs: ' + name)
            if block != 1 and (len(shape) != 2 or not logical.endswith('.weight') or logical.endswith('pos_embed.weight')):
                raise ValueError('quantized mmproj tensor has no defined row projection contract')
            result[logical] = (dtype, shape, path, offset, size)
    if patches.keys() != {'v.patch_embd.weight', 'v.patch_embd.weight.1'}:
        raise ValueError('mmproj requires both original temporal patch kernels')
    a, b = patches['v.patch_embd.weight'], patches['v.patch_embd.weight.1']
    if a[0] != b[0]: raise ValueError('temporal patch kernel encodings differ')
    packed = np.stack((a[1], b[1]), axis=2).reshape(vision['hidden_size'], -1).tobytes()
    patch_path = output.parent / ('vision-patch-' + hashlib.sha256(packed).hexdigest() + '.bin')
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    if patch_path.exists():
        if patch_path.read_bytes() != packed: raise ValueError('existing immutable patch kernel differs')
    else: patch_path.write_bytes(packed)
    logical = 'model.visual.patch_embed.proj.weight'
    result[logical] = (a[0], shapes[logical], patch_path, 0, len(packed))
    if result.keys() != shapes.keys(): raise ValueError('incomplete mmproj visual inventory')
    return result


def prepare_vision(model: Path, inputs: Path, output: Path, *, mmproj: Path | None = None):
    model, inputs, output = model.resolve(), inputs.resolve(), output.resolve()
    config, vision, processor = read_recipe(model)
    # Source identity is the same explicit model binding used by tokenizer and
    # draft assets. No filename, vocabulary size or backend name is an identity.
    from ...model_preparation import prepared_source_identity
    target = prepared_source_identity(inputs)[0]
    roles = (_hf_roles(model, vision_shapes(vision)) if mmproj is None else
             _mmproj_roles(mmproj.resolve(), vision_shapes(vision), vision, output))
    header = ["HYLE_QWEN_VISION_V1", target,
              '\t'.join(str(vision[key]) for key in ('depth', 'hidden_size', 'intermediate_size', 'num_heads',
                  'patch_size', 'temporal_patch_size', 'spatial_merge_size', 'num_position_embeddings', 'out_hidden_size')),
              '\t'.join(str(config[key]) for key in ('image_token_id', 'vision_start_token_id', 'vision_end_token_id')),
              '\t'.join(map(str, config['text_config']['rope_parameters']['mrope_section']))]
    rows = []
    for name, (dtype, shape, path, offset, size) in sorted(roles.items()):
        if any(ch in str(path) for ch in '\t\n\r'): raise ValueError('invalid source path')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            stream.seek(offset)
            remaining = size
            while remaining:
                data = stream.read(min(remaining, 8 << 20))
                if not data: raise ValueError('short visual weight source')
                digest.update(data); remaining -= len(data)
        rows.append('\t'.join((name, dtype, ','.join(map(str, shape)), str(path), str(offset), str(size), digest.hexdigest())))
    identity = hashlib.sha256(('\n'.join(header + rows) + '\n').encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(header + [identity] + rows) + '\n', encoding='utf-8', newline='\n')
    return identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='local complete Qwen3.5 checkpoint')
    parser.add_argument('--inputs', type=Path, required=True, help='prepared target language model')
    parser.add_argument('--output', type=Path, required=True, help='reusable vision.tsv asset')
    parser.add_argument('--mmproj', type=Path, help='optional local Qwen3-VL merger GGUF instead of HF visual tensors')
    args = parser.parse_args(argv)
    print(prepare_vision(args.model, args.inputs, args.output, mmproj=args.mmproj))


if __name__ == '__main__': main()
