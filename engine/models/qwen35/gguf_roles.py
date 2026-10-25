"""Typed source-derived Qwen text/MTP GGUF header inventory, without payload conversion.

This is input preparation, not a generic GGUF loader or an inference executor.
The record is deliberately distinct from HYLE_QWEN_ROLE_FILES_V2 (HF roles).
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

if __package__:
    from .gguf_header import read_local_header, require
    from . import geometry
else:
    from gguf_header import read_local_header, require
    import geometry

SIGNATURE = 'HYLE_QWEN35_GGUF_HEADER_ROLES_V1'
PAYLOAD_SIGNATURE = 'HYLE_QWEN35_GGUF_PAYLOAD_FILES_V1'
ROLE_SIGNATURE = 'HYLE_QWEN_GGUF_ROLE_FILES_V2'
ROOT = Path(__file__).resolve().parents[3]
# Original physical formats carried by the local text consumer.
# They describe original storage, not activation precision or a Provider.
CODECS = {0: ('F32', 1, 4), 1: ('F16', 1, 2), 30: ('BF16', 1, 2),
          8: ('Q8_0', 32, 34), 12: ('Q4_K', 256, 144),
          13: ('Q5_K', 256, 176), 14: ('Q6_K', 256, 210), 11: ('Q3_K', 256, 110),
          23: ('IQ4_XS', 256, 136), 20: ('IQ4_NL', 32, 18), 21: ('IQ3_S', 256, 110),
          40: ('NVFP4', 64, 36)}


def role_specs(g):
    """Original GGUF names mapped to source-derived logical axes and meaning."""
    shapes = geometry.role_shapes(g)
    result = {}

    def add(physical, logical, semantic):
        result[physical] = dict(logical_name=logical, logical_shape=list(shapes[logical]), semantic=semantic)

    add('token_embd.weight', 'model.language_model.embed_tokens.weight', 'embedding_rows')
    add('output_norm.weight', 'model.language_model.norm.weight', 'rms_effective_gamma')
    if not g['tied_head']:
        add('output.weight', 'lm_head.weight', 'linear_weight')
    kinds = [(i, f'model.language_model.layers.{i}.', kind) for i,kind in enumerate(g['layers'])]
    if g['mtp_layers']: kinds.append((len(g['layers']), 'mtp.layers.0.', 'A'))
    for layer,logical,kind in kinds:
        physical = f'blk.{layer}.'
        for source, target in (('attn_norm.weight','input_layernorm.weight'),
                               ('post_attention_norm.weight','post_attention_layernorm.weight')):
            add(physical+source, logical+target, 'rms_effective_gamma')
        for leg in ('gate','up','down'):
            add(physical+f'ffn_{leg}.weight', logical+f'mlp.{leg}_proj.weight', 'linear_weight')
        if kind == 'A':
            for source, target in (('attn_q.weight','q_proj.weight'), ('attn_k.weight','k_proj.weight'),
                                   ('attn_v.weight','v_proj.weight'), ('attn_output.weight','o_proj.weight')):
                add(physical+source, logical+'self_attn.'+target, 'linear_weight')
            for axis in ('q','k'):
                add(physical+f'attn_{axis}_norm.weight', logical+f'self_attn.{axis}_norm.weight', 'rms_effective_gamma')
        else:
            for source, target, semantic in (
                ('ssm_a','A_log','gdn_negative_exp_A_log'), ('ssm_dt.bias','dt_bias','gdn_dt_bias'),
                ('ssm_norm.weight','norm.weight','gdn_effective_gamma'),
                ('ssm_conv1d.weight','conv1d.weight','gdn_convolution_kernel'),
                ('attn_qkv.weight','in_proj_qkv.weight','linear_weight'),
                ('attn_gate.weight','in_proj_z.weight','linear_weight'),
                ('ssm_alpha.weight','in_proj_a.weight','linear_weight'),
                ('ssm_beta.weight','in_proj_b.weight','linear_weight'), ('ssm_out.weight','out_proj.weight','linear_weight')):
                add(physical+source, logical+'linear_attn.'+target, semantic)
    if g['mtp_layers']:
        for source, target in (('eh_proj','fc'), ('enorm','pre_fc_norm_embedding'),
                                ('hnorm','pre_fc_norm_hidden'), ('shared_head_norm','norm')):
            add(f"blk.{len(g['layers'])}.nextn."+source+'.weight', 'mtp.'+target+'.weight',
                'linear_weight' if source == 'eh_proj' else 'rms_effective_gamma')
    return result


def _shape_ne(spec):
    shape = spec['logical_shape']
    if spec['semantic'] == 'gdn_convolution_kernel':
        return [shape[2],shape[0]]
    return list(reversed(shape))


def typed_inventory(header, source, *, header_sha256):
    """Validate a complete header; preserve physical names, bytes and meaning.

    Naming the decay's HF destination A_log does not make its bytes raw A_log:
    its explicit semantic tag requires direct negative-exp consumption. The
    caller must not pass this record to the existing HF-only role loader.
    """
    require(isinstance(source, dict) and bool(source.get('repository')) and
            re.fullmatch('[0-9a-f]{40}', source.get('revision','')) and
            re.fullmatch('[0-9a-f]{64}', source.get('lfs_sha256','')) and
            re.fullmatch('[0-9a-f]{64}', header_sha256) and
            isinstance(source.get('file'),str) and source['file'].endswith('.gguf') and
            type(source.get('size')) is int and source['size'] > 0,
            'GGUF inventory requires its fixed original file/revision/SHA/size identity')
    metadata = header['metadata']
    g = geometry.from_gguf(metadata, header['tensors'])
    require(metadata.get('split.count',1) == 1 and metadata.get('split.no',0) == 0,
            'this input inventory requires a complete single-file GGUF header')
    specs = role_specs(g)
    tensors = header['tensors']
    names = [row['name'] for row in tensors]
    require(header['tensor_count'] == len(tensors) == len(specs) and len(set(names)) == len(names),
            'Qwen GGUF tensor inventory is incomplete or duplicated')
    require(set(names) == set(specs), 'Qwen GGUF has missing, extra or visual roles')
    require(metadata.get('split.tensors.count',len(specs)) == len(specs), 'split GGUF tensor total differs')
    start = header['tensor_data_offset']
    require(type(start) is int and 24 <= start <= 16*1024*1024 and start <= source['size'],
            'invalid complete GGUF tensor-data offset')
    alignment = metadata.get('general.alignment',32)
    require(type(alignment) is int and 0 < alignment <= 4096 and start % alignment == 0,
            'invalid GGUF role/data alignment')
    roles = []
    for row in tensors:
        spec = specs[row['name']]
        require(row['shape'] == _shape_ne(spec), 'GGUF role shape differs: '+row['name'])
        code = row['dtype_code']
        require(type(code) is int and code in CODECS, 'unsupported physical GGUF codec: '+row['name'])
        codec, block_elements, block_bytes = CODECS[code]
        shape = spec['logical_shape']
        require(spec['semantic'] in ('linear_weight','embedding_rows') or block_elements == 1,
                'GGUF scalar/norm/convolution must preserve its plain floating representation: '+row['name'])
        require(row['shape'][0] % block_elements == 0, 'GGUF role has an incomplete physical K block')
        count = math.prod(row['shape'])
        extent = count // block_elements * block_bytes
        offset = row['offset']
        require(type(offset) is int and offset >= 0 and offset % alignment == 0 and
                extent <= source['size']-start-offset, 'GGUF role range exceeds its original source: '+row['name'])
        plain = block_elements == 1
        storage = dict(kind='plain' if plain else 'gguf_blocked', dtype=codec,
                       logical_dtype=codec if plain else 'F32', logical_shape=shape, original_ne=row['shape'])
        if not plain:
            storage.update(block_axis=1, elements_per_block=block_elements, bytes_per_block=block_bytes,
                           byte_strides=[shape[1]//block_elements*block_bytes, block_bytes])
        roles.append(dict(physical_name=row['name'], logical_name=spec['logical_name'],
                          semantic=spec['semantic'], dtype_code=code, storage=storage,
                          original_offset=offset, absolute_inclusive=[start+offset,start+offset+extent-1],
                          bytes=extent, materialization='none; original physical bytes and view-only layout'))
    ordered = sorted(roles,key=lambda row:row['absolute_inclusive'][0])
    require(all(a['absolute_inclusive'][1] < b['absolute_inclusive'][0] for a,b in zip(ordered,ordered[1:])),
            'GGUF physical role ranges overlap')
    counts = Counter(row['storage']['dtype'] for row in roles)
    return dict(signature=SIGNATURE, family='qwen35', complete_header=True,
        source=dict(source, header_sha256=header_sha256, header_bytes=start),
        physical_role_count=len(roles), main_role_count=sum(not row['logical_name'].startswith('mtp.') for row in roles),
        mtp_role_count=sum(row['logical_name'].startswith('mtp.') for row in roles),
        physical_codecs=dict(sorted(counts.items())), selected_payload_bytes=sum(row['bytes'] for row in roles),
        geometry=g,
        representation=dict(rms_norm='effective F32 gamma; do not add one',
            gdn_norm='original floating gamma; no delta conversion', gdn_decay='ssm_a is negative exp(A_log); do not Exp/Neg again',
            gdn_convolution='GGUF ne[kernel,qkv_width] viewed as logical[qkv_width,1,kernel], no repacking',
            matrix_layout='logical[N,K] view of original GGUF rows; packed block axis1',
            activation_precision='not selected by this header inventory',
            convention_basis='qwen35 GGUF model contract; header identifies roles/storage, no parameter values were read'),
        vocabulary=dict(embedding_physical='token_embd.weight', target_head_physical='token_embd.weight' if g['tied_head'] else 'output.weight',
            draft_embedding_physical='token_embd.weight', draft_head_physical='token_embd.weight' if g['tied_head'] else 'output.weight',
            independent_head_present=not g['tied_head'], scope='actual complete role inventory, not a source-family assumption'),
        roles=roles, tensor_payload_bytes_read=0)


def inspect_local(path, source):
    header, raw = read_local_header(path,whole_file_bytes=source['size'])
    return typed_inventory(header,source,header_sha256=hashlib.sha256(raw).hexdigest())


def _quoted(text):
    text = str(text)
    require(not any(char in text for char in '\r\n\0'), 'GGUF source field has a control character')
    return '"'+text.replace('\\','\\\\').replace('"','\\"')+'"'


def _range_sha256(path,offset,count):
    require(type(offset) is int and type(count) is int and offset >= 0 and count > 0 and
            offset+count <= path.stat().st_size, 'short original GGUF role source')
    result = hashlib.sha256()
    with path.open('rb') as source:
        source.seek(offset)
        while count:
            raw = source.read(min(count,1024*1024))
            require(bool(raw), 'short original GGUF role read')
            result.update(raw);count -= len(raw)
    return result.hexdigest()


def prepare_role_files(path, mtp):
    """Authenticate borrowed actual role ranges; emit distinct typed TSV.

    Always write the complete source role metadata. MTP-off selection is a loader
    decision after full validation, not permission to hide a damaged NextN.
    """
    original = json.loads(Path(path).read_bytes())
    require(original.get('signature') == PAYLOAD_SIGNATURE and original.get('complete') is True,
            'GGUF preparation requires complete acquired payload records, not a header-only inventory')
    source,header = original['source'],original['header']
    header_path = (ROOT/header['path']).resolve()
    raw = header_path.read_bytes()
    received = hashlib.sha256(raw).hexdigest()
    require(header['bytes'] == source['header_bytes'] == len(raw) and
            header['sha256'] == source['header_sha256'] == received,
            'cached original GGUF header identity differs')
    inventory = inspect_local(header_path,source)
    require(not mtp or inventory['geometry']['mtp_layers'] != 0,
            'source GGUF geometry has no NextN layer')
    selected = {row['physical_name']:row for row in inventory['roles']}
    bindings = original['roles']
    require(len(bindings) == len(selected) and
            {row['physical_name'] for row in bindings} == set(selected),
            'GGUF acquired payload bindings are incomplete or duplicated')
    lines,identities = [],[]
    for binding in sorted(bindings,key=lambda row:row['physical_name']):
        row = selected[binding['physical_name']]
        require(binding['bytes'] == row['bytes'] and binding['original_offset'] == row['absolute_inclusive'][0] and
                re.fullmatch('[0-9a-f]{64}',binding.get('sha256','')),
                'GGUF acquired original role range identity differs')
        body_path = (ROOT/binding['path']).resolve()
        require(_range_sha256(body_path,binding['offset'],row['bytes']) == binding['sha256'],
                'original GGUF role SHA256 differs: '+row['physical_name'])
        shape = row['storage']['logical_shape']
        lines.append('\t'.join((row['logical_name'],row['physical_name'],row['storage']['dtype'],row['semantic'],
            str(len(shape)),*(str(d) for d in shape),_quoted(body_path.as_posix()),str(binding['offset']),
            str(row['bytes']),str(binding['original_offset']))))
        identities.append(dict(row,sha256=binding['sha256']))
    identity = dict(source=source,geometry=inventory['geometry'],roles=identities)
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    source_line = '\t'.join(('source',_quoted(source['repository']),source['revision'],_quoted(source['file']),
        str(source['size']),source['lfs_sha256'],_quoted(source.get('api_xet_hash') or ''),
        source['header_sha256'],str(source['header_bytes'])))
    chosen = [row for row in inventory['roles'] if mtp or not row['logical_name'].startswith('mtp.')]
    return [ROLE_SIGNATURE,fingerprint,geometry.line(inventory['geometry']),source_line,*sorted(lines)],dict(
        source_id=fingerprint,source_representation='gguf_converted',source_repository=source['repository'],
        source_revision=source['revision'],original_gguf_source=source,physical_roles=len(chosen),
        geometry=inventory['geometry'],metadata_tensor_count=len(lines),selected_tensor_count=len(chosen),selected_weight_bytes=sum(row['bytes'] for row in chosen),
        text_weight_bytes=sum(row['bytes'] for row in inventory['roles'] if not row['logical_name'].startswith('mtp.')),
        mtp_weight_bytes=sum(row['bytes'] for row in inventory['roles'] if row['logical_name'].startswith('mtp.')),
        storage='borrowed original GGUF role ranges; no repacking or whole-bank dequantization')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--header',type=Path,required=True)
    parser.add_argument('--source-identity',type=Path,required=True,
        help='JSON object: repository/revision/file/size/lfs_sha256; header-only source, no HTTP')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = inspect_local(args.header,json.loads(args.source_identity.read_bytes()))
    with args.output.open('x',encoding='utf-8',newline='\n') as output:
        json.dump(result,output,ensure_ascii=False,indent=2); output.write('\n')


if __name__ == '__main__':
    main()
