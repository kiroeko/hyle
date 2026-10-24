"""Source Qwen text geometry and logical roles, shared by local preparers."""
from __future__ import annotations

import math

AXES = ('hidden', 'intermediate', 'vocabulary', 'query_heads', 'kv_heads',
        'key_depth', 'value_depth', 'rotary', 'gdn_key_heads', 'gdn_value_heads',
        'gdn_key_depth', 'gdn_value_depth', 'conv_kernel', 'context_limit')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(g):
    for key in AXES:
        require(type(g.get(key)) is int and 0 < g[key] <= (1 << 63)-1,
                'invalid source geometry: '+key)
    require(g['vocabulary'] <= (1 << 31)-1 and g['context_limit'] <= (1 << 31)-1 and
            g['query_heads'] % g['kv_heads'] == 0 and
            g['gdn_value_heads'] % g['gdn_key_heads'] == 0 and
            g['rotary'] % 2 == 0 and g['rotary'] <= g['key_depth'] and g['conv_kernel'] >= 2,
            'inconsistent source attention/recurrent geometry')
    for key in ('rope_theta', 'epsilon'):
        require(type(g.get(key)) in (int, float) and math.isfinite(g[key]) and
                0 < g[key] <= 3.4028234663852886e38, 'invalid source geometry: '+key)
    require(type(g.get('tied_head')) is bool and type(g.get('mtp_layers')) is int and
            g['mtp_layers'] in (0, 1) and g.get('gdn_layout') in ('grouped', 'cyclic') and
            type(g.get('layers')) is str and 0 < len(g['layers']) <= (1 << 32)-1 and
            all(x in 'GA' for x in g['layers']), 'invalid source head/layer geometry')
    hk, hv, dk, dv = (g[x] for x in ('gdn_key_heads','gdn_value_heads','gdn_key_depth','gdn_value_depth'))
    products = (g['hidden']*g['vocabulary'], g['hidden']*g['intermediate'],
        (2*hk*dk+hv*dv)*g['conv_kernel'], hv*dk*dv,
        g['query_heads']*(g['key_depth']+g['value_depth']),
        g['query_heads']*g['value_depth'], 2*g['hidden'],
        2*g['kv_heads']*g['key_depth'], 2*g['kv_heads']*g['value_depth'])
    require(all(x <= (1 << 63)-1 for x in products), 'source geometry product overflows')
    return g


def line(g):
    validate(g)
    values = [*(str(g[x]) for x in AXES), format(g['rope_theta'], '.17g'),
              format(g['epsilon'], '.17g'), str(int(g['tied_head'])),
              str(g['mtp_layers']), g['gdn_layout'], '"'+g['layers']+'"']
    return '\t'.join(('geometry', *values))


def _sections(sections, rotary):
    require(isinstance(sections, list) and bool(sections) and
            all(type(x) is int and x >= 0 for x in sections) and sum(sections)*2 == rotary,
            'source RoPE sections disagree with the rotary dimension')


def from_hf(original):
    c = original.get('text_config', original)
    require(c.get('model_type') == 'qwen3_5_text' and c.get('attn_output_gate') is True and
            c.get('attention_bias') is False and c.get('hidden_act') == 'silu' and
            c.get('output_gate_type') in (None, 'silu', 'swish') and
            not c.get('mtp_use_dedicated_embeddings', False),
            'source is outside the Qwen gated text/NextN mathematics')
    mapping = dict(hidden='hidden_size', intermediate='intermediate_size', vocabulary='vocab_size',
        query_heads='num_attention_heads', kv_heads='num_key_value_heads', key_depth='head_dim',
        value_depth='head_dim', gdn_key_heads='linear_num_key_heads', gdn_value_heads='linear_num_value_heads',
        gdn_key_depth='linear_key_head_dim', gdn_value_depth='linear_value_head_dim',
        conv_kernel='linear_conv_kernel_dim', context_limit='max_position_embeddings')
    g = {name:c.get(source) for name,source in mapping.items()}
    rope = c.get('rope_parameters', {})
    require(rope.get('rope_type') == 'default' and type(c.get('head_dim')) is int and
            type(rope.get('partial_rotary_factor')) in (int,float), 'missing source RoPE geometry')
    rotary = c['head_dim']*rope['partial_rotary_factor']
    require(math.isfinite(rotary) and rotary == int(rotary), 'nonintegral source rotary dimension')
    g['rotary'] = int(rotary)
    _sections(rope.get('mrope_section'), g['rotary'])
    kinds = c.get('layer_types')
    require(isinstance(kinds,list) and len(kinds) == c.get('num_hidden_layers') and
            all(x in ('linear_attention','full_attention') for x in kinds), 'missing/invalid source target layers')
    tied = c.get('tie_word_embeddings', original.get('tie_word_embeddings'))
    if 'tie_word_embeddings' in original:
        require(tied == original['tie_word_embeddings'], 'conflicting source tied vocabulary flags')
    g.update(layers=''.join('G' if x == 'linear_attention' else 'A' for x in kinds),
        rope_theta=rope.get('rope_theta'), epsilon=c.get('rms_norm_eps'), tied_head=tied,
        mtp_layers=c.get('mtp_num_hidden_layers'), gdn_layout='grouped')
    return validate(g)


def from_gguf(metadata, tensors):
    require(metadata.get('general.architecture') == 'qwen35' and metadata.get('general.type') == 'model',
            'source is not a Qwen gated text GGUF')
    by_name = {r['name']:r for r in tensors}
    embedding = by_name.get('token_embd.weight', {}).get('shape', [])
    require(len(embedding) == 2, 'missing source GGUF embedding geometry')
    mapping = dict(hidden='embedding_length', intermediate='feed_forward_length',
        query_heads='attention.head_count', kv_heads='attention.head_count_kv',
        key_depth='attention.key_length', value_depth='attention.value_length', rotary='rope.dimension_count',
        gdn_key_heads='ssm.group_count', gdn_value_heads='ssm.time_step_rank',
        gdn_key_depth='ssm.state_size', conv_kernel='ssm.conv_kernel', context_limit='context_length')
    g = {name:metadata.get('qwen35.'+source) for name,source in mapping.items()}
    inner, hv = metadata.get('qwen35.ssm.inner_size'), g['gdn_value_heads']
    require(type(inner) is int and type(hv) is int and inner > 0 and hv > 0 and inner % hv == 0,
            'source GGUF recurrent value width differs')
    count, mtp, interval = (metadata.get('qwen35.'+name) for name in
                            ('block_count','nextn_predict_layers','full_attention_interval'))
    require(type(count) is int and type(mtp) is int and mtp in (0,1) and count > mtp and
            type(interval) is int and interval > 0, 'missing source GGUF target/NextN layer geometry')
    g.update(vocabulary=embedding[1], gdn_value_depth=inner//hv, mtp_layers=mtp,
        tied_head='output.weight' not in by_name, gdn_layout='cyclic',
        layers=''.join('A' if (i+1)%interval == 0 else 'G' for i in range(count-mtp)),
        rope_theta=metadata.get('qwen35.rope.freq_base'),
        epsilon=metadata.get('qwen35.attention.layer_norm_rms_epsilon'))
    _sections(metadata.get('qwen35.rope.dimension_sections'), g['rotary'])
    return validate(g)


def role_shapes(g, *, include_mtp=True):
    validate(g)
    h, i, v = (g[x] for x in ('hidden','intermediate','vocabulary'))
    q, kv, dk, dv = (g[x] for x in ('query_heads','kv_heads','key_depth','value_depth'))
    hk, hv, gk, gv = (g[x] for x in ('gdn_key_heads','gdn_value_heads','gdn_key_depth','gdn_value_depth'))
    result = {'model.language_model.embed_tokens.weight':(v,h), 'model.language_model.norm.weight':(h,)}
    if not g['tied_head']: result['lm_head.weight'] = (v,h)
    kinds = [(f'model.language_model.layers.{n}.',kind) for n,kind in enumerate(g['layers'])]
    if include_mtp and g['mtp_layers']: kinds.append(('mtp.layers.0.','A'))
    for prefix,kind in kinds:
        result.update({prefix+'input_layernorm.weight':(h,), prefix+'post_attention_layernorm.weight':(h,),
            prefix+'mlp.gate_proj.weight':(i,h), prefix+'mlp.up_proj.weight':(i,h), prefix+'mlp.down_proj.weight':(h,i)})
        branch = ({'q_proj.weight':(q*(dk+dv),h), 'k_proj.weight':(kv*dk,h), 'v_proj.weight':(kv*dv,h),
            'o_proj.weight':(h,q*dv), 'q_norm.weight':(dk,), 'k_norm.weight':(dk,)} if kind == 'A' else
            {'A_log':(hv,), 'dt_bias':(hv,), 'norm.weight':(gv,), 'conv1d.weight':(2*hk*gk+hv*gv,1,g['conv_kernel']),
             'in_proj_qkv.weight':(2*hk*gk+hv*gv,h), 'in_proj_z.weight':(hv*gv,h),
             'in_proj_a.weight':(hv,h), 'in_proj_b.weight':(hv,h), 'out_proj.weight':(h,hv*gv)})
        result.update({prefix+('self_attn.' if kind == 'A' else 'linear_attn.')+name:shape for name,shape in branch.items()})
    if include_mtp and g['mtp_layers']:
        result.update({'mtp.fc.weight':(h,2*h), 'mtp.norm.weight':(h,),
            'mtp.pre_fc_norm_embedding.weight':(h,), 'mtp.pre_fc_norm_hidden.weight':(h,)})
    return result
