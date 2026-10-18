import json
from pathlib import Path

official = json.loads(Path('.scratch/official-weight-audit.json').read_text(encoding='utf-8'))['models']
gguf = json.loads(Path('.scratch/gguf-weight-audit.json').read_text(encoding='utf-8'))
facts = json.loads(Path('tests/compute_models/d5_moe_profile_facts.json').read_text(encoding='utf-8'))
official_names = ('q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight',
                  'b_proj.weight','f_a_proj.weight','f_b_proj.weight','g_proj.weight',
                  'g_a_proj.weight','g_b_proj.weight')
gguf_names = ('attn_q.weight','attn_k.weight','attn_v.weight','attn_output.weight',
              'ssm_beta.weight','ssm_f_a.weight','ssm_f_b.weight','ssm_g.weight',
              'ssm_g_a.weight','ssm_g_b.weight')
result = dict(schema_version=1, purpose='Exact KDA projection roles from fixed tensor headers. Group only equal profile/dtype/input/output signatures; every source name retained. Not an execution verdict.',
              source_files=facts['source_files'], cases=[], parameters=[])
for profile in facts['profiles']:
    model = profile['id'].split(':')[0]
    if model not in ('kimi_k3','glm53_flash'):
        continue
    is_official = profile['profile']['format_family']=='safetensors'
    if is_official:
        audit=official[model]
        prefix=('language_model.model.' if model=='kimi_k3' else 'model.language_model.')+'layers.0.self_attn.'
        tensor_map={t['name']:t for t in audit['tensors']}
        names=[prefix+n for n in official_names if prefix+n in tensor_map]
        params=[t for t in audit['tensors'] if t['name'].startswith(prefix) and any(n in t['name'] for n in ('conv1d','A_log','dt_bias','o_norm'))]
    else:
        identity=profile['profile']
        matches=[a for a in gguf.values() if a['repo']==identity['artifact_repository'] and a['revision']==identity['artifact_revision'] and a['variant']==identity['variant']]
        assert len(matches)==1
        audit=matches[0]
        tensor_map={t['name']:t for t in audit['tensors']}
        names=['blk.0.'+n for n in gguf_names if 'blk.0.'+n in tensor_map]
        params=[t for t in audit['tensors'] if t['name'].startswith('blk.0.ssm_') and any(n in t['name'] for n in ('conv1d','ssm_a','ssm_dt','ssm_norm'))]
    assert len(names)==(8 if model=='kimi_k3' else 9), (model, names)
    groups={}
    for name in names:
        tensor=tensor_map[name]
        shape=[int(x) for x in tensor['shape']]
        assert len(shape)==2 and tensor['dtype'] in ('BF16','F32','Q8_0')
        columns,reduction=shape if is_official else shape[::-1]
        signature=(tensor['dtype'],columns,reduction)
        groups.setdefault(signature,[]).append(tensor)
    for (dtype,columns,reduction),tensors in groups.items():
        result['cases'].append(dict(profile=profile['symbol'],dtype=dtype,columns=columns,reduction=reduction,tensors=tensors))
    result['parameters'].append(dict(profile=profile['symbol'], tensors=params))
result['parameter_notes']=[
    'Kimi official A_log is padded [128], while H=96. Pinned SGLang kimi_k3.py:1630-1656 explicitly loads the first num_heads entries before TP sharding. GGUF ssm_a is folded -exp(A_log), [96].',
    'Kimi conv/norm are F32; GLM Flash official conv/norm are BF16, while its GGUF versions store these as F32. Projection receipts do not qualify these conversion boundaries.'
]
base=Path('tests/compute_models')
(base/'d5_kda_projection_facts.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
lines=['#pragma once','','#include <array>','#include <cstdint>','', '#include "d5_moe_profile_cases.h"','',
       '// Generated from the fixed tensor observations in d5_kda_projection_facts.json.',
       '// Equal native MatMul signatures share a witness; all weight roles stay named.',
       'namespace d5_kda_projection_cases {','namespace c = hyle::compute;',
       'enum class Storage { kF32, kBFloat16, kQ8_0 };',
       'struct Case final {',
       '  const c::ModelArtifactProfileDescription *profile;',
       '  const char *roles;',
       '  std::int64_t columns, reduction;',
       '  Storage storage;',
       '};','inline constexpr std::array kCases{']
types={'F32':'kF32','BF16':'kBFloat16','Q8_0':'kQ8_0'}
for item in result['cases']:
    roles=' | '.join(t['name'] for t in item['tensors'])
    lines += ['    Case{&d5_moe_profile_cases::'+item['profile']+',',
              '         '+json.dumps(roles)+',',
              f'         {item["columns"]}, {item["reduction"]}, Storage::{types[item["dtype"]]}'+'},']
lines += ['};','} // namespace d5_kda_projection_cases','']
(base/'d5_kda_projection_cases.h').write_text('\n'.join(lines),encoding='utf-8')
print(len(result['cases']),'unique profile/signature cases for',sum(len(c['tensors']) for c in result['cases']),'exact projection roles.')
