import json
from pathlib import Path

official=json.loads(Path('.scratch/official-weight-audit.json').read_text(encoding='utf-8'))['models']
gguf=json.loads(Path('.scratch/gguf-weight-audit.json').read_text(encoding='utf-8'))
profiles=json.loads(Path('tests/compute_models/d5_moe_profile_facts.json').read_text(encoding='utf-8'))
cases=[]
for item in profiles['profiles']:
    model=item['id'].split(':')[0]
    config=official[model]['config'].get('text_config',official[model]['config'])
    native=item['profile']['format_family']=='gguf'
    if native:
        identity=item['profile']
        matches=[g for g in gguf.values() if (g['repo'],g['revision'],g['variant'])==(identity['artifact_repository'],identity['artifact_revision'],identity['variant'])]
        assert len(matches)==1
        audit=matches[0]
        weights={t['name']:t for t in audit['tensors']}
        router=weights['blk.3.ffn_gate_inp.weight']
        tensors=[t for t in audit['tensors'] if t['name'].startswith('blk.3.') and any(s in t['name'] for s in ('ffn','exp_probs'))]
        expert=weights['blk.3.ffn_gate_exps.weight']
        latent,intermediate,experts=[int(x) for x in expert['shape']]
        shared=int(weights['blk.3.ffn_gate_shexp.weight']['shape'][1])
        formats={role:weights['blk.3.'+name]['dtype'] for role,name in [('gate','ffn_gate_exps.weight'),('up','ffn_up_exps.weight'),('down','ffn_down_exps.weight'),('shared_gate','ffn_gate_shexp.weight'),('shared_up','ffn_up_shexp.weight'),('shared_down','ffn_down_shexp.weight')]}
        if model=='kimi_k3':
            formats['latent']=weights['blk.3.ffn_routed_down.weight']['dtype']
    else:
        prefix={'kimi_k3':'language_model.model.layers.3.block_sparse_moe.', 'deepseek_v4_pro':'layers.3.ffn.','deepseek_v4_flash':'layers.3.ffn.','glm53':'model.layers.3.mlp.','glm53_flash':'model.language_model.layers.3.mlp.'}[model]
        tensors=[t for t in official[model]['tensors'] if t['name'].startswith(prefix) and ('.experts.' not in t['name'] or '.experts.0.' in t['name'])]
        router=next(t for t in tensors if t['name']==prefix+'gate.weight')
        experts=int(router['shape'][0])
        latent=config.get('routed_expert_hidden_size',config['hidden_size'])
        intermediate=config['moe_intermediate_size']
        shared=intermediate*config.get('n_shared_experts',config.get('num_shared_experts',1))
        formats={}
    top=config.get('num_experts_per_token',config.get('num_experts_per_tok'))
    scoring=config.get('scoring_func',config.get('moe_router_activation_func'))
    assert scoring in ('sigmoid','sqrtsoftplus')
    assert config.get('n_group',config.get('num_expert_group',1))==1
    cases.append(dict(profile=item['symbol'],model=model,hidden=config['hidden_size'],experts=experts,top_k=top,
                      latent=latent,intermediate=intermediate,shared_intermediate=shared,
                      scoring=scoring,scaling=config['routed_scaling_factor'],router_dtype=router['dtype'],
                      gguf=native,formats=formats,tensors=tensors))
out=Path('tests/compute_models')
facts=dict(schema_version=1,source_files=profiles['source_files'],cases=cases,
           scope='Full expert populations and true inner widths. Resident expert capacity may be reduced to top_k+1; selection still scores the complete population. No run verdict.',
           correction='Kimi 3584 is latent expert input/output width; intermediate SiTU width is 3072, shared width is 6144, dense width is 33792.')
(out/'d5_moe_execution_facts.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
lines=['#pragma once','','#include <array>','#include <cstdint>','#include "d5_moe_profile_cases.h"','',
       '// Exact tensor observations: d5_moe_execution_facts.json.',
       'namespace d5_moe_execution_cases {','namespace c = hyle::compute;',
       'enum class Storage { kQ4K, kQ5K, kQ8_0, kMxfp4, kBFloat16 };',
       'struct Case final {',
       '  const c::ModelArtifactProfileDescription *profile;',
       '  std::int64_t hidden, experts, top_k, latent, intermediate, shared_intermediate;',
       '  float scaling;',
       '  bool sqrt_softplus, router_bf16, gguf;',
       '  Storage gate, up, down, shared_gate, shared_up, shared_down, latent_storage;',
       '};','inline constexpr std::array kCases{']
mapping={'Q4_K':'kQ4K','Q5_K':'kQ5K','Q8_0':'kQ8_0','MXFP4':'kMxfp4','BF16':'kBFloat16'}
for case in cases:
    lines += ['    Case{&d5_moe_profile_cases::'+case['profile']+',',
              '         '+', '.join(str(case[k]) for k in ('hidden','experts','top_k','latent','intermediate','shared_intermediate'))+',',
              '         '+str(float(case['scaling']))+'F, '+', '.join(str(x).lower() for x in (case['scoring']=='sqrtsoftplus',case['router_dtype']=='BF16',case['gguf']))+',',
              '         '+', '.join('Storage::'+mapping[case['formats'].get(k,'BF16')] for k in ('gate','up','down','shared_gate','shared_up','shared_down','latent'))+'},']
lines += ['};','} // namespace d5_moe_execution_cases','']
(out/'d5_moe_execution_cases.h').write_text('\n'.join(lines),encoding='utf-8')
print('Fixed',len(cases),'router profiles and',sum(c['gguf'] for c in cases),'native GGUF MoE chains.')
