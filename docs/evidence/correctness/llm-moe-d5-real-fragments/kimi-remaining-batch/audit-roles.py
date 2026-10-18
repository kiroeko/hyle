import collections
import hashlib
import json
import re
from pathlib import Path

official_path = Path('.scratch/official-weight-audit.json')
gguf_path = Path('.scratch/gguf-weight-audit.json')
official = json.loads(official_path.read_text(encoding='utf-8'))['models']['kimi_k3']
ggufs = json.loads(gguf_path.read_text(encoding='utf-8'))
result = {'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (official_path, gguf_path)}, 'profiles': []}
for name, source in [('official', official), ('primary', ggufs['kimi_k3_ud_q4_k_xl']), ('fidelity', ggufs['kimi_k3_ud_q8_k_xl'])]:
    groups = collections.defaultdict(list)
    for tensor in source['tensors']:
        raw = tensor['name']
        if name == 'official':
            if not raw.startswith('language_model.'):
                continue
            selected = any(role in raw for role in ('.mlp.', '_res_', 'layernorm.weight', '.q_a_proj.', '.q_a_layernorm.', '.q_b_proj.', '.kv_a_proj_with_mqa.', '.kv_a_layernorm.', '.kv_b_proj.'))
            selected |= '.self_attn.g_proj.' in raw or '.self_attn.o_proj.' in raw or raw == 'language_model.model.norm.weight'
            role = re.sub(r'\.layers\.\d+\.', '.layers.*.', raw)
        else:
            selected = any(role in raw for role in ('attn_q_a.', 'attn_q_a_norm.', 'attn_q_b.', 'attn_kv_a_mqa.', 'attn_kv_a_norm.', 'attn_k_b.', 'attn_v_b.', 'attn_gate.', 'attn_output.', 'res_score.', 'attn_norm.', 'ffn_norm.'))
            selected |= bool(re.search(r'\.ffn_(gate|up|down)\.weight$', raw)) or raw == 'output_norm.weight'
            role = re.sub(r'^blk\.\d+\.', 'blk.*.', raw)
        if selected:
            groups[(role, tensor['dtype'], tuple(map(int, tensor['shape'])))].append(raw)
    rows = [{'role': role, 'dtype': dtype, 'shape': list(shape), 'count': len(names), 'first': names[0], 'last': names[-1], 'names_sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest()} for (role, dtype, shape), names in sorted(groups.items())]
    result['profiles'].append({'id': name, 'revision': source['revision'], 'groups': rows})
    print(name, len(rows), 'signatures', sum(r['count'] for r in rows), 'tensors')
    for row in rows:
        print(' ', row['count'], row['role'], row['dtype'], row['shape'])
out = Path('.scratch/validation/d5-kimi-corrected/all-layer-roles.json')
out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
