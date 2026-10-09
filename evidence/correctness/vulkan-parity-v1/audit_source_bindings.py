"""Finite original-source binding audit; never emits runtime qualification."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NEW = {
    'kQ4_0': 'GGML_TYPE_Q4_0', 'kQ4_1': 'GGML_TYPE_Q4_1',
    'kQ2K': 'GGML_TYPE_Q2_K', 'kIq2Xxs': 'GGML_TYPE_IQ2_XXS',
    'kIq2Xs': 'GGML_TYPE_IQ2_XS', 'kIq2S': 'GGML_TYPE_IQ2_S',
    'kIq1S': 'GGML_TYPE_IQ1_S', 'kIq1M': 'GGML_TYPE_IQ1_M',
}
EMBEDDING = {'kQ4K', 'kQ6K', 'kQ8_0'}
NEW_EMBEDDING = {'kQ2K', 'kQ3K', 'kQ4_0', 'kQ4_1', 'kQ5K'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    demand_path = ROOT / 'docs/implementation/tasks/compute/cpu-final-demand-carriers.json'
    demand = json.loads(demand_path.read_text(encoding='utf-8'))
    provider = ROOT / 'src/compute_provider_ggml_vulkan/region.cc'
    text = provider.read_text(encoding='utf-8')
    mapper = text[text.index('ggml_type native_quantized_type('):text.index('bool native_quantized(')]
    current = dict(re.findall(r'case GgufQuantizedType::(\w+):\s*return (GGML_TYPE_\w+);', mapper))
    backend = args.source / 'src/ggml-vulkan/ggml-vulkan.cpp'
    generator = args.source / 'src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp'
    upstream = backend.read_text(encoding='utf-8')
    capability = upstream[upstream.index('static bool ggml_backend_vk_device_supports_op('):]
    # The original backend intentionally shares one body between these labels.
    matmul = capability[capability.index('case GGML_OP_MUL_MAT:'):capability.index('case GGML_OP_FLASH_ATTN_EXT:')]
    get_rows = capability[capability.index('case GGML_OP_GET_ROWS:'):capability.index('case GGML_OP_CPY:')]
    assert 'case GGML_OP_MUL_MAT_ID:' in matmul
    generated = generator.read_text(encoding='utf-8')
    for name in ('GGML_TYPE_Q3_K', 'GGML_TYPE_Q5_K'):
        assert 'case ' + name + ':' in get_rows, name
    for name in NEW.values():
        assert 'case ' + name + ':' in matmul, name
        assert 'case ' + name + ':' in get_rows, name
        assert '"' + name.removeprefix('GGML_TYPE_').lower() + '"' in generated, name
    rows = []
    for item in demand['gguf_ports']:
        quant = item.get('quantized_type')
        status = ('PLAIN_SOURCE_DOMAIN_REVIEW_REQUIRED' if not quant else
            'TYPE_MAPPED_RUNTIME_UNVERIFIED' if quant in current else
            'UNBOUND_ORIGINAL_VULKAN_BODY_PRESENT' if quant in NEW else
            'UNBOUND_SOURCE_QUALIFICATION_REQUIRED')
        rows.append({**item, 'vulkan_static_binding': status, 'runtime_verdict': 'UNVERIFIED'})
    record = {
        'baseline': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'kind': 'Original source/admission audit only; no runtime PASS',
        'counts': demand['counts'],
        'inputs_sha256': {str(demand_path.relative_to(ROOT)): digest(demand_path),
            str(provider.relative_to(ROOT)): digest(provider)},
        'upstream': {'revision': '456172ec733a135778adcd32d00e576a58232e45',
            'backend_sha256': digest(backend), 'generator_sha256': digest(generator)},
        'static_counts': dict(Counter(r['vulkan_static_binding'] for r in rows)),
        'unbound_types': dict(Counter(r['quantized_type'] for r in rows
            if r.get('quantized_type') and r['quantized_type'] not in current)),
        'initial_fix': {'type_bindings': NEW, 'contracts': sum(r.get('quantized_type') in NEW for r in rows),
            'embedding_types': sorted(NEW_EMBEDDING), 'embedding_contracts': sum(
                r['operation_candidate'] == 'Gather' and r.get('quantized_type') in NEW_EMBEDDING for r in rows)},
        'rows': rows, 'packed': demand['non_gguf_ports'], 'plain': demand['plain_ports'],
        'families': demand['families'],
        'limits': ['Source body/generator is not device/shape/numeric/resource/delivery qualification.',
            'All runtime verdicts are UNVERIFIED.', 'Three extended IQ1 formats still need Vulkan source qualification.'],
    }
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'static_counts': record['static_counts'], 'initial_fix': record['initial_fix']}, indent=2))

if __name__ == '__main__':
    main()
