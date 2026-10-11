"""Official Safetensors adapter for the explicit local Qwen F3 consumer.

Usage: python prepare_qwen.py MODEL.safetensors OUTPUT_DIRECTORY
Requires the official `safetensors` Python package. No Torch, download, or
whole-checkpoint materialization/hash is performed. Only the declared three
MLP projections and one F32 norm are read for checked payload/reference inputs.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import sys
from safetensors import safe_open

model, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
with model.open('rb') as file:
    header_size = struct.unpack('<Q', file.read(8))[0]
    if header_size > 16 * 1024 * 1024:
        raise ValueError('unexpectedly large Safetensors header')
    header = file.read(header_size)
metadata = json.loads(header)
base = header_size + 8
names = ['model.language_model.layers.0.mlp.gate_proj.weight',
         'model.language_model.layers.0.mlp.up_proj.weight',
         'model.language_model.layers.0.mlp.down_proj.weight',
         'model.language_model.layers.0.linear_attn.norm.weight']
out.mkdir(parents=True, exist_ok=True)
rows, bounded = [], base
with safe_open(str(model), framework='np') as official, model.open('rb') as file:
    for name in names:
        tensor = metadata[name]
        view = official.get_slice(name)
        if view.get_shape() != tensor['shape'] or view.get_dtype() != tensor['dtype']:
            raise ValueError('official shape/dtype disagreement')
        begin, end = tensor['data_offsets']
        size = end - begin
        if not 0 <= begin <= end or size > 32 * 1024 * 1024:
            raise ValueError('declared tensor exceeds bounded input policy')
        file.seek(base + begin)
        payload = file.read(size)
        if len(payload) != size:
            raise ValueError('short declared tensor payload')
        bounded += size
        shape = tensor['shape']
        if tensor['dtype'] == 'BF16' and len(shape) == 2:
            rows_count, width = shape
            reference = []
            for row in range(rows_count):
                left_bits = struct.unpack_from('<H', payload, row * width * 2)[0]
                right_bits = struct.unpack_from('<H', payload, (row * width + width - 1) * 2)[0]
                left = struct.unpack('<f', struct.pack('<I', left_bits << 16))[0]
                right = struct.unpack('<f', struct.pack('<I', right_bits << 16))[0]
                reference.append(left * 0.125 - right * 0.0625)
        elif tensor['dtype'] == 'F32' and shape == [128]:
            reference = [x * (1.0 + (i % 7) * 0.125) for i, x in enumerate(struct.unpack('<128f', payload))]
        else:
            raise ValueError('unexpected local Qwen mixed-dtype tensor')
        digest = hashlib.sha256(payload).hexdigest()
        rows.append({'name': name, 'dtype': tensor['dtype'], 'shape': shape,
                     'file_offset': base + begin, 'bytes': size, 'payload_sha256': digest})
        (out / (name.replace('.', '_') + '.reference.txt')).write_text(
            '\n'.join(format(x, '.17g') for x in reference) + '\n', encoding='utf-8')
lines = ['HYLE_F3_QWEN_RANGES_V1', str(model), str(model.stat().st_size)]
for item in rows:
    shape = item['shape']
    lines.append('\t'.join([item['name'], item['dtype'], str(item['file_offset']), str(item['bytes']),
                           str(shape[0]), str(shape[1] if len(shape) > 1 else 1), item['payload_sha256']]))
(out / 'ranges.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
receipt = {'kind': 'checked local input, not an execution verdict', 'source': str(model),
           'safetensors_version': importlib.metadata.version('safetensors'),
           'header_sha256': hashlib.sha256(header).hexdigest(), 'bounded_bytes_read': bounded,
           'whole_model_read_or_hashed': False, 'ranges': rows}
(out / 'checked-input.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print('Prepared four official checked ranges; bounded bytes:', bounded)
