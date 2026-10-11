"""Use the mature gguf reader/writer to check a small mixed-encoding consumer artifact.

python prepare_gguf.py OUTPUT_DIRECTORY creates the deterministic GGUF fixture.
python prepare_gguf.py OUTPUT_DIRECTORY INPUT.gguf checks an existing fixture with
the same eight names/encodings/shapes/bytes. This is not a full model loader.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import gguf
import numpy as np

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
model = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else out / 'mixed-fixture.gguf'
expected = {}
for lane in range(2):
    left = np.array([0x3f80, 0x4000, 0x4040, 0x4080], dtype='<u2').reshape(2, 2)
    if lane: left ^= np.uint16(0x8000)
    right = np.array([0x40a0, 0x40c0, 0x40e0, 0x4100], dtype='<u2').reshape(2, 2)
    activation = np.full((1, 32), .25 if lane == 0 else .5, dtype='<f4')
    q8 = np.empty((2, 34), dtype=np.uint8)
    q8[:, 0] = 0; q8[:, 1] = 0x3c
    q8[0, 2:] = 1; q8[1, 2:] = 0xfe
    for index, (data, shape, kind) in enumerate([
        (left, (2, 2), gguf.GGMLQuantizationType.BF16),
        (right, (2, 2), gguf.GGMLQuantizationType.BF16),
        (activation, (1, 32), gguf.GGMLQuantizationType.F32),
        (q8, (2, 32), gguf.GGMLQuantizationType.Q8_0),
    ]):
        expected[f'lane{lane}.tensor{index}'] = (data, shape, kind)
if len(sys.argv) == 2:
    if model.exists(): raise ValueError('refuse to overwrite existing fixture')
    writer = gguf.GGUFWriter(model, 'hyle_f3_fixture')
    for name, (data, shape, kind) in expected.items():
        # Quantized byte shape is decoded by the mature writer to logical shape.
        writer.add_tensor(name, data, raw_dtype=kind)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
reader = gguf.GGUFReader(model)
actual = {tensor.name: tensor for tensor in reader.tensors}
rows = []
for name, (data, shape, kind) in expected.items():
    tensor = actual[name]
    if tensor.tensor_type != kind or list(tensor.shape) != list(reversed(shape)):
        raise ValueError('GGUF fixture encoding/shape differs: ' + name)
    payload = tensor.data.tobytes()
    if payload != data.tobytes(): raise ValueError('GGUF fixture original payload differs: ' + name)
    rows.append({'name': name, 'encoding': kind.name, 'shape': list(shape),
                 'file_offset': tensor.data_offset, 'bytes': tensor.n_bytes,
                 'sha256': hashlib.sha256(payload).hexdigest()})
lines = ['HYLE_F3_GGUF_FIXTURE_V1', str(model), str(model.stat().st_size)]
lines.extend(str(row['file_offset']) + '\t' + str(row['bytes']) for row in rows)
(out / 'gguf-ranges.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
receipt = {'kind': 'Mature GGUF parser checked input, not execution evidence',
           'gguf_version': importlib.metadata.version('gguf'), 'source': str(model),
           'file_bytes': model.stat().st_size, 'ranges': rows,
           'scope': 'Deterministic BF16/F32/GGUF Q8 mixed fixture, not full model inference'}
(out / 'gguf-checked-input.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print('Checked eight original GGUF ranges with mature reader.')
