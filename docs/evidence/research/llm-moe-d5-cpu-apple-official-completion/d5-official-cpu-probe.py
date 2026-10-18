import ast
import ctypes
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ctypes.windll.kernel32.SetErrorMode(1 | 2 | 0x8000)
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from torch._subclasses.fake_tensor import is_fake

torch.set_num_threads(2)
root = Path('.scratch/validation/d5-cpu-apple-official-sources')
out = root / 'cpu-probe'
out.mkdir(exist_ok=True)
namespace = {'torch': torch, 'List': List, 'Optional': Optional, 'Tuple': Tuple,
             'math': math, 'is_fake': is_fake, 'MBITS_F32': 23,
             'E8M0_EXPONENT_BIAS': 127, 'E8M0_EXPONENT_NAN_VAL': 255,
             '_TORCH_VERSION_AT_LEAST_2_14': False}
borrowed = []

def load_original(path, names):
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            original = ast.get_source_segment(source, node)
            # Disable only package registration; function/class bodies stay intact.
            node.decorator_list = []
            nodes.append(node)
            borrowed.append({'path': str(path), 'symbol': node.name,
                             'sha256': hashlib.sha256(original.encode()).hexdigest()})
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)

load_original(root / 'ao/torchao/quantization/quant_primitives.py',
              ['_RoundToFloat8', '_get_reduction_params', '_choose_scale_float8',
               '_maybe_expand_scale_to_tensor_shape', '_quantize_affine_float8'])
load_original(root / 'ao/torchao/prototype/mx_formats/mx_tensor.py',
              ['_f32_to_e8m0_rceil', '_e8m0_scale_to_reciprocal_fp32', '_to_mx_rceil'])

table = np.array([math.ldexp(1.0 + (i & 7) / 8, (i >> 3) - 7)
                  if i >> 3 else math.ldexp(i & 7, -9) for i in range(127)], dtype=np.float32)

def independent(x, power):
    grouped = x.reshape(-1, 128)
    maximum = np.maximum(np.max(np.abs(grouped), axis=1, keepdims=True),
                         np.float32(1e-4 if power else 1e-10))
    scaled = maximum * np.float32(1 / 448) if power else maximum / np.float32(448)
    if power:
        mantissa, exponent = np.frexp(scaled)
        scales = np.ldexp(np.ones_like(scaled), exponent - (mantissa == .5)).astype(np.float32)
    else:
        scales = scaled.astype(np.float32)
    value = np.clip(grouped / scales, -448, 448).astype(np.float32)
    magnitude = np.abs(value)
    high = np.searchsorted(table, magnitude).clip(0, 126)
    low = np.maximum(high - 1, 0)
    dh = np.abs(magnitude.astype(np.float64) - table[high])
    dl = np.abs(magnitude.astype(np.float64) - table[low])
    codes = np.where((dh < dl) | ((dh == dl) & ((high & 1) == 0)), high, low)
    codes = (codes | (np.signbit(value).astype(np.uint8) << 7)).astype(np.uint8)
    return codes, scales

def mature(x, power):
    data = x.reshape(-1, 128).to(torch.float32)
    if power:
        maximum = data.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        exponent, _ = namespace['_to_mx_rceil'](data, maximum, 448.0)
        reciprocal = namespace['_e8m0_scale_to_reciprocal_fp32'](exponent)
        scales = reciprocal.reciprocal()
    else:
        scales = namespace['_choose_scale_float8'](data, [1, 128], hp_value_lb=1e-10)
    codes = namespace['_quantize_affine_float8'](data, scales, torch.float8_e4m3fn)
    return codes, scales

results = []
started = time.monotonic()
with torch.inference_mode():
    old = Path('.scratch/validation/d5-mature-native-q1')
    for filename in ['all_finite_bf16.bin', 'ordinary.bin', 'tiny_and_scale.bin']:
        raw = (old / filename).read_bytes()
        values = np.frombuffer(raw, dtype=np.float32).copy()
        for power in [False, True]:
            expected_codes, expected_scales = independent(values, power)
            codes, scales = mature(torch.from_numpy(values), power)
            actual_codes = codes.view(torch.uint8).numpy()
            actual_scales = scales.numpy()
            item = {'input': filename, 'input_sha256': hashlib.sha256(raw).hexdigest(),
                    'mode': 'DeepSeek-power2' if power else 'GLM-ordinary',
                    'elements': values.size,
                    'code_mismatch': int(np.count_nonzero(actual_codes != expected_codes)),
                    'scale_bit_mismatch': int(np.count_nonzero(actual_scales.view(np.uint32) != expected_scales.view(np.uint32)))}
            results.append(item)
            print(json.dumps(item), flush=True)
    # A cold-traced native program exercises ATen matrix kernels and the exact
    # group scaling order already used by the admitted Apple FP8 composition.
    for rows in [1, 3]:
        for slots in [1, 2]:
            k, n, experts = 256, 576, 3
            data = torch.sin(torch.arange(rows * k, dtype=torch.float32) * .13).reshape(rows, k).to(torch.bfloat16)
            codes, scales = mature(data, False)
            w = torch.zeros(experts, n, k, dtype=torch.float32)
            w[:, :, 7] = .5
            w[:, :, 143] = -2
            ws = torch.full((experts, (n + 127) // 128, k // 128), 1.25)
            ids = torch.arange(rows * slots, dtype=torch.int64).reshape(rows, slots) % experts
            weights = w.index_select(0, ids.reshape(-1)).reshape(rows * slots, n, 2, 128).permute(0, 2, 3, 1)
            activation = codes.to(torch.float32).reshape(rows, 1, 2, 128).expand(rows, slots, 2, 128).reshape(rows * slots, 2, 1, 128)
            dots = torch.matmul(activation, weights)
            a_scales = scales.reshape(rows, 1, 2, 1, 1).expand(rows, slots, 2, 1, 1).reshape(rows * slots, 2, 1, 1)
            weight_scales = ws.index_select(0, ids.reshape(-1)).repeat_interleave(128, 1)[:, :n, :].transpose(1, 2).unsqueeze(2)
            scaled = (dots * a_scales) * weight_scales
            actual = (scaled[:, 0] + scaled[:, 1]).reshape(rows, slots, n).to(torch.bfloat16)
            ec, es = independent(data.float().numpy(), False)
            decoded = table[ec & 127] * np.where(ec & 128, -1, 1).astype(np.float32)
            expected = (decoded[:, 7] * np.float32(.5) * es.reshape(-1)[:rows * 2:2] * np.float32(1.25)
                        + decoded[:, 15] * np.float32(-2) * es.reshape(-1)[1:rows * 2:2] * np.float32(1.25)) if False else None
            first = decoded.reshape(rows, 2, 128)[:, 0, 7] * np.float32(.5)
            second = decoded.reshape(rows, 2, 128)[:, 1, 15] * np.float32(-2)
            base = ((first * es.reshape(rows, 2)[:, 0]) * np.float32(1.25)
                    + (second * es.reshape(rows, 2)[:, 1]) * np.float32(1.25)).astype(np.float32)
            expected = torch.from_numpy(np.broadcast_to(base[:, None, None], (rows, slots, n)).copy()).to(torch.bfloat16)
            item = {'case': 'ATen-selected-block-GEMM-composition', 'rows': rows, 'slots': slots,
                    'k': k, 'n': n, 'elements': actual.numel(),
                    'output_mismatch': int(torch.count_nonzero(actual != expected))}
            results.append(item)
            print(json.dumps(item), flush=True)

receipt = {'source_revision': 'a701b6a6058720c21f95908b7ae4a24bf0cae1b6',
           'torch_version': torch.__version__, 'torch_commit': torch.version.git_version,
           'machine': platform.platform(), 'threads': torch.get_num_threads(),
           'seconds': time.monotonic() - started, 'borrowed': borrowed, 'results': results,
           'scope': 'Original CPU group128 functions and ATen composition qualification; no HYLE ordinary entry/delivery/P1 or full model PASS.'}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
raise SystemExit(1 if any(any(v for k, v in r.items() if k.endswith('mismatch')) for r in results) else 0)
