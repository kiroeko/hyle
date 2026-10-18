import ast
import ctypes
import hashlib
import json
import math
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
out = root / 'cpu-carrier-probe'
out.mkdir(exist_ok=True)
ns = dict(torch=torch, Tensor=torch.Tensor, List=List, Optional=Optional,
          Tuple=Tuple, math=math, is_fake=is_fake, MBITS_F32=23, EBITS_F32=8,
          F32_EXP_BIAS=127, EBITS_F4_E2M1=2, MBITS_F4_E2M1=1,
          E8M0_EXPONENT_BIAS=127, E8M0_EXPONENT_NAN_VAL=255,
          _TORCH_VERSION_AT_LEAST_2_14=False)
borrowed = []

def load(path, names):
    source = path.read_text(encoding='utf-8')
    selected = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            original = ast.get_source_segment(source, node)
            borrowed.append(dict(path=str(path), symbol=node.name,
                                 sha256=hashlib.sha256(original.encode()).hexdigest()))
            node.decorator_list = []
            selected.append(node)
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), ns)

load(root / 'ao/torchao/quantization/quant_primitives.py',
     ['_RoundToFloat8', '_get_reduction_params', '_choose_scale_float8',
      '_maybe_expand_scale_to_tensor_shape', '_quantize_affine_float8'])
load(root / 'ao/torchao/prototype/mx_formats/mx_tensor.py',
     ['_f32_to_e8m0_rceil', '_e8m0_scale_to_reciprocal_fp32', '_to_mx_rceil'])
load(root / 'ao/torchao/prototype/custom_fp_utils.py',
     ['_n_ones', '_floatx_unpacked_to_f32'])
load(root / 'ao/torchao/prototype/mx_formats/kernels.py',
     ['f4_unpacked_to_f32', 'unpack_uint4', 'up_size'])

fp4 = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=np.float32)
fp8 = np.array([math.ldexp(1.0 + (i & 7) / 8, (i >> 3) - 7)
                if i >> 3 else math.ldexp(i & 7, -9) for i in range(127)], dtype=np.float32)

def quantize(a, power):
    data = a.reshape(-1, 128).float()
    if power:
        exponent, _ = ns['_to_mx_rceil'](data, data.abs().amax(-1, keepdim=True).clamp_min(1e-4), 448.)
        scale = ns['_e8m0_scale_to_reciprocal_fp32'](exponent).reciprocal()
    else:
        scale = ns['_choose_scale_float8'](data, [1, 128], hp_value_lb=1e-10)
    return ns['_quantize_affine_float8'](data, scale, torch.float8_e4m3fn), scale

def independent_quant(a, power):
    data = a.reshape(-1, 128)
    maximum = np.maximum(np.abs(data).max(-1, keepdims=True), np.float32(1e-4 if power else 1e-10))
    scale = maximum * np.float32(1 / 448) if power else maximum / np.float32(448)
    if power:
        mantissa, exponent = np.frexp(scale)
        scale = np.ldexp(np.ones_like(scale), exponent - (mantissa == .5)).astype(np.float32)
    normalized = np.clip(data / scale, -448., 448.)
    high = np.searchsorted(fp8, np.abs(normalized)).clip(0, 126)
    low = np.maximum(high - 1, 0)
    high_distance = np.abs(np.abs(normalized).astype(np.float64) - fp8[high])
    low_distance = np.abs(np.abs(normalized).astype(np.float64) - fp8[low])
    indices = np.where((high_distance < low_distance) | ((high_distance == low_distance) & ((high & 1) == 0)), high, low)
    decoded = np.copysign(fp8[indices], normalized).astype(np.float32)
    return decoded.reshape(a.shape), scale.reshape(*a.shape[:-1], -1)

def weights(e, n, k, four):
    # Format-valid sparse payload. Reference uses original positions/values,
    # independently of the target decoder and carrier parsing.
    codes = np.zeros((e, n, k), dtype=np.uint8)
    rows = np.arange(n)
    positions, values = [], []
    group = 32 if four else 128
    scales = np.empty((e, n if four else (n + 127) // 128, k // group), dtype=np.float32)
    blocks = np.arange(k // group)
    for ex in range(e):
        for term in range(2):
            col = (17 * rows + 31 + ex * 7 + term * (k // 2 + 13)) % k
            code = (1 + (rows + ex + term * 3) % 7) | (((rows + term) % 2) * (8 if four else 128))
            codes[ex, rows, col] = code.astype(np.uint8)
            value = (fp4[code & 7] if four else fp8[code & 127]) * np.where(code & (8 if four else 128), -1., 1.).astype(np.float32)
            positions.append((ex, col.copy()))
            values.append(value)
        if four:
            scales[ex] = np.ldexp(np.ones((n, k // group), dtype=np.float32), -5 + (rows[:, None] + blocks[None, :] + ex) % 3)
        else:
            scales[ex] = (17 + (ex * 5 + np.arange(scales.shape[1])[:, None] * 3 + blocks[None, :] + 13) % 11).astype(np.float32) / np.float32(1024)
    packed = (codes[..., ::2] | (codes[..., 1::2] << 4)) if four else codes
    carrier = np.concatenate([packed.reshape(-1), scales.view(np.uint8).reshape(-1)])
    return torch.from_numpy(carrier), scales, positions, values

def program(carrier, a, ids, n, k, e, four):
    if bool(((ids < 0) | (ids >= e)).any()):
        raise IndexError('selected expert is out of range')
    t, slots = ids.shape
    rows = t * slots
    selected = ids.reshape(-1).long()
    payload = e * n * k // (2 if four else 1)
    raw = carrier[:payload].reshape(e, n, k // (2 if four else 1)).index_select(0, selected).contiguous()
    scales = carrier[payload:].view(torch.float32).reshape(e, n if four else (n + 127) // 128, k // (32 if four else 128)).index_select(0, selected)
    if four:
        w = ns['f4_unpacked_to_f32'](ns['unpack_uint4'](raw))
    else:
        w = raw.view(torch.float8_e4m3fn).float()
        scales = scales.repeat_interleave(128, 1)[:, :n]
    q, aq_scale = quantize(a, four)
    q = q.reshape(t, a.shape[1], k).float()
    aq_scale = aq_scale.reshape(t, a.shape[1], k // 128)
    if a.shape[1] == 1:
        q = q.expand(t, slots, k)
        aq_scale = aq_scale.expand(t, slots, k // 128)
    group = 32 if four else 128
    block_count = k // group
    q = q.reshape(rows, block_count, 1, group)
    w = w.reshape(rows, n, block_count, group).permute(0, 2, 3, 1)
    dots = torch.matmul(q, w)
    aq_scale = aq_scale.reshape(rows, k // 128).repeat_interleave(4 if four else 1, 1)
    scaled = dots * aq_scale[:, :, None, None]
    scaled = scaled * scales.transpose(1, 2)[:, :, None, :]
    result = torch.zeros((rows, 1, n), dtype=torch.float32)
    for block in range(block_count):
        result = result + scaled[:, block]
    return result.reshape(t, slots, n).to(a.dtype)

def reference(a, ids, scales, positions, values, n, k, four):
    decoded, activation_scales = independent_quant(a, four)
    output = np.zeros((*ids.shape, n), dtype=np.float32)
    group = 32 if four else 128
    rows = np.arange(n)
    for token in range(ids.shape[0]):
        for slot, expert in enumerate(ids[token]):
            act_slot = 0 if a.shape[1] == 1 else slot
            first_pos, second_pos = positions[expert * 2][1], positions[expert * 2 + 1][1]
            partials = []
            for term, col in enumerate([first_pos, second_pos]):
                dot = decoded[token, act_slot, col] * values[expert * 2 + term]
                part = (dot * activation_scales[token, act_slot, col // 128]) * scales[expert, rows if four else rows // 128, col // group]
                partials.append(part.astype(np.float32))
            output[token, slot] = np.where(first_pos // group <= second_pos // group,
                partials[0] + partials[1], partials[1] + partials[0])
    return torch.from_numpy(output).to(torch.bfloat16)

results = []
started = time.monotonic()
with torch.inference_mode():
    packed = torch.arange(256, dtype=torch.uint8).reshape(1, 256)
    decoded = ns['f4_unpacked_to_f32'](ns['unpack_uint4'](packed)).numpy()
    nibble = np.stack([np.arange(256) & 15, np.arange(256) >> 4], -1).reshape(1, 512)
    expected = np.copysign(fp4[nibble & 7], np.where(nibble & 8, -1., 1.)).astype(np.float32)
    mismatch = int(np.count_nonzero(decoded.view(np.uint32) != expected.view(np.uint32)))
    results.append(dict(case='original-TorchAO-FP4-all-packed-bytes', elements=512, bit_mismatch=mismatch))
    for four, n, k in [(False, 4096, 6144), (False, 6144, 2048),
                       (False, 4096, 4096), (False, 4096, 2048),
                       (True, 3072, 7168), (True, 7168, 3072),
                       (True, 2048, 4096), (True, 4096, 2048)]:
        carrier, scales, positions, values = weights(3, n, k, four)
        for shared in [True, False]:
            # Duplicate expert IDs and changed inputs replay the same fixed shape.
            ids = torch.tensor([[2, 0, 2]], dtype=torch.int32)
            shape = (1, 1 if shared else 3, k)
            for seed in [0, 1]:
                a = torch.sin(torch.arange(math.prod(shape), dtype=torch.float32) * .013 + seed).reshape(shape).to(torch.bfloat16)
                actual = program(carrier, a, ids, n, k, 3, four)
                expected = reference(a.float().numpy(), ids.numpy(), scales, positions, values, n, k, four)
                mismatch = int(torch.count_nonzero(actual != expected))
                item = dict(case='encoded-grouped-FP4' if four else 'encoded-grouped-FP8', n=n, k=k,
                            shared=shared, seed=seed, elements=actual.numel(), bf16_mismatch=mismatch)
                results.append(item)
                print(json.dumps(item), flush=True)
        try:
            program(carrier, a, torch.tensor([[3]], dtype=torch.int32), n, k, 3, four)
        except IndexError:
            results.append(dict(case='invalid-expert-ID', n=n, k=k, four=four, rejected=True))
        else:
            raise AssertionError('invalid ID was accepted')
        del carrier, scales

receipt = dict(revision='a701b6a6058720c21f95908b7ae4a24bf0cae1b6', torch_version=torch.__version__,
               torch_commit=torch.version.git_version, seconds=time.monotonic() - started,
               threads=torch.get_num_threads(), borrowed=borrowed, results=results,
               scope='Native CPU mature tensor composition; exact encoded weights/real inner dimensions. No HYLE full region, C++ delivery, or P1 PASS.')
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
raise SystemExit(1 if any(any(v for k, v in r.items() if k.endswith('mismatch')) for r in results) else 0)
