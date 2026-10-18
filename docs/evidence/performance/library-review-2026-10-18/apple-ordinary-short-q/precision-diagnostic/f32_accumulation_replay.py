"""Arithmetic hypothesis only; not a GPU kernel trace or a passing oracle."""
from pathlib import Path
import json
import struct


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


keys = 8193
probability = f32(1 / keys)
separate = fused = 0.0
for index in range(keys):
    value = 1.125 if index % 2 == 0 else -1.375
    separate = f32(separate + f32(probability * value))
    # These dyadic F32 operands have an exact product/sum in Python's F64.
    fused = f32(fused + probability * value)
reference = f32(1.25 / keys - 0.125)
observed = -0.1248360276222229
result = {
    'kind': 'CPU explicit-F32 arithmetic replay, no native/GPU execution',
    'keys': keys,
    'probability_f32': probability,
    'probability_mass_f64': keys * probability,
    'reference_f32': reference,
    'separate_f32_product_sum': separate,
    'fma_f32_replay': fused,
    'previously_observed_first_gpu_scalar': observed,
    'sequential_vs_gpu_abs': abs(separate - observed),
    'sequential_vs_reference_abs': abs(separate - reference),
    'probability_rounding_only_error_f64':
        abs((keys * probability - 1) * (1.25 / keys - 0.125)),
    'interpretation': 'Strong PV accumulation hypothesis only. Does not prove '
        'the actual selected kernel, internal P, full output parity or accuracy '
        'of either route. Original FP64 budget failures are retained.'
}
path = Path(__file__).with_suffix('.json')
path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result))
