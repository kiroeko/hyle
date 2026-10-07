# Original bitsandbytes CPU NF4 sequences

Origin: bitsandbytes-foundation/bitsandbytes
`833649043474794b8fe7a4136e0c40faf077b2e0`, MIT.
`bitsandbytes_cpu.h` translates the original tensor steps in
`bitsandbytes/backends/default/ops.py::_dequantize_4bit_compute`,
`_dequantize_blockwise_compute`, and nested-scale restoration into ATen C++.
Decorators, autograd and model loading are absent. Complete K groups are
required by the HYLE schema; no remainder mathematics is replaced.
Shape checks, host mapping, allocator errors and output ownership belong to
the HYLE adapter. CPU execution belongs to the fixed LibTorch runtime.

CUDA carrier: same original tensor arithmetic and precision; tensors stay on the selected GPU.
AWQ/CT factories use original input device; constant shifts/permutation are cold prepared.
No new mathematical CUDA kernel; CPU product translation remains unchanged.
