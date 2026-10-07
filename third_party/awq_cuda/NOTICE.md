# Original AutoAWQ CPU unpack/dequantization

Origin: casper-hansen/AutoAWQ `88e4c76b20755db275574e6a03c83c84ba3bece5`, MIT.
`awq_cpu.h` translates `awq/utils/packing_utils.py::unpack_awq`,
`reverse_awq_order`, and `dequantize_gemm` into ATen C++.
It preserves column packing, the original eight-column reverse permutation,
actual GEMM zero points and scale dtype. It does not substitute an approximate
centered-int4 bias or implement a HYLE dot product. The final contiguous
transpose exports the original decoded weights in the explicit [N,K] layout.

CUDA carrier: same original tensor arithmetic and precision; tensors stay on the selected GPU.
AWQ/CT factories use original input device; constant shifts/permutation are cold prepared.
No new mathematical CUDA kernel; CPU product translation remains unchanged.
