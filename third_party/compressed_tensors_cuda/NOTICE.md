# Original compressed-tensors CPU symmetric int4 preparation

Origin: vllm-project/compressed-tensors0.18.0
`ac8e2ba82f4e0d5eaa40069f8fc642738f124cd4`, Apache-2.0.
`compressed_tensors_cpu.h` translates the admitted complete four-bit path of
`compressors/pack_quantized/helpers.py::unpack_from_int32`,
`quantization/lifecycle/forward_helpers.py::_dequantize`, and `_process_group`
into ATen C++. Four bits divide32 exactly; group K is complete, so padding and
cross-word cases are outside this schema. Actual target headers have no
g_idx/zero-point. Signed offset8 and scale-dtype rounding are preserved.
No sparsity, model parsing, activation-order solver or HYLE GEMM is introduced.

CUDA carrier: same original tensor arithmetic and precision; tensors stay on the selected GPU.
AWQ/CT factories use original input device; constant shifts/permutation are cold prepared.
No new mathematical CUDA kernel; CPU product translation remains unchanged.
