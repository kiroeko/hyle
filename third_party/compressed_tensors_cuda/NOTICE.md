# Original compressed-tensors CUDA int4 preparation

Origin: vllm-project/compressed-tensors0.18.0
`ac8e2ba82f4e0d5eaa40069f8fc642738f124cd4`, Apache-2.0.
`compressed_tensors_cuda.h` translates the admitted complete four-bit paths of
`compressors/pack_quantized/base.py::PackedQuantizationCompressor.decompress`,
`compressors/pack_quantized/helpers.py::unpack_from_int32`,
`quantization/lifecycle/forward_helpers.py::_dequantize`, and `_process_group`
into original ATen CUDA tensor operations. Four bits divide32 exactly and K
contains complete groups. Symmetric codes use the original signed offset8.
For asymmetric natural column groups, codes are packed along K while zero
points are packed along N; the latter is unpacked and truncated to the real N
before dequantization. Both stored payloads are I32 words, not unpacked int8.
Signed offset8, scale-dtype subtraction and scale-dtype multiplication are
preserved; an explicit output conversion follows that product. No initialized
g_idx/permutation input or global scale is admitted by these recipes.
No sparsity, model parsing, activation-order solver or HYLE GEMM is introduced.

Tensors stay on the selected GPU and use the existing prepared operation and
original Torch CUDA stream. Constant shifts are cold prepared; the AWQ-specific
permutation remains confined to its separate recipe. No new mathematical CUDA
kernel, execution protocol or capture capability is introduced.
