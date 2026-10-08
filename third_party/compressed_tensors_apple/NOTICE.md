# Original compressed-tensors symmetric4 preparation on MLX GPU

Origin: vllm-project/compressed-tensors0.18.0
`ac8e2ba82f4e0d5eaa40069f8fc642738f124cd4`, Apache-2.0.
The admitted complete four-bit path of
`compressors/pack_quantized/helpers.py::unpack_from_int32` and
`quantization/lifecycle/forward_helpers.py::_dequantize`/`_process_group`
is expressed in MLX GPU operations. Low nibble first, offset8, complete K
groups and the original cast-to-scale-dtype before multiplication are kept.
This current artifact has no g_idx or zero-point. Masked U32 shifts preserve
the original signed-word nibbles. No sparse math, HYLE GEMM or shader is added.
Native output/resource/delivery qualification remains separate.
