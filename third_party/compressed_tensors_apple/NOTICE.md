# Original compressed-tensors four-bit preparation on MLX GPU

Origin: vllm-project/compressed-tensors0.18.0
`ac8e2ba82f4e0d5eaa40069f8fc642738f124cd4`, Apache-2.0.
The admitted complete four-bit path of
`compressors/pack_quantized/helpers.py::unpack_from_int32` and
`quantization/lifecycle/forward_helpers.py::_dequantize`/`_process_group`
is expressed in MLX GPU operations. Low nibble first, offset8, complete K
groups and the original cast-to-scale-dtype before multiplication are kept.
The symmetric two-port path has no zero-point. The explicit asymmetric
three-port path unpacks codes along K and zero-points along N (discarding
padded N lanes), with natural complete K groups and no g_idx. Subtract and
multiply each retain the scale dtype storage boundary before a requested
output cast. Same-width MLX View bitcasts stop compile fusion at these
boundaries and share storage without copying a decoded weight payload.
Masked U32 shifts preserve the original signed-word nibbles.
No sparse math, HYLE GEMM, shader, model-specific route or extra executor is added.
Native output/resource/delivery qualification remains separate.
