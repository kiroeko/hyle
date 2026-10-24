# Original compressed-tensors CPU int4 preparation

Origin: vllm-project/compressed-tensors0.18.0
`ac8e2ba82f4e0d5eaa40069f8fc642738f124cd4`, Apache-2.0.
`compressed_tensors_cpu.h` translates the admitted complete four-bit path of
`compressors/pack_quantized/helpers.py::unpack_from_int32`,
`quantization/lifecycle/forward_helpers.py::_dequantize`, and `_process_group`
into ATen C++. Four bits divide32 exactly and group K is complete. The symmetric
path uses codes on axis1; the asymmetric path also follows
`pack_quantized/base.py::_decompress` for zero points packed on axis0, trimming
only padded output rows. Natural groups have no g_idx/permutation. Both signed
offset8 values, scale-dtype subtraction/multiplication and final output casting
are preserved. The selected Qwen3.8 artifact's actual zero-point payload is I32
packed storage, independently of its configuration's unpacked `zp_dtype=int8`.
No sparsity, model parsing, activation-order solver or HYLE GEMM is introduced.
