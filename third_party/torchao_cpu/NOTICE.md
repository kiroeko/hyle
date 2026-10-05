# Original TorchAO CPU tensor sequences

Source: `pytorch/ao` at `a701b6a6058720c21f95908b7ae4a24bf0cae1b6`, BSD-3-Clause.
Runtime: machine-supplied CPU-only LibTorch `2.8.0+cpu`.

`torchao_cpu.h` spells the following original tensor operations in ATen C++:

- `prototype/mx_formats/mx_tensor.py::_f32_to_e8m0_rceil` and
  `_e8m0_scale_to_reciprocal_fp32`, using the original pre-2.14 fallback;
- `prototype/custom_fp_utils.py::_floatx_unpacked_to_f32` specialized to the
  original E2M1 constants, preserving the zero/denormal/normal/sign operations;
- `prototype/mx_formats/kernels.py::unpack_uint4` and `f4_unpacked_to_f32`;
- `quantization/quant_primitives.py::_choose_scale_float8` positive lower-bound
  reduction path, `_quantize_affine_float8`, and `_RoundToFloat8.forward`.

Package-registration decorators and backward/autograd support are not needed
for inference. HYLE validates shapes/finite activation/IDs and composes ATen
matrix calls. No original mathematical rule is replaced with a HYLE codec or
scalar dot product. Original source/file/function hashes and native numerical
qualification are recorded in
`evidence/research/llm-moe-d5-cpu-apple-official-completion`.

CPU SDK headers/libraries are external, have PyTorch's own BSD-3-Clause and
transitive notices, and are not vendored in this directory. HYLE static does
not mean a statically linked LibTorch runtime.
