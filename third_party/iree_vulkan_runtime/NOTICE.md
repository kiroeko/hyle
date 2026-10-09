# IREE Vulkan resource integration

Original runtime: https://github.com/iree-org/iree at
`6ef606b6fa3e12bd2c12d36a07463069eb71185f` (Apache-2.0 WITH LLVM-exception).

The exact patch manifest applies only to authenticated build-local source.
It adds synchronous external queue ownership and actual VkDeviceMemory
allocation/free accounting hooks, retaining original allocators, pools, queue
progress and numerical programs. The bytecode build generator uses CMake's
selected Python interpreter instead of a process-wide `python3` alias.
No original source cache or compiler mathematics is changed.

Build integration privately renames the original Vulkan header targets to allow
composition with the separately locked HYLE/ggml headers. Offline parameterized
programs preserve the qualified original scan, conversion, reduction and MatMul
recipes. Their byte identities are in programs/manifest.json; no compiler or
Python is required by an installed C++ consumer.

External queue lookup preserves the actual VkDeviceQueueCreateFlags supplied
by its owner, including internally synchronized queues. This changes resource
wrapping only; the original scheduling and math remain unchanged.

The external wrapper (including distinct logical lanes on one native queue) exposes original stop/drain completion through
a retained HAL semaphore. No new worker, scheduler or math is introduced.

The POSIX and Windows IOCP semaphore completion paths clear their operation tracker reference
before the final callback, which may free the operation under the original
async ownership contract. This fixes a reproduced host use-after-free without
changing numerical programs or adding a new execution mechanism.

## Original artifact tensor recipes

`programs/artifacts.mlir` is the original NF4, AutoAWQ and compressed-tensors
decode sequence expressed in standard tensor/arith/linalg operations. The
locked IREE compiler retains its own lowering, workgroup selection and runtime.
No HYLE dequantization kernel, quantizer or custom compiler pass is introduced.
The maintainer generator preserves the original scale dtype rounding before
the separately requested output conversion. Explicit decode output is charged
to that operation; it is not a hidden persistent F32 model weight bank.

- bitsandbytes `833649043474794b8fe7a4136e0c40faf077b2e0`,
  `bitsandbytes/backends/default/ops.py` `_dequantize_4bit_compute` and
  `_dequantize_blockwise_compute`; Copyright (c) Facebook, Inc. and its
  affiliates, MIT. See `programs/licenses/bitsandbytes.txt`.
- AutoAWQ `88e4c76b20755db275574e6a03c83c84ba3bece5`,
  `awq/utils/packing_utils.py` unpack/reverse order and dequantize;
  Copyright (c) 2023 MIT HAN Lab, MIT. See `programs/licenses/awq.txt`.
- compressed-tensors `ac8e2ba82f4e0d5eaa40069f8fc642738f124cd4`,
  original `unpack_from_int32` and symmetric grouped dequantization;
  Copyright contributors to the vLLM project, Apache-2.0.
  See `programs/licenses/compressed-tensors.txt`.

Original source identities and parent tensor borrows are registered in
`third_party/{bitsandbytes,awq,compressed_tensors}_cpu/SOURCE.json` and their
notices. The new recipe/bytecode/exports are recorded in programs/manifest.json;
offline compilation is recorded separately from runtime qualification.

## Original MLX packed tensor programs

`programs/mlx.mlir` expresses the original MLX group sequences in standard
tensor/arith/linalg/scf operations. Origin: ml-explore/mlx
`2b2bb49790158ecd008d1685c94e3ef119792a00`, Copyright © 2023-2026 Apple Inc.,
MIT; see `programs/licenses/mlx.txt` and the registered source identities in
`third_party/mlx_cpu_quant/SOURCE.json`. Source functions are
`backend/cpu/quantized.cpp::_qmm_t_simd`, `fp_qmm_t_simd` and
`dequantize_scale<T,32>`, `ops.cpp::affine_dequantize`, and
`python/mlx/nn/layers/quantized.py::QuantizedEmbedding::__call__`.

Original nibble order, FP4_LUT, E4M3FN conversion and E8 scale semantics remain
unchanged. MX dot precedes each source-typed group scale, with F32 accumulation.
Only selected experts and one K group are expanded; embedding selects rows
before typed multiplication/addition. IREE's original compiler owns MatMul,
loop execution, workgroup selection and resources. No HYLE kernel/quantizer,
compiler pass or full F32 expert bank is introduced. Tensor barriers and shape
ties preserve source dtype materialization; their byte identities and offline
compilation are recorded in the existing manifest and qualification receipt.

## Original official float and QAT programs

`programs/official.mlir` translates the registered original DeepSeek
weight_dequant/activation and FP4 group equations, SGLang `_per_token_group_quant_8bit_colmajor`
and `_w8a8_block_fp8_matmul`, and ds4
`9139e2ae58a41503968a500f36f75895c1ba63fc` original BF16 QAT roundtrip equations.
Source identities/functions are in `third_party/deepseek_fp8/provenance.json`,
`third_party/sglang_activation/fp8/source.json` and
`third_party/ds4_quantization/source.json`. See `programs/licenses/deepseek.txt`,
`sglang.txt` and `ds4.txt`; original copyrights and terms are retained.

Only standard tensor/arith/linalg/scf operations are emitted. Original IREE
owns native conversion, matrix execution, group loops and resource management.
FP4 nearest-even lookup and low BF16 significand handling preserve the original
format and already qualified source tensor equations; no new quantization
algorithm, HYLE kernel or compiler pass is introduced. Dynamic finite control
returns a native scalar flag; failed output remains private. Original dot then
activation scale, weight scale and ordered F32 accumulation are retained.
