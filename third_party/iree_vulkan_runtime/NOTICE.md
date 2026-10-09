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
