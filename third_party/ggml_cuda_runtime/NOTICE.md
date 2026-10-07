# ggml CUDA resource integration patches

Upstream: https://github.com/ggml-org/ggml at
`456172ec733a135778adcd32d00e576a58232e45` (MIT; see LICENSE).

The owning CMake applies the exact context-matched edits in `patches.json` to
a build-local copy of authenticated upstream source. Shared source caches and
upstream checkouts remain unchanged. The manifest pins both original and
result hashes; this directory does not contain a second runtime distribution.

Edits preserve native CUDA Runtime, Driver and cuBLAS errors at the private
C++ adapter boundary. Resource owners roll back partial construction; cleanup
does not throw. Capture begin/end/encoding failures release the global capture
counter and discard incomplete graphs. Virtual-memory mapping failures release
their allocation handle and partial mapping. Non-owning buffer wrappers never
free external tensor payload. Backend streams, pools, algorithms, fusion and
capture selection remain upstream-owned. No mathematical kernels change.

The private bridge implementation is in `src/compute_provider_ggml_cuda` and
is compiled into the upstream translation unit. It is not an installed API.
These source edits alone do not certify ordinary Direct/Graph execution or
failure recovery; those need the M1-C native and delivery receipts.

The narrow IQ1_XS/XXS/XXXS CUDA programs in `iq1-patches.json` and the three
MMQ template instances originate from Unsloth llama.cpp
`46cbf0e95786fe8f5b7c0e86d57aaf8f8eceea7f` (MIT, same LICENSE).
Original CUDA tables, conversion, dot products, MMVQ/MMQ math and dispatch
blocks remain unchanged. The newer core's three block-size traits and the
template include path are ABI/build adaptations. `iq1-origins.json` retains
the original block and template SHA-256 identities. They share the existing
physical types/codecs with the original CPU port, not a second ggml runtime.
Only the CUDA route compiles these programs; CPU-only/Metal/Vulkan pins and
math remain unchanged. This source port is not runtime or performance proof.

The F32 MMF tiles in this pin execute TF32 multiplication. The original regular
MUL_MAT, MUL_MAT_ID and corresponding capture-suitability selection are guarded
so a requested native F32 operand contract on F32 weights reaches the original
F32 cuBLAS/sorting program instead. Default precision, quantized and F16/BF16
dispatch and every mathematical kernel remain upstream. This is a private
precision-selection integration change, not a new GEMM implementation.
