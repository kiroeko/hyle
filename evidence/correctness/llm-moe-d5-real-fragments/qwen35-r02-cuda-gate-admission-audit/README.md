# Qwen3.5 CUDA R02 gate admission audit

Baseline `8b9dc42abe47439317d61dd733474bc0cf119b50` prepares the official BF16
Qwen3.5-2B R02 separate SwiGLU through the ordinary CUDA Direct entry, but
binds the frozen `hyle-cuda` fallback instead of a mature Provider.

The exact `[1,6144]` test consumer ran on the Windows canonical workspace with
the Qwen workload profile. Preparation succeeded and the test failed only on
the Provider identity assertions:

```text
CudaDefaultCompute.D5Qwen35OfficialFfnAdmissionDoesNotUseFrozenFallback
prepared provider: hyle-cuda
expected provider: ggml-cuda
1 test, 0 passed, 1 failed (80 ms)
```

The pinned ggml `456172ec733a135778adcd32d00e576a58232e45-hyle` source exposes
`ggml_swiglu` and `ggml_swiglu_split`. Its CUDA GLU dispatch accepts separate
inputs and F16/F32 storage; the HYLE ggml CUDA adapter currently neither
qualifies nor lowers `GatedActivation`. The adapter already owns upstream Cast
nodes and validates every lowered node with `ggml_backend_supports_op`.

This evidence fixes D5-F6 to the existing upstream operation plus a BF16/F32
boundary Cast. It does not authorize new math, CUDA kernels, upstream changes,
route reordering, other activation kinds, or a native fallback exception.

Machine: Windows 11, NVIDIA GeForce RTX 4070, compute capability 8.9, 12,281
MiB reported VRAM, CUDA Toolkit 13.3.73. The baseline test used the reusable
Release static CUDA factor tree with `CMAKE_CUDA_ARCHITECTURES=89-real`.
