# Qwen3.5 CUDA R04 RoPE admission audit

Baseline `1c826fa7b` retains a mismatch between the ggml CUDA adapter's cold
qualification and the pinned upstream CUDA RoPE implementation. The adapter
admits BF16, while upstream `rope.cu` executes only F16/F32.

The full official Qwen3.5-2B CUDA Direct batch used the local checkpoint
fixture and ordinary route. R02 FFN completed against its independent oracle.
The following R04 test then prepared its exact GQA 8/2 by 256 BF16 nodes and
reached the upstream assertion during invocation initialization/execution:

```text
CudaDefaultCompute.D5Qwen35OfficialGqaNormRopeAndAttentionUsePreparedCalls
rope.cu: GGML_ASSERT(src0->type == GGML_TYPE_F32 ||
                     src0->type == GGML_TYPE_F16) failed
```

The adapter already owns upstream Cast nodes, casts BF16 for other fixed
upstream type boundaries, validates lowered nodes with
`ggml_backend_supports_op`, and casts results back to the public output type.
This evidence therefore fixes D5-F7 to the existing Cast/RoPE graph and its
type-domain regression. It does not authorize a new RoPE implementation,
kernel, upstream patch, fallback, route reorder, or hot selection.

Machine: Windows 11, NVIDIA GeForce RTX 4070, compute capability 8.9, CUDA
Toolkit 13.3.73. Candidate execution used the reusable Release static CUDA
factor tree and the fixture from `Qwen/Qwen3.5-2B@15852e8`.
