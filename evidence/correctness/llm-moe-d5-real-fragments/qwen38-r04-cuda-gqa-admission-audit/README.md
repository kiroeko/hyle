# Qwen3.8 CUDA R04 GQA admission audit

- Date: 2026-09-25
- Windows product checkout baseline: `bc7c5d66c17861bea973d59fd272e61fffb77c99`
- Machine: Windows, NVIDIA GeForce RTX 4070, SM89, 12,281 MiB VRAM
- Toolchain: MSVC 19.50, CUDA Toolkit 13.3.73
- Owner: `compute_provider_ggml_cuda`

## Reproduction

The ordinary CUDA Direct consumer used the exact official Qwen3.8-27B R04
shape: BF16 BSHD query `[1,1,24,256]`, key/value `[1,1,4,256]`, causal
attention, no explicit mask, scale 0.0625 and F32 accumulation. RMSNorm and
partial interleaved RoPE prepared successfully. Preparing SDPA failed with:

```text
scaled-dot-product-attention: hyle-cuda: CUDA tensor provider does not implement this operation family
```

The same executable already accepts the Qwen3.5 8/2x256 form. This is a cold
prepare rejection, before execution or timing; no performance result is
involved.

## Source boundary

`src/compute_provider_ggml_cuda/ggml_cuda_provider.cc` already lowers BF16
attention to one fixed ggml CUDA program: cast/permute, `ggml_mul_mat`, scale,
causal mask, softmax and value projection. ggml performs head-batch broadcast.
The adjacent `internal::supports` check nevertheless requires `q[2] == 8` and
`k[2] == 2`. The lowering contains no 8/2-specific indexing or kernel.

FlashInfer has a separately qualified native 8/2x256 SM89 kernel and its
shape contract remains unchanged. Qwen3.8 24/4x256 is assigned to the existing
ggml CUDA graph path.

## Finite correction

The only product correction is to replace the ggml CUDA 8/2 head constants
with positive integer-ratio GQA validation while preserving all existing
dtype, layout, mask, accumulation, depth, K/V equality, output and overflow
checks. Regression covers both 8/2 and 24/4, followed by the exact Qwen3.8
Direct/Graph consumer. No public API, provider order, upstream source, HYLE
kernel or alternative attention implementation is authorized.
