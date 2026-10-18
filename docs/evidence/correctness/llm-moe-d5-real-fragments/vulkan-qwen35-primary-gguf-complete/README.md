# Qwen3.5 primary GGUF complete Vulkan acceptance

Immutable tests-only candidate `e3b2a5fa3ce8fc67ff22ed881a8c3faa60f219f8`
completes the ordinary Vulkan Direct and Graph cells for
`qwen35_2b:primary_gguf:UD-Q4_K_XL` on the Windows RTX 4070 route.

- R01 reuses the accepted Q6_K embedding/head path and checked-failure
  publication receipt. R03 and R14 reuse the exact Q4_K, Q5_K, Q6_K, Q8_0
  and IQ4_XS projection receipt.
- R02 runs the complete 2048 -> 6144 -> 2048 RMSNorm/SwiGLU FFN with native
  Q4_K gate/up and Q6_K down weights. Direct prepares the mature calls once;
  Graph compiles the full logical chain once and preserves it as one managed
  region. Both replay two changed activations against independent packed-weight
  and elementwise oracles.
- R04 runs BF16 query/key normalization, half-split RoPE and exact 8/2 x 256
  GQA semantics with F32 attention accumulation. R05 runs the K4 causal
  convolution and 16 x 128 Gated DeltaNet state transition. The existing
  semantic tests now run both official and exact primary-GGUF route identities.
- Every new Direct call has a fixed ggml Vulkan managed region or admitted IREE
  dispatch, and the Graph report retains the fixed managed/native-GGUF policy.
  No HYLE shader, fallback, provider, or hot-submit selection was added.
- R13 is inapplicable to this fixed artifact: its 320-tensor header and fixed
  repository tree contain no MTP/NextN tensor or sidecar. The official BF16
  profile retains R13 independently.

The immutable Windows selector passed docs, closure, Release/WERROR configure,
Vulkan Direct (1/1 CTest executable, 55.38 s) and Vulkan Graph (1/1, 14.79 s);
127 unrelated gates were skipped or reused. Before the formal run, the focused
D5 batch passed Direct 9/9 and Graph 8/8, followed by both complete owner
suites. D5-A5 source/static/shared delivery and D5-A6 P1 remain route-batch work
and are not claimed here. Machine-readable details are in
[receipt.json](receipt.json).

The uniform immutable-candidate report remains **57,920 lines across 228
Compute+Graph implementation files**, excluding tests. See
[compute-graph-loc.json](compute-graph-loc.json).
