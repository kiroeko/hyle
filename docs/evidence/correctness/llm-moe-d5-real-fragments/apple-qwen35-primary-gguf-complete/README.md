# Qwen3.5 primary GGUF complete Apple acceptance

Immutable tests-only candidate `690d5f2c26e717f237455669a6b08085b8cdbf90`
completes the ordinary Apple Direct and Graph cells for
`qwen35_2b:primary_gguf:UD-Q4_K_XL` on the formal MacBook Air M4 route.

- R01 reuses the accepted Q6_K embedding/head path and checked-failure
  publication receipt. R03 and R14 reuse the exact Q4_K, Q5_K, Q6_K, Q8_0
  and IQ4_XS projection receipt.
- R02 runs the complete 2048 -> 6144 -> 2048 RMSNorm/SwiGLU FFN with native
  Q4_K gate/up and Q6_K down weights. Direct prepares each mature call once;
  Graph compiles the full logical chain once. Both replay two changed
  activations and match independent packed-weight and elementwise oracles.
- R04 runs BF16 query/key normalization, half-split RoPE and exact 8/2 x 256
  GQA semantics with F32 attention accumulation. R05 runs the K4 causal
  convolution and 16 x 128 Gated DeltaNet state transition. The existing
  official semantic tests now also run the exact primary-GGUF model identity,
  so this is a model-qualified route rather than a family-level inference.
- Quantized embedding and projection logs load real ggml Metal Q/K/IQ kernels;
  the remaining regions stay on the admitted MLX 0.32.2 or complete ggml Metal
  paths. No HYLE math, fallback, provider, or hot-submit selection was added.
- R13 is inapplicable to this fixed artifact: its 320-tensor header and fixed
  repository tree contain no MTP/NextN tensor or sidecar. The official BF16
  profile retains R13 independently.

The immutable macOS selector passed docs, closure, Release/WERROR configure,
Apple Direct (1/1 CTest executable, 71.05 s) and Apple Graph (1/1, 10.65 s);
127 unrelated gates were skipped or reused. Before the formal run, all eight
focused Graph D5 tests and both complete owner suites passed with the current
fixture. D5-A5 source/static/shared delivery and D5-A6 P1 remain route-batch
work and are not claimed here. Machine-readable details are in
[receipt.json](receipt.json).

The uniform immutable-candidate report remains **57,920 lines across 228
Compute+Graph implementation files**, excluding tests. See
[compute-graph-loc.json](compute-graph-loc.json).
