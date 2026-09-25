# Qwen3.5 primary GGUF complete CPU acceptance

Immutable tests-only candidate `b2e6a27fb59cc867f74f32da1138c78743a373c0`
completes the ordinary CPU Direct and Graph cells for
`qwen35_2b:primary_gguf:UD-Q4_K_XL`.

- R01 reuses the accepted Q6_K embedding/head path and checked-failure
  publication receipt. R03 and R14 reuse the exact Q4_K, Q5_K, Q6_K, Q8_0
  and IQ4_XS projection receipt.
- R02 now runs a complete 2048 -> 6144 -> 2048 RMSNorm/SwiGLU FFN with native
  Q4_K gate/up and Q6_K down weights. Direct and Graph prepare once, replay two
  changed activations, and perform no metadata allocation or reallocation in
  the measured replay interval.
- R04 runs F32 query/key normalization, half-split RoPE and exact 8/2 x 256
  GQA semantics. R05 runs the K4 causal convolution and 16 x 128 Gated DeltaNet
  state transition. Each intermediate is checked against an independent
  calculation; the already accepted mixed projection receipt supplies the
  quantized projection part of both regions.
- Every prepared operation and Graph compile report names fixed ggml CPU or
  oneDNN providers. The tests add no HYLE math, fallback, provider, or hot-path
  selection.
- R13 is inapplicable to this fixed artifact: its 320-tensor header and fixed
  repository tree contain no MTP/NextN tensor or sidecar. The official BF16
  profile retains R13 independently.

The immutable Windows selector passed docs, closure, the fixture tool,
Release/WERROR configure, CPU Direct, and CPU Graph; 127 unrelated gates were
skipped or reused. D5-A5 source/static/shared delivery and D5-A6 P1 remain route
batch work and are not claimed here. Machine-readable details are in
[receipt.json](receipt.json).

The uniform implementation-only report remains **57,920 lines across 228
Compute+Graph files**; this package changes tests and evidence only. See
[compute-graph-loc.json](compute-graph-loc.json).
