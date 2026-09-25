# Qwen3.5 fidelity GGUF complete Apple acceptance

Immutable tests-only candidate `65bdb5f57701767bbcf4115c97900b49cfcfadec`
completes the ordinary Apple Direct and Graph cells for
`qwen35_2b:fidelity_gguf:UD-Q8_K_XL` on the formal MacBook Air M4 route.

- The fixed artifact remains a mixed-format tensor inventory: F16 embedding,
  Q8_0 FFN gate/up/down and attention V/output, F16 Gated DeltaNet output and
  attention K/Q, plus F32 convolution/state/norm tensors.
- R01 uses the admitted MLX F16 gather path with changed-token replay and
  failure-without-publication. R02, R03 and R14 preserve the exact Q8_0 or F16
  weights at the Qwen3.5-2B dimensions; Q8_0 calls execute original ggml Metal
  kernels while dense F16 calls execute MLX Metal.
- R04 and R05 retain the exact 8-query-head/2-KV-head x 256 GQA, K4 causal
  convolution and 16 x 128 Gated DeltaNet semantics. Direct and Graph replay
  changed inputs and compare with independent packed-weight, dense-half or
  fragment oracles.
- Every prepared operation and Graph compile report fixes an admitted MLX or
  ggml Metal implementation. The candidate adds no HYLE math, fallback,
  provider or hot-path selection.
- R13 is inapplicable to the fixed 320-tensor artifact because its audited
  header and repository tree contain no MTP/NextN tensor or sidecar.

The immutable macOS selector passed docs, closure, Release/WERROR configure,
Apple Direct (1/1 CTest executable, 52.35 s; 59/59 tests) and Apple Graph (1/1,
12.47 s; 32/32 tests); 127 unrelated gates were skipped or reused. Before that
formal run, the six focused fidelity/semantic tests in each entry and both full
owner suites passed. D5-A5 source/static/shared delivery and D5-A6 P1 remain
route-batch work and are not claimed here. Machine-readable details are in
[receipt.json](receipt.json).

The uniform immutable-candidate report remains **57,920 lines across 228
Compute+Graph implementation files**, excluding tests. See
[compute-graph-loc.json](compute-graph-loc.json).
