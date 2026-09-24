# Qwen3.5 primary GGUF Vulkan projection partial acceptance

Immutable tests-only candidate `e31a6e7d35c9b72c58bfd839adb953857d4287ff`
completes the R03 mixed-projection and R14 quantized-MatMul fragments for
`qwen35_2b:primary_gguf:UD-Q4_K_XL` through ordinary Vulkan Direct and Graph
entry points on the Windows RTX 4070.

- One parameterized consumer preserves the audited full reduction/output
  widths and native Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS storage for representative
  FFN, full-attention and Gated DeltaNet tensors. Only batch/token is reduced
  to one row.
- Direct and Graph both cold-bind fixed ggml Vulkan. Every prepared artifact is
  replayed with two changed device-resident activation rows and matches an
  independent stored-weight decode plus F32 dot oracle.
- The oracle uses maximum absolute error `8e-3` and RMSE `2e-3` to separate
  ggml's native quantized activation-row dot path from the already-applied
  weight quantization. All five formats passed the same fixed gate.
- Direct observes one mature ggml Vulkan region for every projection; Graph
  compiles the single MatMul through its mature managed-region path. No HYLE
  Vulkan shader, fallback or hot-path selection was added.
- The immutable Vulkan SDK 1.4.357.0, driver 610.62, MSVC 19.50 WERROR selector
  passed docs, closure, configure, Vulkan Direct and Vulkan Graph, with 127
  gates skipped or reused overall.

This receipt claims R03 and R14 only. Existing R01 receipts remain independent;
R02, R04, R05, R13, R15, complete profile cells, three deliveries and P1 remain
D5 work. Machine-readable details are in [receipt.json](receipt.json), and the
uniform implementation-only LOC report is in
[compute-graph-loc.json](compute-graph-loc.json).
