# Qwen3.5 primary GGUF Apple projection partial acceptance

Immutable tests-only candidate `ef9c0562184507030de850a015dd9498b96d34d7`
completes the R03 mixed-projection and R14 quantized-MatMul fragments for
`qwen35_2b:primary_gguf:UD-Q4_K_XL` through ordinary Apple Direct and Graph
entry points on the formal MacBook Air M4.

- One parameterized consumer preserves the audited full reduction/output
  widths and native Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS storage for representative
  FFN, full-attention and Gated DeltaNet tensors. Only batch/token is reduced
  to one row.
- Direct and Graph cold-bind fixed ggml Metal. Every prepared artifact is
  replayed with two changed shared-buffer activation rows and matches an
  independent stored-weight decode plus F32 dot oracle.
- The oracle uses maximum absolute error `8e-3` and RMSE `2e-3` to separate the
  mature quantized activation-row dot path from the already-applied weight
  quantization. All five formats passed the same fixed gate.
- Runtime logs identify the native Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0 Metal
  `mul_mv` pipelines; Graph also reports the fixed `ggml Metal` implementation
  policy. No HYLE Metal kernel, fallback or hot-path selection was added.
- The immutable AppleClang 21, macOS SDK 27.0, Release/WERROR selector passed
  docs, closure, configure, Apple Direct and Apple Graph on MacBook Air
  Mac16,12, with 127 gates skipped or reused overall.

The first full owner run used an obsolete partial R03 fixture and four existing
D5 tests rejected missing files. Regenerating the current 12-tensor fixture from
the canonical local Qwen3.5-2B checkpoint and synchronizing it to the Air made
both complete owner suites pass 2/2; the failure was fixture provenance, not a
product defect.

This receipt claims R03 and R14 only. Existing R01 receipts remain independent;
R02, R04, R05, R13, R15, complete profile cells, three deliveries and P1 remain
D5 work. Machine-readable details are in [receipt.json](receipt.json), and the
uniform implementation-only LOC report is in
[compute-graph-loc.json](compute-graph-loc.json).
