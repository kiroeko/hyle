# Qwen3.5 primary GGUF CPU projection partial acceptance

Immutable tests-only candidate `6834a6c940b5501775e6fbbceeb9c2f0596d02c0`
completes the R03 mixed-projection and R14 quantized-MatMul fragments for
`qwen35_2b:primary_gguf:UD-Q4_K_XL` through ordinary CPU Direct and Graph
entry points.

- The tensor names, full reduction/output widths, storage formats and roles
  come from `unsloth/Qwen3.5-2B-GGUF@f6d5376`: Q4_K FFN gate, Q6_K FFN
  down, Q5_K attention K, IQ4_XS FFN gate, and Q8_0 Gated DeltaNet output.
  Only the batch/token extent is reduced to one row.
- Format-valid deterministic packed weights are decoded independently for the
  oracle. Both entries cold-bind fixed ggml CPU, preserve the exact workload
  identity, and replay the same prepared artifact with two changed activation
  rows.
- The oracle evaluates the decoded stored weights in F32. The gate separately
  allows the bounded error from ggml's native Q8 activation-row dot path:
  maximum absolute error `8e-3` and RMSE `2e-3`.
- Graph compiles every projection to one mature ggml region and calls
  `prepare_replay` once. No HYLE operator, fallback, kernel or hot-path
  selection was added.
- The immutable Windows MSVC 19.50 WERROR selector passed docs, closure, the
  fixture tool, configure, CPU Direct and CPU Graph; 127 unrelated gates were
  skipped or reused.

This receipt claims R03 and R14 only. Existing R01 receipts remain independent;
R02, R04, R05, R13, R15, complete profile cells, three deliveries and P1 remain
D5 work. Machine-readable details are in [receipt.json](receipt.json), and the
uniform implementation-only LOC report is in
[compute-graph-loc.json](compute-graph-loc.json).
