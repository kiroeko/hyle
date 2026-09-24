# Qwen3.5 CPU R04 RoPE admission audit

- Windows canonical baseline: `d12388f70f5c1183eacedcd53e28623e75a8d376`.
- Consumer: ordinary CPU Graph compile/replay with exact Qwen3.5-2B official
  `GQA 8/2 x 256`, BF16 Q/K/V ports, F32 normalization and accumulation, and
  workload region R04.
- Reproduction: the graph prepared successfully, then the first replay entered
  fixed ggml `ggml_compute_forward_rope` and aborted at `ops.cpp:6257` because
  the BF16 source reaches a switch that only accepts F16 and F32.
- Source cause: `ggml_cpu_provider.cc` admits F16 or BF16 for RoPE. Its lowering
  forwards the admitted tensor unchanged. The pinned upstream CPU RoPE accepts
  F16 or F32, so the adapter also rejects a valid upstream F32 route.
- Verdict: reproducible Provider admission/lowering defect. This is not a new
  operator or HYLE kernel requirement; D5-F4 is limited to aligning the adapter
  with the fixed upstream conversion and execution domain.
