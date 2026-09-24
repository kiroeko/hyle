# Qwen3.5-2B official CPU Direct acceptance

Immutable candidate `3798fd1129323c53c373dbdf7bd628b1a1f6365d`
completes every required region of the official BF16 Qwen3.5-2B CPU Direct
cell through ordinary cold `prepare`, reusable invocation, queue submission,
and completion entry points.

- R01: real embedding rows, checked Gather, normalization and tied LM head.
- R02: real layer-0 norm and gate/up/down weights with BF16 boundaries; fixed
  ggml owns the independent SwiGLU and matrix operations.
- R03: the earlier complete real `[2048,2048]` projection receipt is reused.
- R04: actual Q/K norm weights and exact GQA `8/2 x 256` BF16 decode shape.
- R05: actual K4 convolution, `A_log` and `dt_bias`, plus F32 16x128 state.
- R13/R14: actual MTP `[2048,4096]` weight and BF16/F32 boundary.
- R15: cold workload binding, changed-input replay, no replay metadata
  allocation, checked failure, and no failed-output publication.

The candidate passed the complete ggml CPU Provider, CPU Direct and CPU Graph
selected suites. Delivery and P1 remain route-level D5 work and are not
claimed by this correctness receipt. Machine-readable details are in
[receipt.json](receipt.json).
