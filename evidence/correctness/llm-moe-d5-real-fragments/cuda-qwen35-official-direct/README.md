# Qwen3.5-2B official CUDA Direct acceptance

Immutable candidate `9917438bb44ebc29f647f2b0ce2919a73342fa3e`
completes every required region of the official BF16 Qwen3.5-2B CUDA Direct
cell through ordinary cold `prepare`, reusable invocation, queue submission,
and completion entry points.

- R01: real embedding rows, checked Gather, normalization and tied LM head.
- R02: real layer-0 norm and gate/up/down weights; fixed ggml CUDA owns BF16
  SwiGLU through its F32 Cast boundary, while mature MatMul providers retain
  their qualified domains.
- R03: the earlier complete real `[2048,2048]` projection receipt is reused.
- R04: actual Q/K norm weights and exact GQA `8/2 x 256` BF16 decode shape;
  fixed ggml CUDA owns RoPE through its F32 Cast boundary.
- R05: actual K4 convolution, `A_log` and `dt_bias`, plus F32 16x128 state.
- R13/R14: actual MTP `[2048,4096]` weight and BF16/F32 boundary.
- R15: cold workload binding, changed-input replay, stable warmed metadata,
  checked failure, and no failed-output publication.

The immutable SM89/WERROR selector passed the complete ggml CUDA Provider,
CUDA Direct, ggml CUDA Graph Provider and CUDA Graph selected gates. Static
and shared owner suites also passed, with DLLs beside executables. Relocated
external three-delivery consumers and P1 remain route-level D5 work and are
not claimed by this correctness receipt. Machine-readable details are in
[receipt.json](receipt.json).
