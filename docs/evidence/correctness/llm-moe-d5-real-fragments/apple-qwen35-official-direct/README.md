# Qwen3.5-2B official Apple Direct acceptance

Immutable candidate `e99c2bf929bbee747dbb8a392ab3ff813299be3f`
completes every required region of the official BF16 Qwen3.5-2B Apple Direct
cell through ordinary cold `prepare`, invocation creation, binding, submission,
and checked completion entry points on the formal Apple M4 machine.

- R01 uses real embedding rows, checked Gather, normalization, residual, and tied LM-head weights.
- R02 uses the real layer-0 norm and gate/up/down weights through MLX Metal.
- R03 reuses the complete real `[2048,2048]` projection receipt.
- R04 uses actual Q/K norms and exact GQA `8/2 x 256` BF16 RoPE/attention.
- R05 combines the actual K4 convolution parameters with the admitted ggml Metal Gated DeltaNet state region.
- R13/R14 use the actual MTP `[2048,4096]` weight; R15 covers BF16/F32 boundaries, changed-input replay, retained output, and checked failure.

The focused batch passed 5/5 and the complete Apple Direct owner suite passed
52/52 with WERROR. No HYLE math, Metal kernel, fallback, or route change was
added. Final stable-route delivery and P1 remain D5-A5/A6 work. Machine-readable
details are in [receipt.json](receipt.json).
