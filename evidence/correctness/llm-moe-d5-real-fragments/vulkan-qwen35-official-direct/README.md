# Qwen3.5-2B official Vulkan Direct acceptance

Immutable candidate `1fa67a8facf76feb78f66b5576aee45ac7d16d99`
completes every required region of the official BF16 Qwen3.5-2B Vulkan
Direct cell through ordinary cold `prepare`, invocation creation, binding,
submission, and completion entry points.

- R01 uses real embedding rows, checked Gather, normalization, residual, and
  tied LM head weights.
- R02 uses the real layer-0 norm and gate/up/down weights.
- R03 reuses the complete real `[2048,2048]` projection receipt.
- R04 uses actual Q/K norm weights and the exact GQA `8/2 x 256` BF16 decode
  shape for RoPE and attention.
- R05 uses the actual K4 convolution, `A_log`, and `dt_bias` parameters with
  the F32 `16x128` recurrent state.
- R13/R14 use the actual MTP `[2048,4096]` weight; R15 exercises the BF16/F32
  boundary, changed-input replay, retained earlier output, and checked failure.

Every cold prepare contains a fixed ggml Vulkan managed region or qualified
IREE dispatch. The immutable WERROR selector passed the complete Direct and
Graph Vulkan factors. Relocated external three-delivery consumers and P1 remain
route-level D5 work and are not claimed by this correctness receipt.
Machine-readable details are in [receipt.json](receipt.json).
