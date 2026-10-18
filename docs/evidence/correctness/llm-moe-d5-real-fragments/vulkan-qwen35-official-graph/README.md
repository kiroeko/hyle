# Qwen3.5-2B official Vulkan Graph acceptance

Immutable candidate `1fa67a8facf76feb78f66b5576aee45ac7d16d99`
completes every required region of the official BF16 Qwen3.5-2B Vulkan Graph
cell through ordinary `GraphBuilder`, cold `graph::vulkan::compile`, invocation
creation, binding, submission, and completion entry points.

The graph tests use the same real checkpoint slices and independent numerical
oracles as Direct for embedding/head, FFN, GQA/RoPE/attention, causal
convolution/GDN, and MTP/BF16 boundaries. Each compiled program is replayed
with changed activation, token, or recurrent state, and the invalid Gather path
does not publish an output. R03 reuses the prior exact real projection receipt.

Each graph compilation reports at least one fixed mature managed region. The
immutable WERROR selector passed the complete Direct and Graph Vulkan factors.
Relocated external three-delivery consumers and P1 remain route-level D5 work
and are not claimed by this correctness receipt. Machine-readable details are
in [receipt.json](receipt.json).
