# Qwen3.8-27B fidelity GGUF Apple Direct and Graph acceptance

Immutable tests-only candidate `1544c98061f36a8a7e5ea67b5db00a5f47bd8872`
completes both Apple cells for the fixed Qwen3.8-27B `UD-Q8_K_XL` profile. The
ordinary Direct and Graph consumers cover R01-R05 and R13-R15 at H=5120,
FFN=17408, GQA 24/4x256, Gated DeltaNet K16/V48x128, and MTP input width
10240.

The storage tests preserve the artifact's mixed tensor roles: Q8_0 embedding,
dense FFN, fused GDN QKV, attention output, SSM alpha and SSM output; BF16
full-attention Q/K/V, LM head and MTP projection; and F32 norm, convolution and
recurrent state. The Q8_0 roles bind fixed ggml Metal while the semantic chains
use fixed ggml Metal or MLX Metal programs. Every prepared artifact replays
with changed inputs, and an invalid embedding index preserves the published
output. No HYLE-owned math, fallback, route order or product code changed.

The immutable AppleClang 21 Release/WERROR selector passed docs, module
closure, configure, Apple Direct and Apple Graph on the MacBook Air M4 with
MLX 0.32.2, ggml Metal 0.24.0 and macOS SDK 27.0. The complete Direct suite
passed in 39.91 seconds and the Graph suite in 37.79 seconds; 127 unrelated
gates were skipped or reused. Machine-readable details are in
[receipt.json](receipt.json). The implementation-only Compute + Graph count
remains [57,929 lines in 228 files](compute-graph-loc.json), excluding tests.

External source/static/shared delivery and P1 remain route-batch work and are
not claimed by these two correctness cells.
