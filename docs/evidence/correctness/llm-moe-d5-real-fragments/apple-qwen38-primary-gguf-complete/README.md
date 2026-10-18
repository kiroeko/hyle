# Qwen3.8-27B primary GGUF Apple Direct and Graph acceptance

Immutable tests-only candidate `dfa1423767d375154aa75cc717d327d7073d331b`
completes both Apple cells for the fixed Qwen3.8-27B `UD-Q4_K_M` profile on
the formal MacBook Air M4. The ordinary Direct and Graph consumers cover
R01-R05 and R13-R15 at H=5120, FFN=17408, GQA 24/4x256, Gated DeltaNet
K16/V48x128, and MTP input width 10240.

The storage tests preserve the artifact's Q4_K embedding width and run exact
representative tensor shapes for every stored quantized subtype: Q3_K, IQ3_S,
IQ4_NL, IQ4_XS, Q4_K, Q5_K, Q6_K, and Q8_0. A separate Q6_K
`blk.64.nextn.eh_proj.weight` case covers the artifact's real MTP
10240-to-5120 projection. The semantic chains replay changed inputs, preserve
the published output after an invalid embedding index, and bind only fixed MLX
Metal or ggml Metal programs. No HYLE-owned math, fallback, route order, or
product code changed.

The Q4_K row oracle also records the mature Provider's numeric boundary.
ggml Metal evaluates the high-nibble `xb->d / 16.h` scale step in half
precision before publishing F32, whereas the independent decoder uses F32.
The observed maximum absolute difference was `4.506111145019531e-4`; only this
Q4_K row conversion is bounded at `5e-4`. Other embedding formats retain the
existing `1e-6` bound.

The immutable AppleClang 21 Release/WERROR selector passed docs, module
closure, configure, Apple Direct, and Apple Graph. The complete Direct suite
passed in 80.02 seconds and Graph in 40.11 seconds; 127 unrelated gates were
skipped or reused. Machine-readable details are in [receipt.json](receipt.json).
The implementation-only Compute + Graph count remains [57,929 lines in 228
files](compute-graph-loc.json), excluding tests.

External source/static/shared delivery and P1 remain route-batch work and are
not claimed by these two correctness cells.
