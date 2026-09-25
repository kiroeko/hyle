# Qwen3.8-27B official Apple Direct and Graph acceptance

Immutable candidate `6086dd721af664a1d759bb4eea3dfd645e02c99f` completes
both Apple cells for the fixed official BF16 Qwen3.8-27B profile on the formal
MacBook Air M4. The ordinary Direct and Graph consumers cover R01-R05 and
R13-R15 at H=5120, FFN=17408, GQA 24/4x256, partial interleaved RoPE width 64,
Gated DeltaNet K16/V48x128, and MTP input width 10240.

The existing MLX Metal and admitted ggml Metal programs accepted the exact
profile without a product change. Each prepared call or compiled one-node graph
is replayed with changed activation, token, or gate data. Graph compilation
reports a fixed nonempty mature implementation policy, and an invalid embedding
index leaves the previously published result intact. The previously accepted
whole ggml Metal R05 region supplies the recurrent fragment; the tests do not
split it into HYLE kernels.

The immutable Release/WERROR selector passed docs, module closure, configure,
Apple Direct, and Apple Graph; 127 unrelated gates were skipped. The complete
Direct factor passed in 54.37 seconds and the complete Graph factor in 13.05
seconds on AppleClang 21 with MLX 0.32.2 and pinned ggml Metal 0.24.0.
Machine-readable details are in [receipt.json](receipt.json). The uniform
implementation-only Compute + Graph count is
[57,929 lines in 228 files](compute-graph-loc.json), excluding tests.

External source/static/shared delivery and P1 remain route-batch work and are
not claimed by these two correctness cells.
