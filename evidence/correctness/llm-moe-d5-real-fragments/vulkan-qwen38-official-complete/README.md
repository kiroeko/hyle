# Qwen3.8-27B official Vulkan Direct and Graph acceptance

Immutable candidate `5fc3c2f418795e6cd89b86e3f546d5775b96758a` completes
both Vulkan cells for the fixed official BF16 Qwen3.8-27B profile on the
Windows RTX 4070. The ordinary Direct and Graph consumers cover R01-R05 and
R13-R15 at H=5120, FFN=17408, GQA 24/4x256, partial interleaved RoPE width 64,
Gated DeltaNet K16/V48x128, and MTP input width 10240.

The existing fixed ggml Vulkan managed regions and limited admitted IREE
dispatches accepted the exact profile without a product change. Each prepared
call or compiled one-node graph is replayed with changed activation, token, or
gate data, and an invalid embedding index leaves the previously published
result intact. The previously accepted whole ggml Vulkan R05 region supplies
the recurrent fragment; the tests do not split it into HYLE kernels.

The immutable Release/WERROR selector passed docs, module closure, configure,
Vulkan Direct, and Vulkan Graph; 127 unrelated gates were skipped. The complete
Direct factor passed in 31.98 seconds and the complete Graph factor in 16.73
seconds with Vulkan SDK 1.4.357.0. Machine-readable details are in
[receipt.json](receipt.json). The uniform implementation-only Compute + Graph
count is [57,929 lines in 228 files](compute-graph-loc.json), excluding tests.

External source/static/shared delivery and P1 remain route-batch work and are
not claimed by these two correctness cells.
