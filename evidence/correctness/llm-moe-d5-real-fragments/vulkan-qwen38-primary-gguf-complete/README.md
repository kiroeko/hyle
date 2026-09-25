# Qwen3.8-27B primary GGUF Vulkan Direct and Graph acceptance

Immutable tests-only candidate `af6a99c450996250a3f557f1eda832ca91c7ef41`
completes both Vulkan cells for the fixed Qwen3.8-27B `UD-Q4_K_M` profile on
the formal Windows RTX 4070 machine. The ordinary Direct and Graph consumers
cover R01-R05 and R13-R15 at H=5120, FFN=17408, GQA 24/4x256, Gated DeltaNet
K16/V48x128, and MTP input width 10240.

The storage tests preserve the artifact's Q4_K embedding width and run exact
representative tensor shapes for every stored quantized subtype: Q3_K, IQ3_S,
IQ4_NL, IQ4_XS, Q4_K, Q5_K, Q6_K, and Q8_0. A separate Q6_K
`blk.64.nextn.eh_proj.weight` case covers the artifact's real MTP
10240-to-5120 projection. The semantic chains replay changed inputs, preserve
the published output after an invalid embedding index, and bind only fixed
ggml Vulkan managed regions or admitted IREE Vulkan programs. No HYLE-owned
math, fallback, route order, or product code changed.

The immutable MSVC 19.50 Release/WERROR selector passed docs, module closure,
configure, Vulkan Direct, and Vulkan Graph. The complete Direct suite passed
in 86.70 seconds and Graph in 70.37 seconds; 127 unrelated gates were skipped
or reused. Machine-readable details are in [receipt.json](receipt.json). The
implementation-only Compute + Graph count remains [57,929 lines in 228
files](compute-graph-loc.json), excluding tests.

External source/static/shared delivery and P1 remain route-batch work and are
not claimed by these two correctness cells.
