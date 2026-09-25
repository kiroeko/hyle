# Qwen3.8-27B primary GGUF CUDA Direct and Graph acceptance

Immutable tests-only candidate `02b58c8df5b3e281df8d6aea967c47d4eeea08c0` completes both CUDA cells for the
fixed Qwen3.8-27B `UD-Q4_K_M` profile. The ordinary Direct and Graph consumers
cover R01-R05 and R13-R15 at H=5120, FFN=17408, GQA 24/4x256, Gated DeltaNet
K16/V48x128, and MTP input width 10240.

The storage tests preserve the artifact's Q4_K embedding width and run exact
representative tensor shapes for every stored quantized subtype: Q3_K, IQ3_S,
IQ4_NL, IQ4_XS, Q4_K, Q5_K, Q6_K, and Q8_0. A separate Q6_K
`blk.64.nextn.eh_proj.weight` case covers the artifact's real MTP 10240-to-5120
projection. The semantic chains use changed inputs for replay, preserve output
after invalid embedding indices, and bind only fixed ggml CUDA, cuBLASLt,
SGLang/CCCL, or FlashInfer providers. No HYLE-owned math, fallback, route order,
or product code changed.

The immutable MSVC 19.50 Release/WERROR selector passed docs, module closure,
configure, CUDA Direct, and CUDA Graph with CUDA Toolkit 13.3.73 and
`CUDAARCHS=89-real`. The complete Direct suite passed in 63.09 seconds and the
Graph suite in 64.50 seconds; 127 unrelated gates were skipped or reused.
Machine-readable details are in [receipt.json](receipt.json). The
implementation-only Compute + Graph count remains [57,929 lines in
228 files](compute-graph-loc.json), excluding tests.

External source/static/shared delivery and P1 remain route-batch work and are
not claimed by these two correctness cells.
