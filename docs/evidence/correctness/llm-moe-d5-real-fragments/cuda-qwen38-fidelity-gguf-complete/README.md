# Qwen3.8-27B fidelity GGUF CUDA Direct and Graph acceptance

Immutable tests-only candidate `e1ecacc3bd7c27c461b8cc46665180aeeacfd584`
completes both CUDA cells for the fixed Qwen3.8-27B `UD-Q8_K_XL` profile. The
ordinary Direct and Graph consumers cover R01-R05 and R13-R15 at H=5120,
FFN=17408, GQA 24/4x256, Gated DeltaNet K16/V48x128, and MTP input width
10240.

The storage tests preserve the artifact's mixed tensor roles: Q8_0 embedding,
dense FFN, fused GDN QKV, attention output, SSM alpha and SSM output; BF16
full-attention Q/K/V, LM head and MTP projection; and F32 norm, convolution and
recurrent state. Each exact-width Q8_0 role runs through fixed ggml CUDA. The
semantic chains bind only fixed ggml CUDA, cuBLASLt, SGLang/CCCL or FlashInfer
providers, replay with changed inputs, and preserve the published output after
an invalid embedding index. No HYLE-owned math, fallback, route order or
product code changed.

The immutable MSVC 19.50 Release/WERROR selector passed docs, module closure,
configure, CUDA Direct and CUDA Graph with CUDA Toolkit 13.3.73 and
`CUDAARCHS=89-real`. The complete Direct suite passed in 66.53 seconds and the
Graph suite in 68.66 seconds; 127 unrelated gates were skipped or reused.
Machine-readable details are in [receipt.json](receipt.json). The
implementation-only Compute + Graph count remains [57,929 lines in 228
files](compute-graph-loc.json), excluding tests.

External source/static/shared delivery and P1 remain route-batch work and are
not claimed by these two correctness cells.
