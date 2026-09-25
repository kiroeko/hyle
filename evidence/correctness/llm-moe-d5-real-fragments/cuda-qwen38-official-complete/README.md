# Qwen3.8-27B official CUDA Direct and Graph acceptance

Immutable candidate `80bbbba06288709ff85c27cc1e4ab1bec8343e84`
completes both CUDA cells for the fixed official BF16 Qwen3.8-27B profile on
the Windows RTX 4070. The ordinary Direct and Graph consumers cover R01-R05
and R13-R15 at H=5120, FFN=17408, GQA 24/4x256, partial interleaved RoPE width
64, Gated DeltaNet K16/V48x128, and MTP input width 10240.

The candidate repairs two adapter omissions in the fixed mature ggml CUDA
carrier. Its existing attention program now admits positive integer-ratio GQA
instead of only 8/2, and public Sigmoid maps to upstream `ggml_sigmoid`.
FlashInfer keeps its independently qualified 8/2x256 domain. No HYLE-owned
math, CUDA kernel, public schema, provider order, or hot routing was added.

Every prepared call and compiled graph is replayed with changed inputs. CUDA
lazy graph metadata may be created during the first submit; allocation counts
remain stable from the first warm replay through the second. Provider reports
contain only mature ggml CUDA, cuBLASLt, SGLang/CCCL, or FlashInfer adapters,
and invalid embedding indices preserve all previously published outputs.

The immutable Release/WERROR selector passed docs, module closure, configure,
ggml CUDA Provider, CUDA Direct, ggml CUDA Graph Provider, and CUDA Graph
Route; 125 unrelated gates were skipped or reused. The build used CUDA
13.3.73 and `CUDAARCHS=89-real`. Machine-readable details are in
[receipt.json](receipt.json). The uniform implementation-only Compute + Graph
count is [57,929 lines in 228 files](compute-graph-loc.json), excluding tests.

External source/static/shared delivery and P1 remain route-batch work and are
not claimed by these two correctness cells.
