# Qwen3.8-27B official CPU Direct and Graph acceptance

Immutable candidate `667dda0d6a696eab0d3f243386a73f2f54cefe55`
completes both CPU cells for the fixed official BF16 Qwen3.8-27B profile.
The ordinary Direct and Graph consumers cover R01-R05 and R13-R15 at the
official inner widths: H=5120, FFN=17408, GQA 24/4x256, partial interleaved
RoPE width 64, Gated DeltaNet K16/V48x128, and MTP input width 10240.

The candidate repairs two admission omissions in the fixed mature ggml CPU
carrier. Its existing SDPA program now admits positive integer-ratio GQA
instead of only the prior 8/2 shape, and public Sigmoid maps to upstream
`ggml_sigmoid`. No HYLE-owned math, kernel, public schema, or route order was
added or changed.

Each prepared call or graph is replayed with changed inputs. Provider reports
contain only fixed ggml CPU, oneDNN primitive, or oneDNN graph regions; the
invalid embedding path preserves all published outputs. The immutable
Release/WERROR selector passed docs, module closure, the model-input tool,
configure, ggml Provider, CPU Direct, and CPU Graph, while 126 unrelated gates
were skipped or reused. Machine-readable details are in
[receipt.json](receipt.json), and the uniform implementation-only Compute +
Graph count is in [compute-graph-loc.json](compute-graph-loc.json).

External source/static/shared delivery and P1 remain route-batch work and are
not claimed by these two correctness cells.
