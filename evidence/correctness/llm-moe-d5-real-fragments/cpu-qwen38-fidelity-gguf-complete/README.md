# Qwen3.8-27B fidelity GGUF CPU Direct and Graph acceptance

Immutable tests-only candidate `5bbecf49865ffbe638d9639ac0fb051ece539e8c`
completes both CPU cells for the fixed Qwen3.8-27B `UD-Q8_K_XL` profile.
The ordinary Direct and Graph consumers cover R01-R05 and R13-R15 at H=5120,
FFN=17408, GQA 24/4x256, Gated DeltaNet K16/V48x128, and MTP input width
10240.

The artifact remains a mixed-format inventory: embedding, dense FFN, GDN
projections and attention output use Q8_0; full-attention Q/K/V, the LM head
and MTP projection remain BF16; norms, convolution and recurrent state use
F32. Exact-width Q8_0 representatives run through fixed ggml CPU regions, and
the semantic chains use fixed ggml CPU or oneDNN providers. Direct and Graph
prepare once, replay changed inputs, compare with independent host oracles,
and preserve the published embedding output after an invalid index. No
HYLE-owned math, fallback, hot selection or product code changed.

The immutable MSVC 19.50 Release/WERROR selector passed docs, module closure,
the model-input tool, configure, CPU Direct, and CPU Graph. The complete
Direct owner suite passed in 66.90 seconds and Graph plus its executor passed
in 67.12 seconds; 127 unrelated gates were skipped or reused. Machine-readable
details are in [receipt.json](receipt.json). The implementation-only Compute +
Graph count remains [57,929 lines in 228 files](compute-graph-loc.json),
excluding tests.

External source/static/shared delivery and P1 remain route-batch work and are
not claimed by these two correctness cells.
