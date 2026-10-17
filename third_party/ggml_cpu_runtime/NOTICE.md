# ggml CPU resource integration and original IQ1 inference extension

CPU-only builds use ggml `ffa4e8b80930029a35991f94e7c8a93cd67730ab` (MIT).
The three narrow IQ1 types originate from Unsloth llama.cpp
`46cbf0e95786fe8f5b7c0e86d57aaf8f8eceea7f` (MIT), branch
`iq1-narrow-grids-b10775`.

`patches.json` binds every upstream file and resulting file by SHA-256. It
preserves HYLE's existing fallible arena/threadpool and build identity edits,
then borrows the original IQ1_XS/XXS/XXXS blocks, codebook grids, dequantizers,
generic CPU dot products, traits and row validation. Quantizers and GPU
implementations are outside this port. Mathematical bodies and tables are
unchanged; additional CPU switch cases and fallback declarations connect them
to the newer core. `borrowed-fragments.json` records the source paths.

The immutable dependency cache remains unmodified. The runtime owner creates
an exact patched copy inside its build directory. Baseline and AVX2 programs
use that same core and patch. Builds containing a GPU route retain the old
ggml source and resource patches; this port does not qualify those routes.

Original sources:

- https://github.com/ggml-org/ggml/tree/ffa4e8b80930029a35991f94e7c8a93cd67730ab
- https://github.com/unslothai/llama.cpp/tree/46cbf0e95786fe8f5b7c0e86d57aaf8f8eceea7f/ggml
# CPU paged-address integration

Private tagged batch graphs also borrow the original flash attention math with
an owned I32 physical page table. Only effective K/V row counts and four native
K/V read addresses change; vector dot products, tiled GEMM, online softmax and
split reduction remain upstream. Ordinary untagged attention has no page table.
The same address contexts are authenticated on the mixed 456172 core so the
existing CPU IQ1 manifest composition deduplicates these shared edits.
The generic per-token loop resolves its physical row once and applies the
independent K/V byte strides to that same row; attention arithmetic and table
ownership are unchanged.

# CPU unary Hadamard integration

The unary Hadamard bridge reuses each pinned core's existing FWHT body.
Only its input parameter and metadata locals are made explicit; normalization,
F32 accumulation, SIMD butterflies and row partitioning remain native. A unary
CUSTOM entry takes one real input, without a synthetic matrix or payload. Its
callback belongs to the selected baseline/AVX2 CPU program. The legacy matrix
hint and its `use_ref` matrix fallback remain intact; unary CUSTOM execution
uses the same FWHT body in either reference setting.
