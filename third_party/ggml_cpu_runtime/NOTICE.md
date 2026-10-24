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

The official gallocr resource edits also preserve liveness allocation while
returning metadata/backing allocation failures and cleaning partial state.
They match the common resource integration on the older shared core.

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

# CPU implicit causal integration

Ordinary causal attention may carry a private HYCA tag instead of an owned
Q-by-KV half mask. The vector loop limits each query's logical KV end; the
tiled loop fills its existing fixed-size mask scratch with the same zero or
negative-infinity values. Bottom-right positions use the query index within
each head/batch. Native tiled/split selection, QK/softmax/PV arithmetic and
workspace remain unchanged, including the vector path's original Half PV.
The separate boxed-window tag retains its existing reference-path semantics.

# CPU sink denominator validation

The original softmax already includes each head's sink in the maximum. Its
positive-sum assertion now checks the complete denominator after adding the
sink contribution. A fully masked row with a finite sink therefore retains
its valid zero output in assertion-enabled builds. The exponential, sink
addition, reciprocal, normalization and ordinary no-sink arithmetic are
unchanged. The same exact edit is composed onto the mixed CPU core.

# CPU E4 page storage

Private boxed page readers may interpret their original I8 carrier bytes as
E4M3FN using an explicit private tag and independent fixed K/V scales. Only
addressed rows are decoded: K values are scaled in F32 before the F32 dot,
and V values enter the existing bounded F32 tile before the original online
softmax/PV reduction. Each storage multiplication has an F32 boundary. Plain,
Q8 and ordinary untagged attention retain their existing paths. This adds no
resident decoded-cache bank or separate executor; page ownership and Q1 layout
updates retain the original flash operation.
