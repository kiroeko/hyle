# FlashInfer mixed FP8 cache tile patch

Upstream: flashinfer-ai/flashinfer, revision 69ff11fc4954396d98326656dc85debd2223f637 (Apache-2.0).
HYLE applies the exact edits in patches.json only to a build-local verified copy.
The shared K row for FP8 Dqk64/Dv128 uses 128 bytes with a zero upper half; public/global
cache rows and the QK reduction remain 64 elements. Both paged/ragged producers, repack
source stride and dispatcher shared-memory accounting follow that physical tile row.
No complete cache history is padded or dequantized. Other native domains are unchanged.
The immutable upstream source cache is never modified. Original copyright/license apply.

The private encoded-cache adapter also permits a final merge callback and an optional
scale epilogue in the original persistent merge. By default, FP32 partials and accumulation remain
unchanged; scaling precedes the sole final cast, including a single partial. Default
upstream routes retain their original output semantics. This merge epilogue itself adds no public HYLE API.

The 16-bit output writer synchronizes the CTA before reusing its aliased shared
storage, and the output warp before reading its rearranged stores. FP32 direct
stores and the attention/merge mathematics are unchanged.

Every native route in this provider uses the same verified private headers, so
ordinary and encoded attention share these synchronization fixes and merge ABI.

The admitted FP8 prefill route retains the pinned original cooperative
FP8-to-Q-type shared tile conversion for CTA_Q>16 and the original finite
VO domain. Storage, KernelTraits and all three dispatcher budgets agree.
Global cache storage stays compressed; only the current KV tile expands
in shared memory. The corrected packed-row and native16 work bounds,
FP32 partials, final scaling, output synchronization and physical short-K
padding are retained. CTA16 keeps the original register path.

The unused FP32 output shared-memory member retains only one element; the
original FP32 writer already stores fragments directly to global memory.
All live Q/K/V and cross-warp state storage and 16-bit output capacity remain.

FP8 tile validity follows the original producer row coverage:128-byte
swizzled rows require MMA_KV*4 divisible by Q warps;64-byte rows and FP4
retain their original MMA_KV*2 constraint. Dispatch additionally bounds
FP8 per-warp work by the original native16-bit footprint budget because
QK/PV use widened math fragments. Actual compressed shared-memory capacity,
the original register bound, planner and arithmetic remain. The qualified
tile synchronization adjustment is described below.

The opt-in scaled FP32 persistent merge retains all local state writes and
CTA/async synchronization. Only its final output group performs the same
ordered cross-group merge, final normalization, scale and sole output cast.
Zero/single-partial final accesses share that owner; optional LSE has one
scalar writer. The default upstream merge specialization stays unchanged.

Full128-byte FP8 conversion rows in the non-RoPE, non-split, separate-K/V
path follow the exact original producer thread ownership. Existing local
async waits precede conversion; the following full CTA barrier publishes
Q and converted tile data. The adjacent pre-conversion barrier is retained
for every other path. Consumer-retirement, final and output barriers remain;
short-K padding, conversion values, destination layout and FP32 merge remain.

The finite E4M3 F32-partial batch path with equal128/256 head dimensions
loads packed global vectors through the original decoder directly into
separate native16 K/V shared tiles. It replaces compressed shared staging
and the common repack tile without expanding the resident cache. Explicit
wait0 drains Q loads before the retained CTA publication barriers; consumer
retirement, final and output barriers remain. Other paths and single-prefill
retain their original producers, layouts and arithmetic. This structural
change alone does not assert a qualified throughput improvement.

An explicit private BF16-partial attention marker also admits the existing
direct FP8 tile producer for BF16 output at equal D256 and CTA64. All other
direct-stage gates, the packed decoder, shared layout and synchronization
remain unchanged. The scoped SM89 K-pair prefetch described below changes
only the timing of the two raw K loads.
The caller opts into intermediate BF16 partial values while attention and
merge accumulation remain FP32. Final scaling still precedes the final output
cast. Intermediate rounding can differ from the default FP32-partial path.

The private prefill planner accepts a default-false BF16-partial flag after
kv_dtype_bytes. Query and materialization share one implementation: only an
enabled flag with the actual split CTA64 plan uses two bytes per temporary
value; other plans retain four. Temporary scores remain FP32. The native
adapter must admit the finite domain before passing this flag and select
the same predicate for execution. This changes required workspace bytes;
actual reserved memory still follows the caller's workspace cap.

On SM89 only, the private BF16-partial marker enables paired raw K prefetch
for paged causal direct-stage BF16 Q/output at equal D256, CTA64, MMA_KV1
and Q/KV warps4/1. The next tile's two packed16-byte K vectors are loaded
before online softmax; the original row-validity guard applies to both.
The same decoder publishes both vectors in original order and shared layout
at the original producer site. Initial K/V loads, V scheduling, consumer
retirement, all barriers/async waits and other device architectures retain
their original paths. This scheduling change adds no public API or numeric
format and does not expand the resident cache.
Within that same private scope, K and V offsets share one guarded page-index
load while preserving each tensor's own base and strides. Invalid rows keep
both offsets zero; every other producer path retains its original indexing.

Within the same SM89 private BF16-partial scope, the C++ source places the
unchanged paired raw K prefetch after the QK call and before logits transform
and masking, ahead of online softmax. The compiler can schedule these raw
loads across the QK tail. Decoder/publication timing, shared K/V page indexing,
row guards, arithmetic, barriers and all other routes remain.

Windows CUDA uses the same attention and cache kernels. The verified header
set additionally selects CUDA forceinline for MSVC, spells two 16-bit
temporaries as unsigned short, and makes seven scheduler/decode max expressions
independent of LP64 versus LLP64 integer aliases. Two restrict annotations
are removed from an unused host-only rotary launch wrapper because nvcc
emits them as unsupported C99 restrict in MSVC C++. Device kernels,
thread/block dispatch, storage layouts and numerical contracts are unchanged.
The decode header also uses CUDA __shared__ instead of a GNU shared
attribute for the same dynamic shared-memory declaration.

The compact MLA adapter retains the upstream CKV512/KPE64 kernel and scheduler.
The scheduler can count its original workspace layout without writing or
uploading it, and checks its fixed work-array and integer chunk capacities
before materialization. Three dependent template calls use explicit empty
argument lists for MSVC. Its causal kernel uses a ceiling query-row bound,
matching the scheduler when a packed cluster crosses a partial query head
group (including 96 heads); the original elementwise causal mask remains.
The typed MLA adapter stores normalized split outputs directly from the F32
accumulators into F32 partial storage, and the original merge reads those F32
values. Final output conversion and the original probability tile are retained;
the scheduler accounts for four-byte partial elements. No larger shared tile is
introduced.
The zero-KPE specialization reserves one unused query-storage column instead
of declaring a zero-length array; its compile-time KPE load/compute loops stay empty.

MLA workspace allocation follows the scheduler's active work and split rows.
Each per-work integer array uses total_num_works entries, and work_indptr uses
num_clusters + 1; the original 16384-work admission bound is unchanged. All
five merge arrays retain num_sm entries, including blocks without merge work.
Partial outputs use partial_o_nnz * head_dim_o * the selected partial width,
and partial LSE uses partial_o_nnz float entries. A no-split schedule therefore
needs no partial storage. Query and materialization use these same counts;
kernel indexing, split/merge metadata and arithmetic are unchanged.
These are required workspace sizes. The public batch adapter allocates the
queried layout requirement within its explicit positive workspace cap. For
no-split schedules, the scheduler skips both partial allocations, records zero
offsets and reports zero float bytes; the private native wrapper supplies null
partial pointers without arithmetic on an absent buffer. Split schedules keep
the original aligned allocations, F32 partials and merge. The common allocator
and attention arithmetic are unchanged.

Typed ragged MLA reuses the same CKV512/KPE64 scheduler, tiles, F32 split
partials and merge. Its private linear-row mode reads the original packed
cache at the request's global KV indptr plus local row, with the physical
512/576-element stride and the CKV prefix shared by V. No page table or
expanded K/V payload is created. A null page table remains invalid unless
the adapter explicitly selects plain 16-bit or BF16-query scalar E4 unit-row
storage; paged lookup and the original scheduler remain unchanged.

Uniform E4 compact MLA admits BF16 query/output and one immutable F32 cache
scale for the original CKV512 and optional KPE64 row. CKV is also V; it is not
an independently scaled value allocation. Paged and explicit unit-row ragged
adapters use element pitches of the truthful E4 row. Each E4 code is multiplied
by its original F32 scale with F32 round-to-nearest and then rounded once to
BF16 in the existing bounded shared tile. The scale is not first rounded to
BF16. The scalar repack uses no-FTZ PTX multiplication; plain F16/BF16 arithmetic
is unchanged. The group-scale array remains a distinct native path.

On the one-stage SM89 tier only E4 uses KV16 with the original Q64/256-thread
geometry. Its nonsharded shared-storage specialization omits the unused 512-byte
cross-warpgroup m/d state: CKV512+KPE64 needs 101376 bytes and CKV512 needs
91296 bytes. Plain storage, the MLA scheduler, F32 split partial/merge, existing
synchronization and output ownership remain unchanged. This is not the separate
scaled DSA format with inline per-group scales or a BF16 positional tail.
