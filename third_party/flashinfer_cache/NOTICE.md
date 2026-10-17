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
remain unchanged; this does not include the unadopted K-load or overlap experiments.
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
