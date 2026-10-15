# FlashInfer mixed FP8 cache tile patch

Upstream: flashinfer-ai/flashinfer, revision 69ff11fc4954396d98326656dc85debd2223f637 (Apache-2.0).
HYLE applies the exact edits in patches.json only to a build-local verified copy.
The shared K row for FP8 Dqk64/Dv128 uses 128 bytes with a zero upper half; public/global
cache rows and the QK reduction remain 64 elements. Both paged/ragged producers, repack
source stride and dispatcher shared-memory accounting follow that physical tile row.
No complete cache history is padded or dequantized. Other native domains are unchanged.
The immutable upstream source cache is never modified. Original copyright/license apply.

The private encoded-cache adapter also permits a final merge callback and an optional
scale epilogue in the original persistent merge. FP32 partials and accumulation remain
unchanged; scaling precedes the sole final cast, including a single partial. Default
upstream routes retain their original output semantics. No public HYLE API is added.

The 16-bit output writer synchronizes the CTA before reusing its aliased shared
storage, and the output warp before reading its rearranged stores. FP32 direct
stores and the attention/merge mathematics are unchanged.

Every native route in this provider uses the same verified private headers, so
ordinary and encoded attention share these synchronization fixes and merge ABI.

The admitted FP8 cache route uses the original register-fragment conversion
in compute_qk and compute_sfm_v rather than a complete BF16 shared-memory
repack tile. Storage, KernelTraits and all three dispatcher budgets agree.
The original planner and occupancy selection remain; FP32 partials, final
scaling, output synchronization and physical short-K padding are retained.
