# D5 Qwen3.8-2.4T official BF16 Apple and Vulkan fragment batch

Immutable tests-only candidate `10c878305` reuses the existing Qwen3.8
Direct/Graph consumers for the pinned official
`Qwen/Qwen3.8-2.4T-A95B@207bd685` profile. Shape parameters come from the
shared `ProfileCase`: H=8192, shared FFN N=2048, GQA 64/4 x 256, MTP input
16384, and Gated DeltaNet K16/V128 x 128. No product implementation or
operator math changed. The prior R11 BF16 grouped expert receipt remains
separate four-route evidence.

The new subset exercises embedding, normalization, residual, projection and
head; shared FFN; partial interleaved RoPE/GQA/output gate; MTP; and GDN
state update through ordinary prepared Direct calls and compiled Graph
programs. The first four tests replay changing inputs against independent
oracles. The GDN test repeats a bound input and checks the independent state
oracle; changing-input GDN replay remains part of the later complete R05
assembly.

On Windows RTX 4070, Vulkan Direct and Graph passed the new five-test subsets
5/5 each. Their complete suites passed 167/167 and 78/78. On the MacBook Air
M4, Apple Direct and Graph passed 5/5 each; their complete suites passed
83/83 and 56/56. All four suites had no skips. The Windows selector selected
the two Vulkan factors, and the macOS selector selected Apple and Vulkan
factors; each factor was executed on its applicable machine. Module closure
and documentation checks passed, as did the source diff check.

This is partial fragment evidence. The 512-way R10 router, stable top-10,
dispatch and combine chain, plus the R05 model projection/storage boundary,
are still missing from this profile assembly. No full profile cell,
source/static/shared route-level delivery, or P1 is signed. D5 remains at
48 `RUN_PASS` / 128 `VALIDATION_NOT_RUN`. The canonical LOC script reports
57,938 Compute + Graph implementation lines excluding tests. See
[receipt.json](receipt.json) for the machine-readable record.
