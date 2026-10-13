# CPU paged immutable-history performance consumer

Finite G01 BF16 GQA8/2 page_size16: Q1/17 and Q64/128 KV1024/768 D256 causal,
plus Q1/17 KV31/47 D64 noncausal. Physical pages are reversed, one page shared
across requests, unused pages/tails contain sentinels. Independent double oracle
uses only logical valid positions; original8e-3 bound unchanged, preflight actual
maximum error below6.11e-5. Ready pages are checked readonly after correctness.

Both arms use public BatchProgram/BatchInvocation/PreparedBatch, original CPU
Queue, exact page guards, metadata update, Completion and output lease return,
two slots and one native thread. Histories remain immutable throughout each
experiment. Product reads its physical page table. Peer retains actual original
PagedArenaRead owners and cold copies only referenced valid logical rows into
separately owned contiguous K/V. Other backing/topology is rejected; query remains
a current input. Cold copy and extra cache storage are reported:3,670,016bytes
for D256 cases,39,936bytes for D64, separately from arena/mask/query/workspace
and UNKNOWN native private pool. This is a prepared contiguous-cache baseline
for warm readonly attention, not a producer/growth/reorder-production certificate;
G03/G06 and their original acceptance scope remain required.

Original CPU math/plan/pool compile from clean authenticated original ggml
ffa4e8b80930029a35991f94e7c8a93cd67730ab, with no HYLE patches. A private DLL
bridge exposes existing CpuProgram POD with tensor/cplan sizes, src offset and
BF16 enum handshake. Independent graph metadata uses original same-revision
ggml constructors; CPU DLL owns its pool and borrows caller workspace. No
attention arithmetic or candidate compile/lower is used. Explicit AVX2/MSVC19.50
and one normal-priority native thread match candidate's dispatched program.
Windows paths are made native before DLL_LOAD_DIR; original dependent DLLs
stage together. Initial loader failures stay preflight, not product results.

The static candidate SDK already tested at provider1e9/runtime3cf is linked,
avoiding a product rebuild. Its libraries/header/CPU program identities are
pinned. A/A uses16 processes across both implementations, execution/allocation
orders and code placement via original machine/environment/statistics helpers.
Original controls,n15/31 stop,5% margin remain unchanged. Source/compile/DLL
hashes bind the actual original math before sampling. No token/s, capture,
private CPU budget, GPU or G02 claim follows from this finite comparison.
