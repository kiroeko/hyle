# CPU encoded-page resident attention consumer

Finite G04/G05 Q8_0 K/V, BF16 Q/output, GQA8/2, page16, one native thread:
Q1/17 KV1024/768 D256/256 window16; Q64/128 KV1024/768 D256/256 causal unbounded;
holdout Q17/1 KV7/47 D64/128 full noncausal, including Q>KV. These are named
single-machine inference fragments / provisional archetypes, not model frequencies.
Physical pages are reversed and one is shared; unused capacity/tails use sentinels.
Each arm uses identical independently packed Q8 bytes. A double oracle decodes
those actual stored scales/codes, applies the exact valid region/window, and
checks finite BF16 results within8e-3. Both primary/placement preflights observed
worst absolute error0.000244301088 and exact arm agreement; readonly bytes checked.

Both arms use BatchProgram/BatchInvocation, exact retained PagedArenaRead owners,
CPU Queue, same-topology update, two slots, completion wait and output lease
return/release. Product reads physical pages. Peer cold gathers referenced valid
Q8 rows into owned contiguous K/V, reports extra storage/copy1949696bytes for each
D256 case and22032bytes for the holdout, and retains original page owners. No warm
history copy, dirty encoding, cache growth or changing-layout planning claim.

Peer math/plan/pool are from the independently built pristine ggml
ffa4e8b80930029a35991f94e7c8a93cd67730ab DLL, AVX2/MSVC19.50/one thread.
The existing authenticated POD ABI checks tensor336/plan56/src offset/BF16 type
and actual compute function DLL image. Graph constructors are matching original
ABI; no candidate compile/lower or attention arithmetic is called by the peer.
Original generic Q8 already uses original SIMD dot and rowwise V decode, so no
artificial peer downgrade. K/V strides and output use independent Dqk/Dv.

Q->F32->Q8-dot, attention and output BF16 conversion are warm on both arms.
Candidate additionally validates finite/range Q; cost remains charged. Inputs
are guaranteed finite/in-range on both, with no per-operation cancel/concurrent
waiter requirement. Peer cold F16 causal/window mask has O(Q*KV) storage and a
warm skip scan; candidate computes window bounds directly. These are explicit
storage/algorithm differences within the same immutable attention operation.
Neither side defers warm computation or owner release beyond the measured call.
Native pool maxima remain UNKNOWN; cold graph/mask/pool/cache setup/teardown
are separately reported and are outside this explicitly resident-warm scope.

Installed static SDK is the independently tested f772 product, with every linked
library/program SHA authenticated. Existing unified instrument, machine policy,
16-round nanobench compare, workload-matched A/A, both execution/allocation orders,
placement replicas, original n15/31 schedule and5% margin remain fixed.
No P1 started at provenance creation. This contract cannot sign producer/growth,
all G04/G05, native private budgets, capture or end-to-end token/s.
