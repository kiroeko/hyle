# Finite CUDA encoded immutable-history comparison

The consumer is a two-request GQA8/2 page16 attention episode, using BF16 Q/output
and E4M3FN K/V with independent .5/2 scales. The three geometries are Q1/17,
KV1024/768, D256/256, causal window16; Q64/128 with the same history and unbounded
causal mask; and Q17/1, KV7/47, D64/128, full noncausal. Reordered/shared physical
pages and poison tails are preserved. The fixture requires a material difference
from uniform averaging, so missing query-key products cannot pass its oracle.

The candidate uses the actual installed HYLE FlashInfer encoded paged route.
The peer retains the same original FP8 exact-frame owners and constructs an
additional readonly BF16 physical-page snapshot cold, by GPU decoding only the
union of authorized effective rows. The fixed powers-of-two scales are absorbed
once into the exact BF16 representation. Both original compressed and expanded
representations stay resident. Two slots share one immutable expanded snapshot;
their mutable planning/output workspaces remain independent.

The peer then calls FlashInfer69ff BF16 paged prefill and original BF16
partial/split merge, with the original planner, mask and independent K/V strides.
The namespace token prevents kernel/template interposition. A build-local copy
adds exactly the two output synchronization barriers used by CE20: CTA handoff
before reusing shared storage, and warp synchronization before reading rearranged
stores. The original three-shape VERIFY_ONLY racecheck reported 192 peer warnings
and reached the tool hazard limit, despite passing its numerical oracle. The
pristine input headers remain authenticated; effective headers and the selected
two edits have separate hashes. This is an explicitly sync-fixed native peer,
not a pristine throughput claim. Arithmetic, tiles, layout, planner and BF16
partial rounding are unchanged. No FP8 K64 repair or product F32 merge is reused.

Both measured episodes include explicit same-topology update and metadata
staging/upload/wait, original BatchProgram/F1/F2, original FP8 frame identity and
range checks, the actual attention/merge, checked Completion.wait, and owning
output return/release. Initial uploads, extra snapshot allocation/decoder/checked
completion and setup are cold. Their actual seconds, unique-row read/write bytes,
additional physical capacity and teardown seconds are reported separately. No
query-dependent result is precomputed. Layout-content or source-owner changes
cannot silently consume an old snapshot. Dynamic producer/growth is outside this
immutable episode and requires its own charged refresh contract.

The comparison measures resident warm operation rate and its explicit memory
tradeoff. It does not claim direct original FP8 kernel throughput, best-of original
host policy, compressed-cache end-to-end rate, token/s, G06 decode/capture, or a
native-Linux/Formal result. The actual machine is W1-WSL on the existing SM89 GPU;
it shares physical hardware with W1-WIN. The SDK is rebound to the actual CE25
candidate and its source/static/shared consumers; this remains a GPU comparison.
Native private pool upper
bounds remain UNKNOWN. Numerical preflight must pass before A/A or P1 sampling.

The original statistical policy remains eight strata, sixteen workload-matched
A/A runs, dual execution/allocation order, placement perturbation, n15 then the
single predeclared n31 extension, original controls, and five-percent margins.
All qualified failures and measured/unknown costs are retained.

For this WSL workload, the benchmark interleaves one complete episode per
nanobench micro epoch using its original balanced ABBA/BAAB order. Every 2048
paired episodes are summed into one arithmetic-mean group; all elapsed times,
including slow observations, contribute. Sixteen groups feed the original
Result/CompareResult statistics and ordinary operation-rate parser. Group sums
and counts are emitted after timing. This preserves mean complete-episode cost
while sharing the short host/API observation window. It changes only this
benchmark adapter; the public performance core and product remain unchanged.
