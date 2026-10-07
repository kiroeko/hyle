# Runtime staging collector: bounded tool result

Task: [finite build-tool repair](../../../docs/implementation/tasks/build/runtime-staging-diamond-walk-v1.md).
This is dependency-collection correctness/P0 evidence, not an operator or P1 verdict.

The ten-layer diamond has one reachable runtime asset. The original walker visited
4,093 paths; the repaired walker visits 41 target/condition combinations and returns
the same asset. Debug/Release tests also cover duplicate links, nested constant
wrappers, aliases, genuine conditional edges, cycles and independent collections.

Products `afba74122c26d49489bee895c8b14dc5381426d2` and
`61ad57fb6c054e8b1b46844e8aea75dbbdbc8948` implement per-collection visitation and
constant-wrapper normalization. The real Windows CPU+Vulkan configure completed
in 52.297 seconds. The original owned configure was stopped after reproducing the
source defect at 976.375 seconds; its result remains interrupted, not PASS.

[Receipt](receipt.json) binds the [raw before/after trace archive](trace.zip).
Collection retains each distinct complete condition and does not cache across
configuration calls. Darwin loader changes belong to CPU F3 and are not qualified
by this tool result.

[Cleanup record](cleanup.json): both redundant scratch traces, totaling
85,641,196 bytes, were removed only after archive/member hash verification.
Their original content remains in `trace.zip`.

The finite tool task is CLOSED. At `e0388fbda` the normalized collector section
still matches candidate `61ad57fb6c054e8b1b46844e8aea75dbbdbc8948` exactly
(SHA256 `ce0a7f714570d5ee093473b52081ae52320a91a3491b3e52e36f6f5a4df14029`).
Archive SHA and every member were rechecked; no product rebuild or P1 was run
for this documentation-only terminal audit. CPU SDK and Darwin qualification
remain governed by their own cards.
