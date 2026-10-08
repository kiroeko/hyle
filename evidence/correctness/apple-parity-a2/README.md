# Apple A2 first wiring validation

Candidate `d4f80861457fa71a1b94a00b2a92b929e30da070`, shared build on the
current MacBook Air identity recorded in [A1](../apple-parity-a1/README.md).
Complete canonical Git bundle was applied to the clean disposable checkout;
no remote product file was copied back. Existing producer cache was reused,
one configure (tests ON, install OFF), build parallelism2, tests serial.

- [Configure](d4f808614-configure.log) and [three owning test targets](d4f808614-build.log): PASS.
- [Direct entry/resources](d4f808614-direct.log): 23 tests,21 PASS,1 SKIP,
  1 FAIL. All twelve cases within the new MLX packed test were rejected by the
  earlier `admitted(operation.kind())` guard despite their native lowering.
  This is a HYLE wiring defect, not a provider numerical/performance failure.
- [Graph entry/resources](d4f808614-graph.log):20 tests,19 PASS,1 SKIP.
  Neither old projection fixture SKIP is counted as PASS. These tests do not
  qualify the new Graph packed requests.

The finite correction admits `MlxQuantized` through that guard. Tests migrate
the eight original Metal codecs from old-decline expectations to real packed
MatMul replay; original Metal GetRows binding is extended to the same native
codec set. Numeric output, changed input replay, checked IDs and resource
tests are required on the next immutable candidate. The original failure
and its budget are preserved; no P1 has run. Full current demand, non-GGUF
artifact decoding, official FP4, rules and three deliveries remain pending.

## Correction candidate02a573d3

[Incremental build](02a573d3-build.log) PASS, no explicit reconfigure.
[Original Metal tests](02a573d3-metal.log):8/8 PASS, including18 physical
codecs' MatMul/replay and GetRows/changed IDs/bounds. This qualifies their
tested geometry and F32 domain, not all559 real contracts.

[MLX tests](02a573d3-mlx.log): embedding PASS; linear combinations PASS
within the mixed test, but all six grouped combinations fail cold stream
validation. Original `mlx/ops.cpp::indices_or_default`, line99 in the pin,
calls `astype(indices,uint32)` without `s`. The adapter passed checked I32
indices, producing a foreign-stream primitive. The finite integration fix
casts the already checked indices toU32 on the owned stream before the public
call, making the upstream cast a no-op. The stream isolation check is retained.
This is a separate, newly observed integration defect; the admission failure
above is fixed and is not retried/reset. Direct and Graph reuse the same
independent dyadic oracle with distinct expert scales, changed expert IDs,
changed activations, embedding rows and checked failure.

Windows Apple-OFF exact-factor configure PASS, SDK/probe/download skipped.
No Apple runtime PASS is derived from that configure. Next candidate's new
Graph coverage and grouped numeric execution remain unverified here.

## Candidate125b2209

[Incremental build](125b2209-build.log) PASS.
[Direct](125b2209-direct.log):2/2 PASS, containing all twelve formats/precision/
linear/grouped combinations and both embedding precisions, two replays,
different experts/IDs/activations and checked failures. Original native MLX
math is used; no numerical tolerance was widened.
[New Graph tests](125b2209-graph.log):2/2 FAIL at the second cold custom
operation admission list, before native compilation or execution. The earlier
trace list already recognizes the operation. The finite correction replaces
all three duplicated typed-custom lists by one Compute-owned native admission
predicate, so Direct/Graph prevalidation/Graph tracing cannot disagree about
this domain. Actual schema, stream, port and runtime checks remain unchanged.
Its first corrected Graph candidate is pending; the original FAIL is retained.

## Candidate58c49dae9: first packed batch passed

[Incremental shared build](58c49dae-build.log) PASS.
[Direct](58c49dae-direct.log)2/2 and [Graph](58c49dae-graph.log)2/2 PASS:
each uses the same twelve affine/MX4/MX8, F32/BF16, linear/grouped cases,
both affine embedding precisions, full independent output oracle, changed
activation/expert IDs, two replays and checked invalid indices. Neither
entry needs a user custom lowering. The three concrete admission/stream
integration defects found above are fixed; all original FAIL logs remain.

[Windows Apple-OFF standalone headers](win-off-headers.log)2/2 PASS. This
small local compiler check ran serially (50.84s+2.70s); it has finished and
no HYLE local compiler/test process remained in the subsequent process check.
User reported a local CPU peak, so further Apple build/numeric/performance
verification stays on Air and no local compiler batch runs concurrently.
Documentation/status checks do not build or run P1. Native Apple standalone
headers and all three delivery seams are still required in A4; selector
delivery gates are deferred, not claimed PASS from this test.

Scope of this PASS is this first typed/native codec batch. All559 real GGUF
contracts,16 real MLX packed geometries,18 artifact requests,8 IQ1 extension
contracts, current complete family/official/MTP regions, cold rules/P1 and
source/static/shared remain open. Uniform canonical Compute+Graph code is
61,190 implementation lines/258 files at this product tree (tests/vendors excluded).

## Mature artifact and all34 packed-demand batch

Candidate `516cff801` [first build](516cff801-build.log) stopped at dependency
audit: the local metadata generator sorted WindowsPath objects case-insensitively,
while the existing portable digest sorts exact relative path text. The single
metadata correction at `1db9ad96` changes only the three new lock digests;
original source/license bytes and all existing dependency entries are unchanged.
[Corrected native build](1db9ad96-build.log) PASS, reusing the existing producer.

[Direct](1db9ad96-direct34.log) and [Graph](1db9ad96-graph34.log) each PASS
the16 real MLX packed requests and18 original NF4/AWQ/CT preparation requests.
Every N/K output is checked, including the cold complete decoded matrix and
its subsequent mature MatMul, two changed-input replays and applicable checked
indices. MoE expert banks explicitly use the shared fixture's at-most8 expert
sample; each log retains originalE and verifiedE. This proves those samples,
not complete giant model residency or256 expert payloads on the16GiB Air.

Native execution is GPU-only MLX; original tensor formulas/codebooks/packing/
scale dtype come from the three locked mature sources. There is no new HYLE
math kernel. SDK/offline/license consumption remains required in A4. Canonical
Compute+Graph implementation is61,241 lines/260 files, tests/vendors excluded.
Input tooling13/13 and validation-selector tooling59/59 PASS. The selector's
wide plan for ownership/dependency metadata is retained; unchanged existing
dependency records and unchanged non-Apple function bodies are not a reason to
rebuild every closed module. Native owning/consumer checks run on Air; remaining
applicable delivery gates are pending, not PASS. No P1 has run.

## Current559 GGUF batch: explicit remaining failures

`ffd3b90f` [test build](ffd3b90f-build.log) stopped at an unnecessary CPU-only
fixture include (`ggml-quants.h`). The single test-header correction excludes
that include when the actual Apple target has no CPU/CUDA IQ1 extension.
No fake capability macro is defined. [Candidateefa1ef48 build](efa1ef48-build.log) PASS.

[Machine-readable summary](efa1ef48-gguf-summary.json): each ordinary entry
runs559 parameterized cases,544 PASS,7 FAIL and8 IQ1-extension SKIP. Direct
takes116.212s, Graph101.567s. Per-case results allow only failed IDs to be
selected after a concrete correction; the full batch is not repeated.
Original [Direct XML](efa1ef48-direct559.xml.gz)/[log](efa1ef48-direct559.log.gz)
and [Graph XML](efa1ef48-graph559.xml.gz)/[log](efa1ef48-graph559.log.gz) are
compressed without losing any result; summary includes rawSHA256. Skip is
unqualified native support, not numerical PASS.

The seven failures are the same in both entries:

- Five Q4_K embeddings (`g395cdd5a9062e045`, `g59bf8a30bf125c3b`,
  `g7be6f59665a8f0f2`, `g9e2b2e4f1be96db6`, `gbb21248aeb6e572d`) violate
  the shared catalogue's exact decoded output. Original Metal
  `kernels/dequantize.h::dequantize_q4_K` performs high-nibble `xb->d / 16.h`
  before F32 conversion, whereas original CPU `dequantize_row_q4_K` first
  widens the half scale. The error pattern is consistent with loss of the
  scale's half subnormal bits. No tolerance is relaxed; mature-source precision
  investigation remains required before choosing a complete exact carrier.
- Q8_0 embedding/head (`g9d504e80cbfefb7f`, `ga846e3d718ee259b`),
  N248320/K8192, exceed INT32_MAX encoded bytes. The fixture's unchecked flat
  U8 shape cast becomes negative; the product also cold-rejects this otherwise
  representable payload through its flat carrier cap. Native logical tensor/
  MTL buffer indexing is not thereby proved defective. The finite correction
  will use a checked multidimensional U8 carrier and obtain it consistently
  through `input_spec`, preserving exact bytes and logical native program.

An additional original Metal IQ1_XXXS candidate was located in
Anemll/anemll-flash-llama.cpp at `a809594fd2035979ad2b7ef23398127c1d07e891`;
its source/encoding/API qualification is pending. A README claim is not runtime
proof and is not a reason to write a HYLE IQ1 algorithm. Official FP4's old
small-scale domain mismatch remains open. A2/whole-goal verdict stays incomplete.

## Large-carrier correction33de7d258

[Native owning build](33de7d258-build.log) PASS. Only the two failed Q8_0
IDs and the no-allocation carrier boundary test were selected:
[Direct](33de7d258-direct-large.log)3/3 and
[Graph](33de7d258-graph-large.log)2/2 PASS. The exact2,161,377,280-byte
weight now usesU8[248320,8704], retaining original packed bytes and native
ggml region. Small payloads retain the prior flat shape; unrepresentable
row axes still returnOutOfRange. Shared fixture uploads follow `input_spec`
in both entries. No complete559 batch or P1 was repeated.

Reusing unchanged passing domains with this delta leaves546 numerically
qualified GGUF requests per entry,5 original Q4_K exact-output failures and
8 IQ1 extension skips. This is a combined evidence verdict, not a new559 run.
34 packed requests are unaffected. Canonical implementation is61,252 lines/
260 files, tests/vendors excluded. Native/header/seam three-delivery validation,
Q4_K exact carrier, narrow IQ1 and official FP4 remain required; no completion
or universal hardware/performance claim is made.

## Plain45 and exact-Q4_K delta

`ce7f52f6` [native test build](ce7f52f6-build.log) PASS;
[Direct45](ce7f52f6-direct45.log)/[XML](ce7f52f6-direct45.xml) and
[Graph45](ce7f52f6-graph45.log)/[XML](ce7f52f6-graph45.xml)45/45 each PASS.
Actual plain model roles preserve storage and both changed-input replays;
large vocabulary heads use the shared CPU/CUDA all-row4096-row tiling policy.

`a0fc488f` [build](a0fc488f-build.log) PASS. Only the five Q4_K exact
embedding failures were rerun: [Direct](a0fc488f-direct-q4.log)5/5 and
[Graph](a0fc488f-graph-q4.log)5/5 PASS, with exact output unchanged. The
original ggml CPU row decoding sequence executes on MLX GPU tensor operations
for only selected checked rows; original bytes/scale/min/nibble remain.
The old half-arithmetic shader is untouched and no HYLE algorithm is invented.

## Original IQ1 delta

`d7c06ce74` [native build](d7c06ce74-build.log) PASS, including original
Metal shader compilation. The previously unavailable eight contracts were
selected once: [Direct](d7c06ce74-direct-iq1.log)/[XML](d7c06ce74-direct-iq1.xml)
8/8 and [Graph](d7c06ce74-graph-iq1.log)/[XML](d7c06ce74-graph-iq1.xml)8/8
PASS. Original pre-encoded fields, original host codec/double reference,
real K/N/all-row outputs and explicit at-most8 expert sample policy apply.
Original Anemll MV/grouped mathematics/grids are borrowed; only buffer-base
ABI and selecting existing MV instead of absent MM are adapted. No engine,
slot cache or second ggml runtime enters the product. New GetRows is explicitly
not claimed for these formats. Old CPU/CUDA no-Apple conditions keep their
existing source composition and numerical implementations.

Combined evidence now covers559 GGUF,34 packed and45 plain requests at these
sample/geometry boundaries in both entries. Source/core changes still require
targeted provider/resource regression and A4 static/shared/source SDK closure;
this is not a replacement full559 run on every candidate. Model family/full
regions, official FP4, cold rules/bounded best-of and three deliveries remain
open. Aggregate whole-goal verdict remains incomplete. Current implementation
is61,328 lines/261 files, tests/vendors excluded.

Clarification: the34/45 real-geometry runners verify successful output/replay;
they do not each submit their `invalid_index_input` automatically. Checked
failure evidence so far comes from the corresponding small ordinary native
Direct/Graph tests. Geometry-specific negative IDs and new exact-row completion
still need the declared targeted resource checks; they are not claimed PASS
by the presence of fixture metadata alone.

Provider/core delta: [d7c06ce74 original18-codec/resource regression](d7c06ce74-metal-regression.log)
8/8 PASS. Source/old-case changes are covered by this targeted owning regression,
not an unreported repeat of all559 requests.

Official FP4 new-source qualification remains failed: original DS4 Metal
group32 pass at9139e2ae (including its floor) and original Safe-mode host choice
both yield222/65280 mismatches,190 nonzero outputs lost. Raw evidence and a
bounded decision are in [the qualification card](../../../docs/implementation/tasks/compute/apple-ds4-fp4-source-qualification.md).
No tolerance, CPU-model fallback or model precision substitution is accepted.

## Exact-row resources and full existing region delta

`4d7dfb34` [resource test build](4d7dfb34-build.log) failed only because the
test ignored the nodiscard Context::close Status. One assertion correction,
no product/source math change: [1d801604 build](1d801604-build.log) PASS,
[Direct resource](1d801604-direct-resource.log)10/10 and
[Graph resource](1d801604-graph-resource.log)2/2 PASS. New exact Q4_K rows
check negative/out-of-range IDs, no published failed result, unchanged retained
leases, recovery in a fresh invocation, Context close and empty indices.
Existing accepted-prefix, cancellation/drain, late native failure and allocation
refusal tests were selected as owning lifetime consumers.

The full existing owning region factors were then run once, excluding the
already qualified current-demand batches:
[Direct](1d801604-direct-regions.log)/[XML](1d801604-direct-regions.xml)
178 PASS/7 missing-input SKIP of185; [Graph](1d801604-graph-regions.log)/
[XML](1d801604-graph-regions.xml)149 PASS/7 SKIP/1 old diagnostic expectation
FAIL of157. The sole failure expected the old ggml Metal carrier name for
Q4_K, while the exact MLX carrier's numeric output succeeded. The expectation
was corrected for the actual carrier; its old loose Q4 comparison was also
tightened to the existing1e-6 bound, never relaxed.

The seven skips were fixture plumbing, not evidence of missing math: canonical
Windows already retained the117,609,568-byte25-file real Qwen3.5 subset.
All25 file/oracle hashes match [the original receipt](qwen35-real-fixture-receipt.json).
Its complete archive (93,349,615 bytes, SHA256
`da8572c829499ab86fc3a80662cdc13a12527b4b4ecbd86b7b366e67b1cc7717`)
was forwarded and checked on Air; source model was not redownloaded.
The initial transfer runner stopped before any test because Air's Python3.9
does not expose tarfile's newer filter argument. Manual exact-target/path/type
validation on the same archive fixed that tooling call without a rebuild/rerun.

`c8bfeb6b` [incremental Graph test build](c8bfeb6b-build.log) PASS;
only seven skipped names and that one failed Graph diagnostic were selected:
[Direct](c8bfeb6b-direct-real-qwen.log)/[XML](c8bfeb6b-direct-real-qwen.xml)
7/7 and [Graph](c8bfeb6b-graph-real-qwen.log)/[XML](c8bfeb6b-graph-real-qwen.xml)
8/8 PASS, no SKIP. Existing region factor evidence is therefore185/185 Direct
and157/157 Graph under this explicit combined evidence, not a hidden second
whole-factor run. The entire Qwen recurrent gate and actual state feedback also
run across nine source profiles. New official0813/0731/DSpark sidecars still
need their specifically indexed full-region witnesses; older region names do
not silently certify all new revisions. Official FP4 decision and A3/A4 remain
open; no P1 has started.
