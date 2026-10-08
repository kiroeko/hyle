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
