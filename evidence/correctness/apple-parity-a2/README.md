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
