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
