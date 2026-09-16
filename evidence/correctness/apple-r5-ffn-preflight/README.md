# Apple R5 FFN numerical and Graph refusal checkpoint

Scope: [R5 A5/A6](../../../docs/implementation/tasks/compute/apple-v1-four-route-foundation.md),
under the [premeasurement comparison contract](../../../docs/implementation/tasks/compute/apple-r5-performance-contract.md).
This is a correctness checkpoint, not A/A, P1, Mini validation or route closure.

- Air Mac16,12 / M4, macOS 27.0 build 26A428; Release/static/WERROR, Metal-only locked MLX.
- All four executables (Direct, Graph and each raw-native diagnostic) completed
  `HYLE_BENCH_VERIFY_ONLY=1`. M1 and M16 each use two distinct inputs and independently
  decode actual Q4 packed weights into a CPU double full-output oracle.
- Maximum absolute error: `9.68575478e-8`; at least 65,489/65,536 M16 elements and
  4,095/4,096 M1 elements reject an all-zero output under the frozen F32 tolerance.
  Both mathematical arms produce exactly equal outputs for these inputs.
- The initial periodic fixture was rejected before timing: cancellation made its M1
  output smaller than the original absolute tolerance. Mixed-index data, a tighter F32
  tolerance and explicit zero-rejection qualification fix this benchmark defect.
  No product code or performance result was changed by that correction.
- All 943 compiled MLX tracked blobs match the clean independent pinned Git peer
  `1f8e74e3f12f31365464a6867c6579f0e9b29d85`; one native runtime supplies both arms.
- Graph's new fourth test rejects incorrect custom output arity/dtype and preserves
  preparation ResourceExhausted, then successfully compiles and executes real GPU
  work in the same Context. Four Graph GTests pass; the unchanged 11 Compute tests
  retain the [lifecycle checkpoint](../apple-r5-lifecycle/README.md).

Raw FFN logs, executable hashes and durations are in `ffn/`; Graph CTest and GTest
logs are in `graph/`. `ffn/checkpoint-identity.json` records canonical input hashes,
verified against the Windows workspace after copying only evidence back from Air.
The build was an incremental validation copy, not an immutable P1 candidate.
