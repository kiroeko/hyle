# F5 CPU source retirement: verification in progress

Product snapshot: `507bad9698f9988d371c59e5661a69e4394c580a`.
[Current card](../../../docs/implementation/tasks/compute/cpu-final-f5-retirement.md) is CANDIDATE;
this partial archive does not close F5 or the CPU goal.

The [source ledger](../../../docs/implementation/tasks/compute/cpu-final-f5-source-ledger.json)
records physical retirement of 14 redundant semantic kinds, narrowing 20 qualified
kinds to needed branches and retaining named Copy/RMS resource demands. SIMD/Mem,
View and original borrowed TorchAO/ATen/MLX/DS4 programs remain. No new math or P1.

[Windows receipt](windows-correctness-receipt.json) and
[raw logs/XML, complete CTest output, canonical tests and drivers](windows-correctness.zip)
preserve the original failed aggregate and subsequent bounded fixture corrections:

- Native owning: 144 tests pass without skips.
- Ordinary Direct: all 832 tests pass without skips.
- Ordinary Graph: original 851 tests run without skips, 849 pass; two stale
  oneDNN Select identity expectations fail. Numeric/bit/state/replay checks pass.
  Updating those expectations to the fixed ATen default passes the two targeted
  cases, retaining the other 849 results rather than rerunning the full batch.
- Generic Graph: original 70 retired-math fixture failures, then 27/1 migration
  failures remain recorded. Final 283 tests pass without skips. Real mature math
  is used on the CPU backend; asynchronous protocol fixtures retain their original
  test backend and eligible Native resource mechanisms/Exp tail. These mock regions
  qualify Graph protocol behavior, not mathematical carrier or route completion.
- Compute, CPU backend, ggml, oneDNN owning/composition/region, Graph logical and
  the actual CPU executor consumer pass. Unknown mature private storage stays
  unknown; complete finite-budget tests use retained RMS/Add resource mechanisms.

Platform owning, source/static/shared and mixed-envelope deltas remain pending.
Unchanged qualified mathematics and rule conditions reuse F4 evidence; no additional
timing race, sample-budget increase or self optimization is performed.
