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

## Windows CPU-only delivery delta

[Delivery receipt](windows-delivery-receipt.json) and
[producer/relocation/loader/export logs and actual consumer](windows-delivery.zip)
record 17 successful steps and one retained standalone-consumer compile failure.
Four new calls omitted their explicit `Operation` wrapper; only that consumer
was corrected. Product source and rule conditions were unchanged.

Source and relocated shared/static execute all 43 cold requests in both ordinary
contexts and 48 numeric/replay checks. The eight added cases exercise actual
retired-math odd-byte Unary/Binary and stable Softplus through Direct/Graph.
Windows static singleton CPU2 hits all qualified exact choices; unrestricted
and shared use legitimate defaults. Required DLLs neighbour the executable;
installed runs use only System32/Windows PATH. Shared export scan passes.

Other-platform source/static/shared and mixed-envelope deltas remain pending.
Unchanged qualified mathematics and rule conditions reuse F4 evidence; no additional
timing race, sample-budget increase or self optimization is performed.

## Portable owning and equivalent cleanup

[Mac/Linux receipt](portable-owning-receipt.json) and
[complete logs/XML, original failures and input delta](portable-owning.zip)
pass 144 owning cases on each platform without skips. Mac uses the original
507bad96 product; Linux uses `64b454fa23404d26bead60d495eda9475c802e32`.

The exact canonical product delta removes two uncalled private helpers and moves
the existing reduction NaN/return exits into explicit defaults. It changes no
admitted math, resource protocol, ABI or qualified rule. Three original GCC Werror
failures remain; only the finite cleanup was implemented. Windows/Mac mathematical
and delivery evidence is retained under this explicit equivalence, while pending
Linux/mixed delivery uses the updated input. Implementation LOC is 60,161 / 249
files by the unified script, excluding tests/vendor/bench.

## CPU-only three-platform delivery complete

[Mac receipt](mac-delivery-receipt.json) and [raw/source archive](mac-delivery.zip)
record 20 successful steps; [Linux receipt](linux-delivery-receipt.json) and
[raw/source archive](linux-delivery.zip) record 18 successful steps. Both run
the same 43 requests through two contexts and 48 numerical/replay checks in
source/static/shared, with relocated runtime loading and actual exports.
Mac's whole cached delta takes 61.329 seconds; Linux's takes 216.922 seconds.
Mac preserves the 507bad96 input under D5 equivalence; Linux binds updated
64b454fa and records the original archive plus exact three-file delta.

This finishes CPU-only platform delivery. Four mixed-envelope CPU deltas and
F5/F6 terminal audit remain pending; GPU mathematical/performance states are unchanged.

## F5 terminal acceptance

[Terminal receipt](terminal-receipt.json) and [complete terminal delta](terminal-delta.zip)
finish all applicable F5 gates. Win CUDA/Vulkan, Mac Apple and Linux CUDA mixed
CPU envelopes pass 15/15/20/18 steps, all actual source/static/shared 43/48 checks.
Three native header self-containment gates pass with the real platform compilers.
The [current card](../../../docs/implementation/tasks/compute/cpu-final-f5-retirement.md)
records seven-gate closure at product 64b454fa. F6 final goal audit/cleanup remains
separate; no new GPU mathematical/performance claim or performance sampling.
