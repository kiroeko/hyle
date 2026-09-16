# R4 CUDA: captured primary-context delay

This is diagnostic evidence, **not a passed progress regression or P1**.
[Summary](summary.json), [identities](identity.json), [raw records and recipes](raw-and-recipes.zip),
[hashes](sha256.json), and [implementation LOC](loc.json) preserve the result.
The [bounded task](../../../docs/implementation/tasks/compute/cuda-r4-progress-diagnosis.md)
ended with the CUDA owners still `BLOCKED`.

## Actual capture

The diagnostic executable links the unchanged `7b3279c8` product libraries and
adds shared-memory phase markers to the complete FFN fixture. The observer runs
on CPU4; the child is restricted to CPU2 before its first instruction. Capture
was first checked against a known live sleep in the same executable.

The first observation child, PID 36248, remained in its first Context creation
for 10.031 seconds. Its actual primary thread, 63172, accumulated 10.094 seconds
of CPU time; the other four threads each reported zero CPU time. The observer
captured thread context and a minidump, then terminated that child and batch.

Offline x64 unwinding uses the dump's contexts/stack memory and the loaded
images' unwind tables, following the
[Microsoft StackWalk64 contract](https://learn.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-stackwalk64).
The complete primary-thread chain reaches NVIDIA's driver from
`cuda::create_context_impl+0xba`. The executable's map and disassembly independently
identify return RVA `0xa5ca`: call RVA `0xa5c5` targets RVA `0x2eeb0`, the exact
`cuDevicePrimaryCtxRetain` symbol. The phase marker, unfolded stack and call site
agree. The raw stack-address scan is retained separately and is not itself
presented as an unwound call stack.

This occurrence precedes `CompletionReaper::start`. It therefore does not support
changing that lock or Sched. It does **not** establish why the SDK call stopped
making progress, or prove that either earlier uncaptured delay had this phase.

## Bounded intervention: not reached

After locating the actual API, a separate intervention was frozen in `fd16d25d`:
at most 12 launches / 120 seconds, using the same executable. Only another
10-second delay in the first Context creation would trigger a one-time expansion
to CPU2+CPU4 for helper threads, while keeping the caller on CPU2. The observer
ran on CPU6, and the original 30-minute diagnostic budget was not extended.

All 12 launches exited normally in approximately 2.797–3.485 seconds. Therefore
**no affinity intervention occurred**. This does not validate the proposed
recovery, establish single-core causality, or exclude an intermittent problem.
The prescribed batch ended; no further launches or P1 runs were added.

## Verdict

The actual SDK call has been located; a causal remedy remains unproven. Progress
qualification stays `BLOCKED`, and engineering performance stays `INCONCLUSIVE`.
Existing correctness and delivery receipts remain valid for their stated scope.
No HYLE synchronization, GPU algorithm, driver setting, machine profile, common
instrument or performance threshold was changed. Further work requires a bounded
repair/collection contract justified by new evidence; the stopped P1 is not
silently resumed.

The unified LOC script reports **32,665 Compute + Graph implementation lines in
107 files**, excluding tests and third-party code, with its self-check passing.
