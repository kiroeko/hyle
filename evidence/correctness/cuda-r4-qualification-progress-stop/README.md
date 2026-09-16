# CUDA R4 qualification: process progress stop

This is **failure/qualification evidence**, not a passed P1 or a new product
delivery. [Summary](summary.json), [raw logs and recipes](raw-and-recipes.zip)
and [implementation LOC](loc.json) retain the actual observations.

## What ran

- `cc38c398`: fresh MSVC/Ninja static build; actual CUDA/CRT DLLs beside the
  executables. Sixteen full-FFN A/A processes passed, but only the calling
  thread was pinned, while the instrument pins the entire process. These A/A
  results do not qualify the latter policy. The first product child exceeded
  120 seconds and was stopped. Its log contains only initial GPU telemetry.
- `7b3279c8`: the module validator now applies and verifies the same process
  mask `0x4`. All 16 A/A processes passed. Overall median ratios for SGLang
  M1/M16 and HYLE M1/M16 were 0.999181, 1.002842, 1.004556 and 1.001383.
  Individual process and crossed factor limits also passed. The CPU/GPU
  algorithms, 5% margin, order policy and sample schedule did not change.
- The second qualification encountered another long-running child at the
  52nd product process. PID 49576 had an observed age of 168.532 seconds and
  162.797 seconds of CPU time; active thread 35060 accounted for that CPU time.
  **The thread's role and stalled phase were not identified.** It exited before
  the attempted context/dump capture (`OpenProcess`, error 87). Its completed
  log has normal output checks and pair-preparation measurements; these do
  not locate the elapsed-time gap. The remaining qualification was stopped,
  leaving 90 complete product logs plus one interrupted log. No final P1
  manifest, NI gate, default winner, or Graph P1 was accepted.

## Bounded diagnosis

The same complete executable subsequently completed under both one and two
P-core process masks. Native CUDA memory/event/cuBLAS-heuristic programs,
HYLE Context/Q4 packing, a reduced-priority caller, and a phase-logged copy
also completed. A final **24-process maximum**, 10-second observation attempt
completed 24/24 with no reproduced delay. All recipes and results are retained.

These observations **do not establish** a CUDA driver, Sched, reaper spinlock,
or single-core restriction as the cause. No product synchronization, algorithm,
driver setting, machine profile or common instrument was changed on that basis.
The initial description of a cold-start problem is only a working hypothesis;
the second occurrence lacks a stack or phase-local elapsed-time trace.

## Verdict and remaining work

The finite qualification attempt is stopped: `INCONCLUSIVE`; R4 is `BLOCKED`.
Both attempts' samples are excluded from default-rule and performance claims.
Previous ordinary, host/device checking and C5 delivery receipts remain intact;
the newly observed progress risk is unproven. Graph P1 and raw-native cost
timing were not run. Before another qualification, a separate bounded progress
diagnosis must identify the actual phase/owner and freeze any justified repair
or evidence correction. Changing affinity, increasing n or granting an implicit
exception does not resolve this stop.

The canonical product implementation remains **32,665 lines in 107 files**,
excluding tests and third-party code; the unified script's self-check passed at
`7b3279c8`. The goal remains incomplete; Apple, Vulkan and final review remain.
