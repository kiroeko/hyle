# C1-AA-Q1 bounded qualification investigation

Baseline `c0aa5ae7`, frozen F1 executable from `64e1e169`, product `ae270af9`.
This is diagnostic evidence, not a new P1 or a performance pass.

The single cold process used the existing `HYLE_BENCH_VERIFY_ONLY=1` path,
selected oneDNN/oneDNN, and the vendor's `ONEDNN_JIT_DUMP=1`. It completed
both real routes and their independent numerical checks for all 16 workloads,
without entering `Bench::compare`. The binary hash, CPU policy, command script,
environment overrides, raw output and all 32 JIT images are in `raw-bundle.zip`.

The benchmark constructs two routes per workload in its fixed iteration order;
the pinned vendor's `register_jit_code` assigns consecutive IDs under a mutex.
Observed IDs are exactly 0..31. All 16 corresponding pairs have identical code
bytes, including Direct LayerNorm 1x2048. `result.json` records each hash and size.
This only excludes code-generation differences in this new cold process.
It cannot recover the earlier failed process's code/data addresses, cache state
or interruptions, and does not establish stable performance.

Source review found equal mathematics, inputs, oracle, independent state and
submit/wait completion, common paired iteration counts and inherited process
affinity. The individual 10% gross guard and the 5% aggregate/factor gates are
different predeclared requirements; passing the latter does not override a
failure of the former. No concrete SIMD, Sched, timer or semantic defect was
established. The original qualification failure remains authoritative.

The bounded investigation stopped without changing product code, validation
code or gates. Q1 remains BLOCKED by missing historical instance observations;
the main oneDNN owners remain BLOCKED and independent C1 work may continue.
Any future recovery must first audit a concrete module-local observation or
contract correction and freeze a new experiment, retaining this failure and
the original thresholds. An unchanged full rerun is not authorized by this record.

The separate, already-qualified original P1 still supports its exact Direct
shape results. Its Graph fusion admission defect and the F1 correctness/delivery
facts remain unchanged. Neither those facts nor this cold observation certifies
F1 preference rules or deletion of Native implementations.

Archive byte/hash verification passed. Automatic approval review rejected the
cleanup command with `blocked by policy`; the bounded 804,716-byte JIT dump
directory remains in scratch. Current binaries and reusable builds are retained.
