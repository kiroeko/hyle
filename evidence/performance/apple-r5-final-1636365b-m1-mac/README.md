# R5 Apple final performance evidence

Candidate `1636365bbe47779de24e84d7c7d1ed8088fe1af2`, tree
`8e5c24179c53b203ba1509a8d5dff0b805548052`. Executed on MacBook Air
Mac16,12 / Apple M4, macOS 27.0 build 26A428, M1-MAC **engineering** profile.
Interactive QoS and AC power were observed; macOS has no public hard P-core pin.
This evidence has no registered-machine Formal certification.

The [comparison contract](../../../docs/implementation/tasks/compute/apple-r5-performance-contract.md)
fixes affine Q4/group64 gate/up, SiLU and F32 down at K4096/H11008/O4096, M1/M16.
Both formal arms include the same Context admission, owner thread, native MLX
execution, checked GPU completion and owning-result release. The peer independently
expresses the full MLX region. This qualifies frontend integration with the adopted
mature implementation; it does not establish a new algorithm win.

## Executed result

Each Direct/Graph run completed 16 actual route A/A processes, both execution and
allocation orders, two placements, 120 product processes (n=15 in eight strata),
30 controls, full-output oracle and all validators. Neither escalated to n=31 or
changed product code after sampling.

| Entry / workload | Eight-stratum median advantage ratio | Strict direction | 5% NI |
|---|---|---|---|
| Direct M1 | 0.999017–1.000872 | 8 INCONCLUSIVE | PASS |
| Direct M16 | 0.998700–1.003551 | 8 INCONCLUSIVE | PASS |
| Graph M1 | 0.998156–1.000970 | 8 INCONCLUSIVE | PASS |
| Graph M16 | 0.998105–1.003708 | 8 INCONCLUSIVE | PASS |

Ratios greater than one favor HYLE. The instrument's practical verdict is PARITY;
every strict CI crosses one, so the direction ledger remains INCONCLUSIVE.
See [Direct summary](compute-p1-summary.json), [Graph summary](graph-p1-summary.json),
[Direct identity replay](compute-replay.json) and [Graph identity replay](graph-replay.json).
Both manifest replays returned QUALIFIED / CURRENT / EXACT and paths_verified=true.
The executed selector results and every validator/raw observation are in
[raw.zip](raw.zip); identity replay alone is not substituted for product gates.

## Natural native boundary

Bare caller-thread MLX uses the same mathematics and checked GPU synchronization,
but lacks HYLE admission/owner-thread/cancel/close/result services. Its classification
remains NATURAL_ROUTE / NOT_COMPARABLE, with no ranking or NI gate. Each entry used
four order strata × 15 processes and its own full-output oracle.

| Entry / workload | Native median range (µs) | Paired extra median range (µs) |
|---|---|---|
| Direct M1 | 2894.356–3007.700 | 5.719–9.712 |
| Direct M16 | 4907.012–5843.852 | −4.229–6.785 |
| Graph M1 | 2897.711–3010.235 | 4.198–11.062 |
| Graph M16 | 5899.009–5987.307 | 5.719–12.185 |

These are ranges across strata, not a pooled CI or same-contract speed verdict.
Direct and Graph were separate serial experiments; their absolute times are not a
cross-entry comparison. [Direct natural summary](compute-natural-summary.json) and
[Graph natural summary](graph-natural-summary.json) retain every stratum. MLX
allocator statistics omit opaque driver memory; independent teardown latency and
complete GPU memory peaks were not measured.

## Retention and limits

[Archive identity](archive-identity.json) records SHA-256
`18e4574786958a54dc1fd479aa1b319ef1614862bb455e9123fbf34aa617eca1`.
All 498 entries retain per-file hashes and source paths; hashes were checked after
transfer. Redundant correctness build directories are excluded. Complete P1/native
records, selected receipts and executed scripts are retained. The
[serial controller](serial-validation-status.json) records successful Air host
sanitizers, Mini native/three-delivery, then Direct/Graph P1 and native diagnostics
without concurrent heavy validation. Correctness has its own
[Air](../../correctness/apple-r5-final-air/README.md) and
[Mini](../../correctness/apple-r5-final-mini/README.md) records.

Qualification covers this M4/OS/toolchain/MLX pin and the two resident FFN archetypes.
It is not model throughput, a CPU/CUDA comparison, or evidence for M5/NAX and other
hardware. Default binding remains the mature MLX compiled region.
