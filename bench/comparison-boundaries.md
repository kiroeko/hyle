# Current comparison boundaries after R7

Only a pair with the same public service responsibilities may produce a product
ranking or non-inferiority verdict. The R7 cleanup is limited to the registered
Sched replay and four-route fixtures; it does not reopen Scheduler implementation
or qualify a new performance candidate.

| Entry | Current boundary |
|---|---|
| CPU Direct | Default versus Native through the same prepare, invocation, queue, completion and checked wait |
| CPU Graph | Four complete Graph default/Native pairs in [v4](contracts/graph-cpu-route-p1-v4.json); both own the same Graph lifecycle |
| CUDA Direct / Graph | Independent precise SGLang reference through the same HYLE lifecycle; Graph captures both arms |
| Apple Direct / Graph | Independent pinned MLX expression through the same Context owner, result lease and checked completion |
| Vulkan Direct / Graph | Independently compiled ggml shaders through the same admission, completion and output ownership adapter |
| Sched request / wake | [v3](contracts/sched-p1-v3.json) retains complete request collection and persistent external completion episodes |

The following active pairs were removed, not renamed into eligible benchmarks:

- Sched `structured-spawn-join`: per-item typed result/status/cancellation/lifetime
  cells versus caller-owned Marl result slots behind one batch WaitGroup.
- CPU Graph: four fused and four natural-sequence bare Direct pairs. The bare
  calls omit Graph admission, reusable-slot exclusion and Completion services.
- The six CUDA/Apple/Vulkan `native_cost_bench` targets and their bare branches.
  A raw native wait does not supply each route's managed ownership and failure duties.

The two new contracts have **no new P1 qualification**. Current fixtures retain
the complete-service peers and numeric checks. `HYLE_BENCH_VERIFY_ONLY=1` performs
fixture correctness without sampling for the four-route fixtures. Independent
GPU numeric oracles and CPU Native correctness tests remain in place.

Historical Graph v1–v3 and Sched v1–v2 contracts retain their original bytes and
hashes for archived receipts. They are not runnable contracts for current HEAD:
the retired row names no longer exist. Historical source, module validators and
contracts can be found together at `176b8435a8aba43d8e996af969bbc3c6a42b403e` or
the original receipt's exact candidate. Historical numbers remain unchanged and
do not become WIN/PASS through retirement. The current interpretation of the
CPU eight rows is recorded in the
[boundary clarification](../docs/implementation/tasks/compute/cpu-r3-comparison-boundary-clarification.md).
