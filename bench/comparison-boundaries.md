# Benchmark comparison boundaries

Compare implementations with the same public service responsibilities, inputs,
numerical policies and completion boundary. Record cold setup, recurring work
and memory tradeoffs separately. Pure statistics cannot repair unequal work.

| Entry | Current boundary |
|---|---|
| CPU Direct | Default versus Native through the same prepare, invocation, queue, completion and checked wait |
| CPU Graph | Four complete Graph default/Native pairs; both own the same Graph lifecycle |
| CUDA Direct / Graph | Independent precise SGLang reference through the same HYLE lifecycle; Graph captures both arms |
| Apple Direct / Graph | Independent pinned MLX expression through the same Context owner, result lease and checked completion |
| Vulkan Direct / Graph | Independently compiled ggml shaders through the same admission, completion and output ownership adapter |
| Sched request / wake | Complete request collection and persistent external completion episodes |

The following active pairs were removed, not renamed into eligible benchmarks:

- Sched `structured-spawn-join`: per-item typed result/status/cancellation/lifetime
  cells versus caller-owned Marl result slots behind one batch WaitGroup.
- CPU Graph: four fused and four natural-sequence bare Direct pairs. The bare
  calls omit Graph admission, reusable-slot exclusion and Completion services.
- The six CUDA/Apple/Vulkan `native_cost_bench` targets and their bare branches.
  A raw native wait does not supply each route's managed ownership and failure duties.

Current fixtures retain complete-service peers and numeric checks.
`HYLE_BENCH_VERIFY_ONLY=1` performs
fixture correctness without sampling for the four-route fixtures. Independent
GPU numeric oracles and CPU Native correctness tests remain in place.

The old certification runner and its central contracts have been removed.
Historical Graph v1–v3 and Sched v1–v2 sources and contracts can be found at
`176b8435a8aba43d8e996af969bbc3c6a42b403e` or the original experiment's exact
candidate. Their retired row names no longer exist at HEAD. Historical numbers
remain unchanged and do not become successes through retirement. The interpretation of the
CPU eight rows is recorded in the
[boundary clarification](../docs/implementation/README.md#追溯旧文档).
