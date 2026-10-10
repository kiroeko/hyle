# H1 native preflight while C3 builds

Read-only evidence at `3b13bd2c5`; this is not an active closure card, API freeze,
product implementation or hybrid qualification. Windows remains canonical.

| Required mechanism | Inspected current seam | Consequence for the bounded H1 audit |
|---|---|---|
| Typed static DAG and coarse native regions | `include/hyle/graph/graph.h` exposes original node/value IDs, all port descriptions, dependencies and effects; CPU/CUDA route factories retain mature regions | Preserve original IDs/contracts and region boundaries; do not make a kernel dispatcher |
| CPU-required execution | `src/graph/executable.cc:8133` rejects different devices in ordinary CompileEnvironment | Keep ordinary same-device guard; hybrid compilation needs a distinct explicit route entry |
| Complete mature CPU/CUDA mechanism | `src/compute_ggml_runtime/CMakeLists.txt:207–209` enables both original backends only for an actual mixed build | Investigate the already-built ggml scheduler as a whole-region carrier before inventing dispatch; backend array order is private policy, not user preference |
| Cold native allocation and repeat binding | `src/compute_provider_ggml_cuda/ggml_cuda_provider.cc` owns private native buffers/graph per invocation and updates external addresses; entry/exit events bridge native execution | Reuse complete qualified programs; account for private storage and managed/host-blocking behavior |
| CPU access to native pinned host backing | `src/compute_backend_cpu/cpu_backend.cc:41` admits host-accessible resources; CUDA `MemoryKind::kPinnedHost` exists | Verify the exact shared resource and owner lifetime; pageable CPU allocations are not automatically CUDA asynchronous storage |
| Bounded asynchronous transfers and dependencies | `compute/transfer.h`, native CUDA `copy_async_impl` and `wait_completion_impl`; existing internal QueueSequence can import compatible completion | Use existing streams/events and report unsupported or host-wait routes; no new universal Queue/background runtime |
| Output publication and old leases | `compute/result.h` and `detail/generated_result.h` retain native owners until terminal completion and prevent pending/failed publication | Existing borrowed Graph outputs cannot be relabeled as immutable leases; each reusable slot must keep its actual native invocation/backing safe |
| Expert selection/movement | Fixed ggml `456172ec.../src/ggml-backend.cpp:1690–1748` has original host MoE weight range copying by IDs, including host synchronization and checked-by-assert IDs | Preserve full mature MoE math; validate HYLE-facing IDs/ranges first and qualify finite staging/device capacity separately—reduced traffic alone is not bounded residency |
| Scheduler reuse | Same fixed source `:1989–2034` splits/allocates only when its graph is not allocated; reset clears allocation state | Qualify exact prepared/managed replay and lease lifetime; do not reset allocation underneath old results |

The actual H1 card must bind current owners, freeze a finite FIX set, and declare
correctness/resource/cancel/close/replay/three-delivery and native overlap evidence
before product changes. Host-weight/GPU math and true CPU/GPU math remain different
journeys. Engine placement/cache/prediction/eviction, multi-GPU and model loading
remain outside this goal. None of the observations above proves those new journeys.
