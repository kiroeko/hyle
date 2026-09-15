# CPU executor foundation consumer — Windows

Candidate `f6c45efd75e63c96aa406fc5ad29209f73e149b0`, 2026-09-15,
Windows 11 x64 / `kiro-workstation`, MSVC 19.50.35728.0, CMake 4.4.0,
Ninja Release, C++23, locked oneDNN v3.13.2. Scope and correction history:
[probe card](../../../docs/implementation/tasks/compute/cpu-executor-foundation-probe.md).
No product, registry, dependency lock, public build tool or performance instrument
changed from the audit baseline `405fd289`.

## Executed journeys

The same public consumer runs both Native and oneDNN with one/two persistent
Sched workers. Every route checks changed inputs, two independent concurrent
Invocations, nested worker entry, an independent double oracle, prepared replay
metadata allocation counts, and final owner reclamation. Worker shutdown follows
consumer destruction; post-close admission is rejected without running a task.

| Journey | Build/SDK | Result and raw log |
|---|---|---|
| Source `add_subdirectory`, static HYLE | `C:/hyle-scratch/cpu-executor-foundation-source-20260915` | [PASS](windows-source-qualified.log) |
| Installed static | build `cpu-executor-foundation-static-20260915`, SDK `cpu-executor-foundation-sdk-static-20260915`, under `C:/hyle-scratch` | [PASS](windows-static-qualified.log) |
| Installed shared | build `cpu-executor-foundation-shared-20260915`, SDK `cpu-executor-foundation-sdk-shared-20260915`, under `C:/hyle-scratch` | [PASS](windows-shared-qualified.log) |

Producer options were `HYLE_COMPONENTS=graph;sched`, `HYLE_COMPUTE_ONEDNN=ON`,
`HYLE_COMPUTE_CUDA=OFF`, `HYLE_BUILD_TESTS=OFF`, `HYLE_BUILD_BENCH=OFF`,
`HYLE_TRACE=OFF`, `HYLE_INSTALL=ON`; linkage differs as stated above. The shared
producer reused `C:/hyle-scratch/compute-route-onednn-v8-20260914`, reconfigured
for this shipping closure. No SDK was installed into Program Files.

Consumers use `HYLE_SAMPLE_ONEDNN=ON` and either `HYLE_SAMPLE_SOURCE` pointing to
the canonical checkout or only `CMAKE_PREFIX_PATH` pointing to the named SDK.
Builds used `--parallel 2`, followed serially by:

```text
ctest --test-dir <consumer-build> -R ^hyle_cpu_executor_sample$ -V --timeout 120 -j 1
```

The final correction rebuilt only the consumer translation unit/link/staging,
not HYLE or oneDNN. The 17 staged HYLE DLLs matched both the installed SDK and
shared producer by SHA256.

## Observed facts and limits

- Two-worker Native executed 15 executor episodes / 30 worker tasks; oneDNN
  executed 12 / 24. These counts exclude compile/prepare and the standalone queue
  sanity check. oneDNN exercised two nested bulk calls in each worker setting.
- Native's one-task/already-parallel path deliberately executes inline, so zero
  executor episodes for one-worker Native is expected, not a false parallel PASS.
- Native workspace was 61,260 bytes; oneDNN workspace was 43,908 / 43,972 bytes
  for one/two workers. Smaller workspace is not a performance verdict.
- Compute-supplied metadata allocations/reallocations did not increase during
  prepared replay; Compute and Scheduler StatsAllocator instances balanced after
  owner destruction. Vendor-private/STL/OS allocation costs are not inferred.
- P0: Sched bulk uses O(1) caller batch state and O(worker-count) allocation-free
  raw runners, with synchronous quiescence and serial raw-worker nesting. Graph
  gains neither a Sched dependency nor a hidden worker pool. This is not a P1,
  thread-count/affinity optimum or SOTA claim.
- Linux, both Apple machines, sanitizers and the final parallel performance gate
  were not run here. They remain in the overall foundation gate. This bounded
  consumer is not the entire generic Graph package-smoke matrix.

## Artifact identities

| Artifact | SHA256 |
|---|---|
| Consumer `main.cc` | `872B0613341F45D0E6CAD6391BC324FD12482A811A0A99AF6674954F60B27FD7` |
| Source executable | `CCF281998680E51FA14E233FA6FE305F6A56D8866521696EF8D159B226C7EEDC` |
| Installed-static executable | `FAFA91479F2CE6C0754C512387A7809F4B86D5EBEBE058E0121E3BBE49803C58` |
| Installed-shared executable | `EDFC1FC9D9F88C9240266CB0B2734B69C6CF567F0090706B6C497325EEA64641` |
