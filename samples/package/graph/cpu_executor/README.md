# Graph on an application-owned CPU executor

This executable composes the public `sched::Scheduler` and `bulk_execute` with
Compute's borrowed `ParallelExecutorRef`. Graph does not create a second pool or
acquire a product dependency on Sched. It runs RMS normalization and a gated MLP,
using Native or an explicitly requested oneDNN region with the same executor.

```powershell
cmake -S samples/package/graph/cpu_executor -B <build-dir> -G Ninja `
  -DHYLE_SAMPLE_SOURCE=<hyle-source> -DCMAKE_BUILD_TYPE=Release
cmake --build <build-dir> --parallel 2
ctest --test-dir <build-dir> --output-on-failure
```

For an installed SDK, omit `HYLE_SAMPLE_SOURCE` and set `CMAKE_PREFIX_PATH` to an
SDK containing both `graph` and `sched`. Add `-DHYLE_SAMPLE_ONEDNN=ON` to exercise
the oneDNN region as well; this requires the x64 oneDNN feature in either source
configuration or the installed SDK. Static and shared use the same consumer.

## Contract and checks

- Keep the Scheduler valid and **open** until every borrowing Context, artifact,
  Invocation and Completion has finished. The adapter checks bulk failure rather
  than silently dropping work or replaying a possibly completed prefix.
- The executor's body is infallible and never yields or blocks. A Compute call
  nested on a worker callback remains serial there; it must not wait for another
  worker of a one-worker runtime. Scheduler owns the existing nested-progress
  mechanism, not a second implementation in this adapter.
- Input/output storage and each Invocation's scratch are independent. The sample
  uses one/two provider workers, changes inputs, executes two concurrent
  Invocations, and calls an already-prepared Invocation from a worker callback.
  A separate double-precision oracle checks results after actual Completion.
- Task/worker counters are sample instrumentation, not product metadata. They
  report which workers really ran tasks; they do not require the OS to schedule
  every batch simultaneously or establish a speedup.
- Compute metadata and Scheduler lifetime allocations are observed separately.
  No observed replay allocation through Compute's allocator does **not** prove
  anything about oneDNN's private allocator, STL setup or driver storage.

This is a correctness/P0 foundation consumer, not a new public executor type or
a performance result. It does not implement cancellation of partially completed
matrices, choose CPU affinity or provide automatic thread-count tuning.
