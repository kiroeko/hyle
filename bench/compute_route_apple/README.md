# Apple complete-region benchmark

These finite FFN consumers compare typed Direct or Graph lowering with an independent
pinned MLX expression under the same Context lifecycle. Both arms compile a whole
MLX region. The comparison measures integration overhead, rather than an independent
HYLE mathematical algorithm or a complete model engine.

In an existing static Release build configured with the Apple route and benchmarks,
build the relevant targets directly:

```text
cmake --build <build-dir> --target hyle_compute_route_apple_bench hyle_graph_route_apple_bench
```

The corresponding `_placement` targets change function placement. Runtime staging uses
the product's normal helper to supply `mlx.metallib`. Keep the selected toolchain and
linkage explicit; a changed build is a new measurement environment.

- `HYLE_BENCH_VERIFY_ONLY=1` checks every output against a double oracle decoded from
  the actual unsigned Q4/group64 weights, scales and additive biases. F32 tolerance
  is `1e-5 + 1e-4*abs(expected)`; at least half of each output must reject zero.
- `HYLE_BENCH_AA_ROUTE=selected|native` makes both arms use the named implementation
  for a workload-matched A/A check. `HYLE_BENCH_RIVAL_FIRST=1` and
  `HYLE_BENCH_PEER_ALLOCATION_FIRST=1` exchange execution and allocation order.
- Two ready activation sets alternate. Each hot call includes bind, submit, checked
  completion and release of the result/Completion. Immutable compressed weights are
  shared; Context, stream and invocation slots remain independent per arm.
- Run each built executable directly and retain its output. Analyze repeated runs
  with [the measurement methods](../../docs/developer/performance/README.md), including
  A/A, both orders, numerical checks and actual QoS.

Native allocator counters omit opaque driver/process memory; this does not establish
allocation-free replay or total memory usage. macOS QoS is observed, not hard P/E-core
affinity. Cold owner QoS is not inferred from the caller label.

The raw native-cost branches were removed because they omitted the managed admission,
completion and result-lifetime duties. Original experiments and their revision-specific
boundaries remain in the [implementation history](../../docs/implementation/tasks/compute/apple-v1-four-route-foundation.md).
