# Apple complete-region qualification

Owned by [R5 A5/A6](../../docs/implementation/tasks/compute/apple-v1-four-route-foundation.md).
The [frozen comparison contract](../../docs/implementation/tasks/compute/apple-r5-performance-contract.md)
separates lifecycle-equivalent integration from raw native cost. This is a finite FFN consumer,
not an operator inventory or a model inference engine.

- `hyle_compute_route_apple_bench` and its placement variant compare typed Direct lowering
  with an independent pinned MLX expression under the same real Context lifecycle.
- `hyle_graph_route_apple_bench` and its placement variant compare a public typed Graph
  with that native expression; both become one whole compiled MLX region.
- R7 retired `hyle_{compute,graph}_apple_native_cost_bench` and the bare caller-thread
  MLX branch: it lacked the managed owner, admission, checked completion and
  result lifetime services. Historical R5 observations remain bound to their
  frozen revision; no current target can repeat this mismatched A/B.
- `HYLE_BENCH_VERIFY_ONLY=1` checks complete outputs against an independent double oracle
  decoded from actual unsigned Q4/group64 packed weights, scales and additive biases.
  Deterministic mixed-index data avoids short-period cancellation. F32 oracle tolerance
  is `1e-5 + 1e-4*abs(expected)`; at least half of each output must reject an all-zero result.
- `HYLE_BENCH_AA_ROUTE=selected|native` makes both arms use the named implementation.
  `validate_aa.py` checks 16 finite processes across implementation, placement, execution
  and allocation order, and verifies the compiled MLX blobs against the pinned Git peer.
- Two ready resident activation sets alternate. Each managed hot call includes bind,
  submit, native checked completion and result/Completion release. Immutable compressed
  weights are shared; Context, stream and run slots are independent per arm.

The default is the same mature MLX computation on both sides. A/A or a 5% NI verdict proves
integration cost only. Native allocator counters exclude opaque driver/process memory;
managed replay is not advertised as allocation-free. Cold owner QoS is observed, never
inferred from a caller QoS label. macOS uses the registered QoS policy and does not claim
hard P/E-core affinity.

Build only the named targets, with Apple ON, CPU/CUDA OFF, tests as needed, static linkage,
Release, no LTO and the registered Metal toolchain. Stage `mlx.metallib` through the same
product helper as ordinary C++ consumers. P1 must use the immutable candidate and
`bench/contracts/{compute,graph}-apple-ffn-p1-v1.json` through `scripts/hyle-perf.py`;
manual timing output and this preflight are not qualification evidence.
