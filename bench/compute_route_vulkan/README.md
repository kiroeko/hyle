# Vulkan complete-FFN benchmark

The Direct and Graph consumers compare complete FFN execution with independently
compiled original ggml shaders from `456172ec733a135778adcd32d00e576a58232e45`.
`consumer_shapes_v1.json` describes M1/M16, K4096/H11008/O4096 inputs.

`reference_shaders.py` compiles a separate ggml checkout instead of reusing product
SPIR-V. The reference independently describes packing, push constants, specialization
and four-dispatch FFN execution. Both arms include native admission/completion;
Graph also includes its output lease and slot lifetime. This measures integration of
the mature implementation, not an independent HYLE algorithm win.

In an existing static Release build with Vulkan benchmarks enabled, provide
`HYLE_VULKAN_REFERENCE_SOURCE_DIR` for the pinned checkout and build the relevant targets:

```text
cmake --build <build-dir> --target hyle_compute_route_vulkan_bench hyle_graph_route_vulkan_bench
```

The `_placement` variants add a cold prefix object rather than padding hot functions.
Link maps record symbol positions. Use the built executable directly, with the normal
Vulkan loader and staged runtime available.

- `HYLE_BENCH_VERIFY_ONLY=1` checks inputs and every output against a double oracle,
  rounded to F32 at operation boundaries. Timing also rejects invariant outputs or
  changing resident allocation counts.
- `HYLE_BENCH_AA_ROUTE=selected|native` runs workload-matched A/A. Execution and
  allocation order can be exchanged with `HYLE_BENCH_RIVAL_FIRST=1` and
  `HYLE_BENCH_PEER_ALLOCATION_FIRST=1`.
- Save repeated output and analyze A/A, both orders and placement using
  [the measurement methods](../../docs/developer/performance/README.md).

Cold preparation, peak and resident VkDeviceMemory are separate outputs; driver and
pipeline-private allocations are excluded. Record actual caller placement and resources.
MoltenVK on the MacBook Air uses observed QoS, not hard P/E-core affinity.

The retired raw native-cost branch lacked cancellation, abandoned-work quarantine and
owning output leases. Its historical timings do not rank equivalent public routes.
Original experiment details remain in the [implementation history](../../docs/implementation/README.md#追溯旧文档).
