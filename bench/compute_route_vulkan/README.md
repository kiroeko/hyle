# Vulkan complete-FFN qualification

The [R6 performance contract](../../docs/implementation/tasks/compute/vulkan-r6-performance-contract.md)
freezes the consumer, peer, ownership, precision and finite budget before timing.
`consumer_shapes_v1.json` records the registered M1/M16, K4096/H11008/O4096 shapes.

`route_p1_bench.cc` uses ordinary typed Direct or Graph APIs. `reference_shaders.py`
compiles original shaders from a separate, clean ggml checkout at
`456172ec733a135778adcd32d00e576a58232e45`; it does not reuse product-generated SPIR-V.
The reference independently describes packing, push constants, specialization and
the four-dispatch FFN. Both arms pay the same native admission/completion lifecycle.
Graph's private benchmark adapter also pays the same output lease and slot costs.
This qualifies mature implementation integration, not an HYLE algorithm win.

`HYLE_BENCH_VERIFY_ONLY=1` checks both inputs and every output against an independent
double oracle, rounded to F32 at operation boundaries. Timing also compares the two
arms and rejects invariant outputs or changing resident allocation counts. Cold
preparation time, peak and resident VkDeviceMemory bytes are reported separately;
these counts exclude driver/pipeline-private allocation.

`validate_aa.py` uses the existing machine policy and nanobench parser for one finite
16-process qualification per entry/profile: selected/reference implementations,
execution order, allocation order and placement. The placement executable adds one
cold prefix object; it does not pad hot functions or alter the product library.
Link maps record actual symbol positions. Both Windows DLL staging and popup
suppression happen before execution. M1-MAC means the MacBook Air, with measured
interactive QoS, not hard CPU affinity.

Four `bench/contracts/*-vulkan-ffn-*-p1-v1.json` contracts drive the unmodified common
runner. Run only after API/algorithm freeze and candidate registration. Windows uses
the registered two-P-core native-runtime budget with a pinned caller; MoltenVK has
its own machine and toolchain contract. Exact reference inputs/SPIR-V, executable,
machine and child-policy identities bind A/A reuse. A failed finite attempt cannot
be silently repeated by the validator.

R7 retired `*_vulkan_native_cost_bench` and the raw submission branch. Its checked
fence wait lacked cancellation, abandoned-work quarantine and owning output
leases, so the pair cannot establish a public-route NI verdict. Historical R6
raw observations remain at their frozen source/evidence identity. Current
same-service shader references and independent oracles remain; R7 does not run P1.
