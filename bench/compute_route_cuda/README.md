# CUDA FFN comparison adapter

The [audited R4 contract](../../docs/implementation/tasks/compute/cuda-r4-performance-contract.md)
owns this finite comparison. It is not a general inference benchmark or a new
product Provider. Build the owning Compute/Graph benchmark targets in a static,
CUDA-enabled configuration; ordinary product and SDK builds do not include it.

Both rows execute Q4 gate/up → precise SwiGLU → FP16 dense down, with M=1/16,
K=4096, hidden=11008 and output=4096. The ordinary route uses HYLE activation;
the peer uses the pinned SGLang precise activation through the same public
Provider lifecycle. The remaining implementations and numerical contracts are
identical. Graph requires actual native capture on both sides.

The two contexts borrow one device memory resource and immutable compressed
constants. They have independent queues, buffers and two invocation/output
slots. Each completed call alternates between two different resident inputs.
Direct includes three actual Completions and waits; Graph includes its actual
captured-program Completion. Cold packing, uploading, preparing/capturing and
output checking are outside the steady-state measurement. Vendor-owned private
storage is not claimed to be known or allocation-free.

Controls use the existing instrument conventions:

- `HYLE_BENCH_VERIFY_ONLY=1`: independent full-shape double-accumulation oracle
  with FP16 stage boundaries; no timing. Normal runs still compare every output
  between the two implementations and verify that changed inputs change output.
- `HYLE_BENCH_AA_ROUTE=selected|native`: run the same HYLE or reference composition
  on both sides, without changing resources, completion boundaries or topology.
- `HYLE_BENCH_RIVAL_FIRST=1` and `HYLE_BENCH_PEER_ALLOCATION_FIRST=1`: independent
  execution and allocation order controls. Separate placement targets retain the
  existing function-padding/alignment perturbation.
- `HYLE_PERF_CALLER_CPU`: the instrument's caller affinity; GPU facts are queried
  once before any measurement and printed with the run nonce.

The reference's device arithmetic/indexing fragment is preserved at the exact
[pin and hashes](reference/provenance.json). Its standalone compatibility layer
provides the selected 16-byte vectors, casts and disabled PDL helpers; host TVM
glue is excluded. [Apache license](reference/LICENSE-sglang) is retained. No
fast-math or intermediate FP16 activation rounding is enabled.

[Compilation and oracle evidence](../../evidence/correctness/cuda-r4-bench-preflight/README.md)
does not qualify an A/A, formal P1, native-wrapper cost or winner rule. Those
remain separate gates under the R4 contract.

## Raw native cost diagnostic

The separate `hyle_compute_cuda_native_cost_bench` and
`hyle_graph_cuda_native_cost_bench` targets use the same frozen Marlin launcher,
HYLE precise activation and cuBLASLt descriptor/heuristic settings. The caller
owns two native workspaces, cuBLAS handles, the stream, event and captured graph
executables. A complete call restores the CUDA context, launches the full chain
or captured graph, records one final event and waits for it. Native errors stop
the diagnostic; this does not provide the public API's admission, retained
ownership, recoverable error or failure-slot retirement services.

These rows are `NATURAL_ROUTE` only. They quantify the complete native boundary
and wrapper cost without pretending that one raw event supplies Direct's three
Completion objects and lifetime services. They reject the A/A environment knob
and cannot be used as the SAME_CONTRACT peer. They require their own full-chain
oracle check before any cost diagnostic is interpreted.
