# CUDA FFN comparison adapter

The [audited R4 contract](../../docs/implementation/tasks/compute/cuda-r4-performance-contract.md)
owns this finite comparison. It is not a general inference benchmark or a new
product Provider. Build the owning Compute/Graph benchmark targets in a static,
CUDA-enabled configuration; ordinary product and SDK builds do not include it.

Both rows execute Q4 gate/up → precise SwiGLU → FP16 dense down, with M=1/16,
K=4096, hidden=11008 and output=4096. The ordinary route now uses the qualified
SGLang precise activation; the peer retains its independent adapter at the same
pin through the same public Provider lifecycle. The remaining implementations
and numerical contracts are identical. Graph requires actual native capture
on both sides. The v2 evidence compared the previous HYLE activation and found
no complete-chain advantage across all placements and orders. Final-default
A/A and non-inferiority results are not HYLE algorithm wins.

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
- `HYLE_BENCH_AA_ROUTE=selected|native`: run the same default or reference composition
  on both sides, without changing resources, completion boundaries or topology.
- `HYLE_BENCH_RIVAL_FIRST=1` and `HYLE_BENCH_PEER_ALLOCATION_FIRST=1`: independent
  execution and allocation order controls. Separate placement targets retain the
  existing function-padding/alignment perturbation.
- `HYLE_PERF_CALLER_CPU`: the instrument's caller affinity; GPU facts are queried
  once before any measurement and printed with the run nonce.

The v2 W1 contracts use `native-runtime-pair`: a pinned caller and a process CPU
budget of two distinct P cores, with SDK helper placement left to the runtime.
The program reads and checks its actual process mask and priority before and
after measurement. It does not enumerate or take ownership of SDK threads.
The v1 single-CPU qualification remains stopped; its samples are not reused.

Instrumented product runs also require the module's A/A-ready file beside the
executable. The validator clears it to FAIL before checking identities or running
A/A, and writes PASS only after success. A/A processes bypass this cold guard;
ordinary manual/oracle runs without an instrument nonce remain available.

The reference's device arithmetic/indexing fragment is preserved at the exact
[pin and hashes](reference/provenance.json). Its standalone compatibility layer
provides the selected 16-byte vectors, casts and disabled PDL helpers; host TVM
glue is excluded. [Apache license](reference/LICENSE-sglang) is retained. No
fast-math or intermediate FP16 activation rounding is enabled.

[Compilation and oracle evidence](../../evidence/correctness/cuda-r4-bench-preflight/README.md)
does not qualify an A/A, formal P1, native-wrapper cost or winner rule. Those
remain separate gates under the R4 contract.

The final mature-default v3 [Direct evidence](../../evidence/performance/compute-cuda-ffn-p1-v3-755fb7c7-w1-win/README.md)
and [Graph evidence](../../evidence/performance/graph-cuda-ffn-p1-v3-755fb7c7-w1-win/README.md)
now pass qualification and the unchanged 5% NI gate at n=15. Both use the same
precise mature device implementation, so this is integration qualification,
not a HYLE algorithm win. The small negative Graph decode stratum is retained.

## Retired raw-call comparison

R7 removed the two `*_cuda_native_cost_bench` targets and their raw native
branches. One native event did not supply Direct's three Completion boundaries,
admission, retained ownership, recoverable errors or slot retirement. That pair
cannot establish an implementation ranking or a public-route NI verdict.

The [frozen diagnostic](../../evidence/performance/cuda-r4-native-boundary-v1/README.md)
retains its original source identity, oracle and raw observations. Reproduce it
only at that historical revision. Current targets retain the same-service
SGLang reference and the independent numeric oracle; R7 creates no new P1 result.
