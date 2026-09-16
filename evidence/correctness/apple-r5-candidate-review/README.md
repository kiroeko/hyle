# Apple R5 candidate source review

Reviewed immutable candidate `1636365bbe47779de24e84d7c7d1ed8088fe1af2`, tree
`8e5c24179c53b203ba1509a8d5dff0b805548052`, after implementation freeze. This is
the separate review pass required by [A5/A6](../../../docs/implementation/tasks/compute/apple-v1-four-route-foundation.md),
performed from failure, consumer and comparison boundaries. No additional agent,
product edit, timing run or build was used for this review.

## Findings and boundaries

| Counterexample checked | Source and existing evidence | Finding |
|---|---|---|
| Reusing a slot while its old native output remains borrowed | `apple_context.cc`: `Invocation::submit` calls `ResultSlot::begin` before clearing arrays; `Results` owns `ResultValues`. `AppleRegion.ChangedInputsRetainedResultsAndNoRetrace` retains an old result across other runs and close. | No new defect found. Native payload is not copied into a shadow output. |
| Caller abandons work before the native owner publishes completion | `ColdJob::admitted_owner` survives the final wake; invocation worker and Completion each retain the state before admission. `finish` publishes terminal state before release. | No new defect found. The previous ColdJob correction is included in this candidate. |
| Failure or cancellation releases a partly submitted GPU prefix | `execute_invocation` performs checked synchronize before terminal publication; its exception path attempts drain and poisons future Context admission. Cancellation retires the slot after drain. | Covered by the actual queued/in-flight/late-failure [lifecycle checkpoint](../apple-r5-lifecycle/README.md). Deliberate GPU hang/device loss was not injected. |
| Concurrent bind, submit or close overwrites live state | Binding is serialized by `binding_mutex`; result-slot admission prevents live reuse. Queue admission and poison use `queue_mutex`; close serializes join. Programs retain runtime state, while the Context owns shutdown. | No new source defect found under the documented handle-move and immutable-input rules. Host sanitizer verdicts remain pending. |
| Graph reinterprets the original DAG or breaks native fusion on every run | `apple_lowering.cc` builds a cold Plan, checks fixed schemas/effects/control edges, removes dead pure nodes and passes one callable to `Context::compile`. The compiled callable owns constants. | Existing tests destroy Graph/constants and check changed outputs; custom Metal lowering is traced once. Hot execution uses the compiled MLX region. |
| Wrong custom arity/dtype or resource failure publishes a partial Program | `trace` checks every output against the Graph schema; `RegionFailure` preserves HYLE status across the native exception boundary. | The fourth Graph test rejects all three cases and then performs successful GPU work in the same Context; [actual log](../apple-r5-ffn-preflight/graph/ctest.log). |
| Typed semantics silently change packing, accumulation or placement | `apple_operations.cc` checks operation arity, F32, 2-D MatMul/Q4, affine group64 and GPU stream before lowering. Q4 uses unsigned low-nibble-first uint32 words and separate scale/bias arrays. | Matches the shared operation contract and independent packed-weight oracle. Other formats/domains are rejected. |
| Benchmark compares different mathematics or lifecycle responsibilities | `route_p1_bench.cc`: `native_region` independently expresses Q4 gate/up, SiLU and dense down; both formal arms use `Harness::execute` for bind, submit, checked wait and lease release. Raw caller-thread MLX uses a separate `NativeHarness`. | Same-lifecycle comparison tests integration cost, not an algorithm win. Raw MLX is explicitly NATURAL_ROUTE / NOT_COMPARABLE. Both keep required GPU synchronization; no timing winner is selected at runtime. |
| Cold setup, thread wake, native allocation or output copying is hidden | Formal timed `execute` includes admission, owner wake, native managed execution, checked completion and result release. Weights/compile/slot creation are cold; host output copying only occurs in verification. | Cost description matches source. MLX-managed metadata and allocations remain real costs; this is not allocation-free recorded replay. |
| A weak oracle lets invalid results pass | Mixed-index data, actual packed-weight decoding, double accumulation, finite-value checks and zero-rejection admission precede timing. | [Final numerical preflight](../apple-r5-ffn-preflight/README.md) passes. The rejected periodic fixture is not counted as final evidence. |
| Users need provider rankings, private glue or a producer resource path | Public `apple.h` and [guide](../../../docs/user/apple-compute.md) describe ready arrays, native ownership, pure callables, close cost and fixed MLX rules. Ordinary facade samples share source/static/shared entry points. | [Air delivery](../apple-r5-product-delivery/README.md) includes real Q4 FFN/RMSNorm with producer metallib fallback hidden and relocated SDKs. Mini and the remaining Windows selector gates are pending. |

No new product FIX emerged from this bounded source review. The applicable final
host sanitizers, Mini native/delivery gates, Windows selector, A/A, P1 and native
cost diagnostics still require their own executed evidence. This record signs
neither those gates nor R5 closure. Hardware qualification remains the actually
tested M4 environment; it does not establish M5/NAX or other-device performance.

Public APIs reviewed: Context create/environment/compile/prepare/close; Program
slot creation; Invocation bind/submit; native Results; affine Q4 description and
typed lowering; Graph compile/custom lowering. Each maps to A1–A6 and the ordinary
Direct/Graph consumers; no benchmark-only product entry point was introduced.
