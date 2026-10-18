# F1 leased Graph and CUDA native owned imports

Immutable product candidate: `64720f782439811a4bda9c5f423a0418080dcff0`.
Includes leased Graph `3456c4cd8` and native owned imports `64720f782`.
This receipt is partial F1 qualification, not the whole four-route/three-delivery terminal.

## Implemented

- `LeasedInvocation` has fixed cold bindings over the original compiled Graph, automatic use admission,
  original replay/queues, ready owned outputs, source aliases, private workspace retention and close/drain.
  Old result handles reject replay/reset/rebind; independently exported ReadLease prevents writer promotion.
- The private cursor attaches every accepted segment **before** later dependency observation. An opt-in
  retirement latch includes all accepted children plus the final Completion, preserving partial prefixes,
  first failure and original plan exclusion until actual native retirement. The ordinary cursor path uses
  its original terminal publication. No universal executor, new reaper/thread or math/copy kernel.
- CUDA adopted ExternalBuffer can move into a permanent readonly lease or a bounded writable LeasePool.
  The actual ExternalBufferState owns the native resource/Context; common backing/budget destruction
  precedes resource-owner destruction. Borrowed wrappers are rejected, and failed preparation keeps ownership.
- User Graph guide documents ordinary leased usage, results, budgets and the retained compile-time
  borrowed-constant/effect-footprint preconditions. Original private workspace is retained by Graph output
  owner; it is not misrepresented as an independently writable Buffer.

## Actual validation

| Scope | Result / exact evidence |
|---|---|
| Shipping Graph native | [3 PASS](leased-shipping-graph.log): two independent actual cuBLAS branches with synthetic observation decoration; automatic Graph binding/replay/source alias; private workspace view after adapter destruction |
| Instrumented Graph native | [4 PASS](leased-instrumented-with-native-owner.log): adds an accepted source prefix whose following cross-lane dependency fails while the real source stream is held |
| Shipping common lease core | [7 PASS](shipping-common-leases-native-owner.log) |
| Shipping CUDA external ownership | [2 PASS](shipping-cuda-owned-pool.log): existing borrow/adopt regression plus owned lease pool; actual driver address valid with final subview, invalid after its release |
| Shipping Direct native import | [1 PASS](shipping-cublas-owned-read-lease.log): original adopted driver allocation as a readonly cuBLAS MatMul weight, true pending/abandonment and independent numeric reference |
| Isolated headers | leased_invocation.h, detail/lease_access.h, CUDA backend header with actual SDK include context PASS |
| Shipping hook isolation | [Graph](graph-shipping-exports.log), [CUDA](compute_backend_cuda-shipping-exports.log): respective private observation setters absent |

These are **13 shipping directed cases** and 4 instrumented Graph cases. Graph core 31 passed before
the native-owner-only changes; it is not recounted as a new route/delivery gate. No whole operator suite.
The CUDA Direct and Graph use original mature MatMul, not copy-only or a mock Provider.

Eight Graph and eight Direct warm runs call the injected lease metadata allocator zero additional times.
Graph results and independently exported storage reject replay while retained. Real host stream gates
verify no pending publication, admission stopped by close, abandoned Completion retention, explicit
drain, successful result access after close, and native backing recovery. The private-output result
keeps both original workspace and input roots when adapter/compiled handle/business readers are gone.

The prefix fixture initially asserted device-wait > 0; actual preparation has a qualified explicit
host wait between two distinct CUDA execution Contexts. [Original fixture failure](prefix-fixture-assumption.log)
is retained. The assertion was corrected to require an actual device or declared host dependency, and
the same product's [directed correction](leased-prefix-qualified.log) passes. Later four-case run above
passes. This was a fixture route assumption, not a native mathematical/lifetime product failure. The
observation decoration is deterministic test evidence, not a claim that the physical driver error occurred.

## Review, P0 and local adversarial pass

Reference drops use the episode lock barrier so callback tail and owner destruction cannot race drain.
Native Invocation destruction precedes bound payload destruction. Admission reserves all metadata
before native work; rejected tickets roll back, accepted errors poison writes only after actual retirement.
Readonly output publication happens after original successful terminal; every guard slot clears before
reuse. Source aliases retain source lease roots; independent read exports survive result/adapter drop.

P0: cold binding/segment tables, bounded O(bindings²) use deduplication, O(accepted segments) retention,
one original native program/Completion per existing segment. No hot heap growth, routing search, JIT,
new worker, or new native copy/event. Native import keeps a two-word resource-owner release seam in the
opt-in lease domain. It performs no payload conversion/copy. This is mechanism/counter qualification,
**no timed P1 winner claim**. Full package red-team remains pending; this local pass is not its substitute.

## Environment, accounting and pending gates

Canonical Windows, reused `C:/h7/foundation-f1-q0`, Ninja Release/shared, MSVC 19.50.35728.0, CUDA 13.3.73.
Developer factors graph/compute_backend_cuda/compute_provider_cublas; source cache
`C:/public/tools/hyle-dependencies/v1`. Build1/test1 serial, ~44 GiB free RAM, no-window processes,
EXE-local cudart/cuBLASLt/HYLE DLLs. One instrumented qualification and shipping restore; final cache
is **HYLE_TEST_INSTRUMENTATION=OFF**. No Torch, new inference server, P1 or remote product changes.

[Unified LOC](loc-current.json), clean immutable candidate self-check PASS: Compute+Graph non-test
**69,557 code lines / 308 files**; all HYLE self-written non-test **164,784 / 659**.
Counts exclude tests/vendor/samples/docs, and do not mean mathematical kernel count or completed FIX count.

Still unverified: normal CPU route consumer, remaining Apple/Vulkan typed lease bridges and native
math/lifecycles, CPU+GPU composition adapters, applicable other platforms, installed static/shared,
relocation/absence and full F1 cost/red-team. F2/F3 implementation has not begun. Owners stay AUDITED;
no completion or performance PASS is inferred for these pending scopes. Broader selected consumers
and delivery gates remain in the concentrated F1 candidate batch.
