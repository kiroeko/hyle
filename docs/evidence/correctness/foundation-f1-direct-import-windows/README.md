# F1 common Direct / owned import initial validation

Product candidate: `5a38cc09b` (includes Direct adapter `65accb90c`). Windows is canonical.
This receipt qualifies the stated common/CUDA source journeys, not the complete F1 package.

## Implemented and verified

- `LeasedOperationInvocation` delegates the original prepared operation and automatically reserves/adopts
  lease uses into its real Completion. Fixed input/output views and use entries are allocated at creation.
- `LeasePool::adopt` owns an existing Buffer without payload allocation/copy. `ReadLease::adopt` owns a
  ready Buffer permanently read-only. Rejected imports preserve the input owner and return budget charges.
- Windows shared **producer**: common 71 tests pass, including seven lease cases; new real cuBLAS Direct
  case passes. Public `resource_leases.h` and `leased_operation.h` each compile in isolation.
- Real 2x2 F32 cuBLAS MatMul with an owned readonly CUDA weight reproduces the independent identity
  reference. Eight warm runs make zero additional calls to the injected lease metadata allocator.
- A real stream host gate holds native work pending: mutable view/publication/reentry are rejected;
  dropping public Completion/adapter/business input handles retains both pool blocks and the imported
  weight until retirement. Original pool drain waits; the final output remains readable and correct.
- An old result reader prevents writer acquisition; a rejected descriptor/port binding admits no use
  and does not poison the output. After the old reader drops, the existing block can run again.

Logs: [common](common-owned-import.log), [native](native-cublas-import.log).

## Review, cost and local adversarial pass

The adapter captures views before write admission, deduplicates one block's exclusive permission,
rolls back unaccepted tickets, and attaches only a fresh preallocated retention to an accepted valid
Completion. The original invocation owns its program; no redundant PreparedOperation copy is kept.
Readonly adoption allocates all fallible metadata before moving the Buffer, then drops its cold root
without closing native-use admission. A readonly subview cannot promote even as the sole reader.
Create still reserves the byte budget before allocating native backing; adoption trusts the original
Buffer's actual alignment and never invents a stronger one.

P0: cold state/view/use/domain/range allocation; bounded O(ports²) duplicate admission in the finite
use batch, one original mathematical submit/Completion, no new Queue/reaper or routing search. The
lease counter proves only HYLE lease metadata replay cost, not zero private allocation throughout
the mature SDK. No timed winner or P1 claim is made. Plain OperationInvocation/TensorView layouts and
ordinary product paths are unchanged by these adapters.

Local adversarial review checked rejected ownership, readonly promotion, old readers, actual pending
publication, detached handles, and native resource/allocator borrowed lifetimes. Untracked raw writable
aliases remain a precondition, not a fabricated ownership proof. This is not the final independent
whole-package red-team verdict.

## Environment and remaining gates

Reused `C:/h7/foundation-f1-q0`, Ninja Release/shared, MSVC 19.50.35728.0, CUDA 13.3.73, shipping hooks OFF.
Developer factors `compute_backend_cuda,compute_provider_cublas`; build parallel 1, tests serial 1.
Dependency source cache `C:/public/tools/hyle-dependencies/v1`. cudart and cuBLASLt runtime are staged
beside the EXE; test launches use CREATE_NO_WINDOW. No inference server is started.

Graph/native Apple/Vulkan bindings, controlled native external owners, late Direct failures/unknown
retirement at the new adapter, concurrency, other applicable platforms, installed static/shared,
relocation/route absence and whole F1 P0/red-team remain unverified. The selector's broader consumers
and delivery gates are deferred to the declared concentrated frozen F1 candidate, not marked PASS.
Previous common/CUDA completion receipts remain scoped to their original product identities.
