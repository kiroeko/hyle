# F1 Apple/Vulkan native binding and readonly import initial validation

Product candidate `0ee0c77796a30b63ffd241413c9a63b2f798103a`, clean Windows canonical tree.
Initial native implementation `f8e88365e`; subsequent corrections are retained in Git.
Owners compute_route_apple / compute_route_vulkan are AUDITED. Graph native lowerers were consumed,
not reopened or changed. This receipt does not sign the complete F1 package.

## Implemented and proved

`MemoryResource::binding` creates a cold readonly original MLX array / Vulkan Tensor over a common
ReadLease, retaining the actual lease root. `import_readonly` retains a successfully ready original
native owner and exports permanent readonly bytes. No payload copy, new math/copy kernel, universal
Queue, alternate native executor, or repeated Provider selection is introduced.

- [Windows Vulkan case PASS](vulkan-windows.log): original native Context upload → readonly import →
  original MatMul Program → Graph with computed result and input View alias. Native handles match
  original buffer/origin. Descriptor/resource/allocator rejection, numeric reference, business-handle
  abandonment and retained source alias after original Program/Invocation/result destruction are checked.
- [MacBook Air Apple case PASS](apple-air.log): actual shared Metal LeasePool data → owning MLX binding →
  mature MatMul → compiled Graph/View alias → ready native readonly import → native rebinding. Both
  mathematical outputs match the independent F32 reference, binding retains readers, rejected descriptors
  preserve them, and the retained native alias keeps exactly its source block alive after other owners drop.
- Isolated `vulkan_memory.h` and `apple_memory.h` syntax checks pass with their SDK include contexts;
  their conditional ON surfaces are also compiled by actual enabled native consumers. Full installed
  ON/OFF header journeys remain pending.

This is **one case on each actual route**, not a whole operator/delivery suite. No test skip is counted PASS.
The first Vulkan fixture used Tensor arguments to prepare, which declares constants rather than dynamic
inputs; it was corrected to descriptions before binding. Native Result wait returns ResultLease, and
published source alias uses the existing View node. These fixture corrections changed no Provider math.
Apple import needed explicit rejected-backing reset for libc++ and a readonly resource minimum alignment
of one byte; it must not claim the normal new-allocation alignment. Both product corrections were applied
in Windows before syncing and validating on Air.

## Ownership review, P0 and local adversarial boundaries

MLX's real array/set_data/copy_shared_buffer owner chain keeps the ReadLease; its deleter captures that
real owning value. The readonly imported resource retains the actual ready MLX array, with common
LeaseAccess retaining that resource until backing/metadata destruction. Vulkan uses the original
Buffer view plus a holder of both original Buffer and ReadLease. Its imported allocation token stores
the exact metadata allocator and original native Buffer, so it neither frees another allocation nor
charges original backing twice.

Native Program/Invocation/result retention remains the original runtime's responsibility, including
late/unknown retirement. These readonly bindings themselves are retained **readers**; core LeasePool
drain waits admitted LeaseUseBatch uses, not all retained native readers. A resource must outlive its
native bindings/results; its original Context closes/drains native work. No claim is made that this
cold view conversion alone is a complete native lease-aware submission/admission facade.

Vulkan binding rejects staging lacking storage usage, wrong resource and description, and requires
four-byte physical tail/origin within the already-owned lease range. No padding spills into a neighbour.
Apple carrier/offset/shared buffer bounds are checked, imported readonly views cannot become a writer,
and native writable aliases remain an explicit forbidden condition. Options track new wrapper metadata;
native source accounting is retained, not recharged. MLX opaque descriptors use the existing explicit
unattributed budget policy. No total-device-memory or timed performance claim is inferred.

P0: metadata/native wrapper construction is cold, ordinary repeat execution still uses original
Programs, no new event/command/thread or payload allocation/copy. Broader repeat/allocation counters,
negative/quota cases, native pending/cancel/unknown and full red-team are pending. This source review
and small adversarial set are local qualification, not final package closure.

## Machines, transport and accounting

- Windows uses reused `C:/h7/w-vulkan-native`, MSVC 19.50.35728.0, Ninja Release/shared, locked mature
  ggml/IREE sources and shaders; build1/test1, EXE-local HYLE/runtime DLLs and no-window launch.
- Air actual identity: Mac16,12, 16 GiB, macOS27.0.1 / 26A434, AppleClang21.0.0,
  CMake4.3.4, Ninja1.13.2, signed Metal32023.883 through the documented metal-17F109 shim.
  Reuses `.../a-apple/selected-graph/shared-sdk-build` as a validation producer (INSTALL OFF, tests ON).
  Build1/test1, at least4GiB available pages before launch, serial with Windows heavy work.
- Windows full product/build/test archive was mirrored to the disposable Mac source directory;
  [source manifest](source-manifest.txt) identifies the candidate. All identical files preserve mtimes
  by content comparison for cache reuse. No remote product edits or reverse synchronization.
  Earlier missing profile PATH / SDK-tests flag conflicts were setup corrections, not native product failures.
- [Unified LOC](loc-current.json) clean candidate self-check PASS: Compute+Graph non-test
  **69,738 code lines / 308 files**, all HYLE self-written non-test **164,965 / 659**.
  Tests, vendor, samples and docs are excluded. This is implementation LOC, not a kernel count.

Pending: remaining F1 native submission/admission and lifetime negatives, CPU consumers, CPU+GPU
composition seams, Vulkan Air / applicable WSL native qualification, source/static/shared installed
delivery/relocation and whole F1 review/P0/red-team. F2/F3 are still unimplemented. Owners stay AUDITED;
none of these pending scopes is marked PASS or counted as completed package work.
