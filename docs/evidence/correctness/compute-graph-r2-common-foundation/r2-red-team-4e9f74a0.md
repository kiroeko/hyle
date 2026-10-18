# R2 independent source review of frozen candidate

Candidate: 4e9f74a09b7218f6053038cc5c5b091e73ff0bc5.
This source review is separate from the initial implementation; it does not sign unrun validation.

- Result storage checks alignment and byte overflow before its single cold allocation. The payload destructor is the producer callback; no cross-DSO address identity or native-to-Mem conversion is introduced.
- begin_result acquires exclusive admission, then requires the existing RefCounted acquire-load uniqueness. The last Completion release or retained lease must retire before another write can clear the payload. An abandoned asynchronous Completion retains the owner until native retirement.
- Completion retention snapshots and clears its embedded node before releasing inherited and own references. It never dereferences a node after own.release may destroy it. Synchronous terminal attachment follows the same callback order without an allocation.
- Direct delegated submission validates capacity, bindings and dependency handles before acceptance. The optional cold scratch and submission guard protect concurrent calls. Pending plus reference uniqueness protect the terminal-to-release gap. Native accepted prefixes retain the real Completion. The old shared entry and backend event layouts are unchanged.
- Dependencies remain explicitly borrowed through dependent completion, including their source Completion handles; ordinary users are not promised implicit host waiting or dependency ownership.
- Graph rejects a different Device before Provider queries; equality is the existing Device identity operation. Same-device Contexts, lanes and host storage remain valid. First complete feasible binding returns without a timing/profile comparison; actual budget, rollback, packing, resource and effect checks remain.
- Existing failure, OOM, concurrency, packing and asynchronous-consumer tests were retained or migrated. A protocol fixture is evidence for boundary semantics only. The real CPU owning consumer and CUDA kernel are distinguished from complete R3/R4 route qualification.

Result: no additional product correction identified by this bounded review. Windows full selector, three-delivery owning consumer, WSL/macOS and sanitizers must report their own actual results before terminal closure. P1/default native-route qualification remains R3-R6.
