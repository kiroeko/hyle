# ggml Vulkan integration patches

Upstream: https://github.com/ggml-org/ggml at
`456172ec733a135778adcd32d00e576a58232e45` (MIT; see LICENSE).

HYLE does not vendor the runtime here. `patches.json` records exact source and
result hashes and the complete, context-matched changes applied only to a
build-local copy of authenticated upstream source. The immutable source cache
and upstream checkout are never modified.

The changes make `ggml_init` fail without aborting on context/arena allocation,
invalid external alignment or padding overflow; disable nested shader/compiler
parallelism; and accept the owner's pinned build identity instead of probing a
parent application's Git repository.

The SiLU and separate SwiGLU shaders additionally map negative infinity to the
activation's positive-zero limit, as required by HYLE's common semantics.
The gate still multiplies that activation by its other input, preserving
zero-times-infinity/NaN behavior. Finite formulas, fusion and resource planning
remain upstream implementations; these are explicit boundary corrections.

The two Argsort shaders retain the upstream bitonic sorting networks, barriers,
dispatch and scratch management. Their comparison uses original F32 storage
keys and source indices to provide stable ties, equal signed zeros, NaN-last in
either order, and valid elements before tail padding. Descending order is
encoded in that comparison; output reversal is removed to preserve tie order.
This is an explicit semantic adaptation, not an unmodified upstream algorithm.

The native embedding edits isolate Vulkan instance/dispatcher/registry state,
publish fully initialized devices, roll back partial buffer/backend creation,
account device memory through optional HYLE reservations, and permit an owned
native-buffer/queue bridge. The host retains its terminate handler and supplies
loader function pointers. The bridge implementation remains private HYLE code.
Device creation also enables `VK_KHR_portability_subset` when the physical
device advertises it, as required by Vulkan VUID 04451. This uses the existing
extension enumeration; devices without that extension do not request it.
These changes do not certify asynchronous partial-submission failure handling
or the ordinary Direct/Graph adapter; those have separate remaining acceptance.
The CPU ggml provider remains governed by its separate blocked record.
