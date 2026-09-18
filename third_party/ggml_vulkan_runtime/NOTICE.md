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
parent application's Git repository. They do not change operator algorithms.

The native embedding edits isolate Vulkan instance/dispatcher/registry state,
publish fully initialized devices, roll back partial buffer/backend creation,
account device memory through optional HYLE reservations, and permit an owned
native-buffer/queue bridge. The host retains its terminate handler and supplies
loader function pointers. The bridge implementation remains private HYLE code.
These changes do not certify asynchronous partial-submission failure handling
or the ordinary Direct/Graph adapter; those have separate remaining acceptance.
The CPU ggml provider remains governed by its separate blocked record.
