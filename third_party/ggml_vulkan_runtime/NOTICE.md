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

Native Vulkan initialization, queue/ownership bridging and partial-submission
failure handling are not yet admitted by these patches. The CPU ggml provider
remains governed by its separate blocked record.
