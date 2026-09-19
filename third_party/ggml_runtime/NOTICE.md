# ggml common resource integration patches

Upstream: https://github.com/ggml-org/ggml at
`456172ec733a135778adcd32d00e576a58232e45` (MIT; see LICENSE).

The owning CMake authenticates the source and this exact patch manifest before
materializing a build-local copy. Shared cache originals are never modified.
Every edit has before/after hashes and unique context; upgrades require a fresh
resource audit. This directory contains integration edits, not a second ggml
source distribution.

The common edits return NULL for context/pool allocation failures, validate
context alignment/padding and thread counts, preserve native initialization
errors, and roll back partially created worker threads. Apple deallocation
uses the original allocation size. A private pool-creation entry carries the
native failure code to HYLE; it is not an installed public C API.

CPU initialization calls the original one-time time initialization without
allocating a temporary context on every execution. The original math tables,
operators, worker execution, barriers and fusion algorithms are unchanged.
The build identity is supplied by the lock instead of probing the parent's Git
repository. Vulkan backend-specific patches remain in its separate manifest.
