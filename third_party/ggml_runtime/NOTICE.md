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

The conditional `metal-patches.json` owns the M1-A Apple resource seams: retained
native MTLBuffer imports (including heap-backed buffers), checked completion,
and failure-safe encoding. Original Metal kernels, operator semantics and
fusion remain unchanged. C++ allocation failures are caught before crossing
libdispatch callbacks; accepted commands are drained before owners are released.
The bridge symbols are private. The original embedded Metal sources provide
relocatable delivery; library/pipeline construction stays in cold preparation.
The imported array remains owned and allocated by MLX, never freed by ggml.
