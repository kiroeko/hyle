# ggml CUDA resource integration patches

Upstream: https://github.com/ggml-org/ggml at
`456172ec733a135778adcd32d00e576a58232e45` (MIT; see LICENSE).

The owning CMake applies the exact context-matched edits in `patches.json` to
a build-local copy of authenticated upstream source. Shared source caches and
upstream checkouts remain unchanged. The manifest pins both original and
result hashes; this directory does not contain a second runtime distribution.

Edits preserve native CUDA Runtime, Driver and cuBLAS errors at the private
C++ adapter boundary. Resource owners roll back partial construction; cleanup
does not throw. Capture begin/end/encoding failures release the global capture
counter and discard incomplete graphs. Virtual-memory mapping failures release
their allocation handle and partial mapping. Non-owning buffer wrappers never
free external tensor payload. Backend streams, pools, algorithms, fusion and
capture selection remain upstream-owned. No mathematical kernels change.

The private bridge implementation is in `src/compute_provider_ggml_cuda` and
is compiled into the upstream translation unit. It is not an installed API.
These source edits alone do not certify ordinary Direct/Graph execution or
failure recovery; those need the M1-C native and delivery receipts.
