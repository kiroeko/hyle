# ggml Vulkan shader subset

These unmodified sources come from ggml revision
`456172ec733a135778adcd32d00e576a58232e45` and retain its MIT license.
`SOURCE.json` identifies every upstream shader/include byte sequence.

HYLE compiles the F32 matrix, matrix-vector, SiLU gating and private layout-copy
variants. Conditional includes retain other upstream definitions; their presence
does not claim those formats or operations are exposed by HYLE.

The host push-constant layouts, fast division constants and legal tile recipes
in HYLE's Vulkan operation adapter are adapted from this revision's
`src/ggml-vulkan/ggml-vulkan.cpp` under the same license. HYLE owns the native
resource, preparation, recording, completion and Graph integration code.

This snapshot is independent of HYLE's separately locked ggml CPU dependency.
It neither changes that dependency's identity nor resolves its blocker.
