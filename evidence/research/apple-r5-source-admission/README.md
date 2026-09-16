# R5 pinned source admission

2026-09-17, canonical Windows baseline `553a7d65`. Source inspection and archive
identity only; no shipping build, correctness or performance PASS is implied.
Current task: [Apple R5](../../../docs/implementation/tasks/compute/apple-v1-four-route-foundation.md).

- MLX v0.32.2 `1f8e74e3f12f31365464a6867c6579f0e9b29d85` (MIT).
- Official Apple Metal-C++ 26 `f567ed836e4cbb85788c42115a2682bbe68097ee` (Apache-2.0).
- nlohmann/json v3.11.3 `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03` (MIT).
- fmt 12.1.0 `407c905e45ad75fc29bf0f9bb7c5c2fd3475976f` (MIT).

[sources.json](sources.json) records official codeload archives, raw expanded tree
identities using the existing HYLE path-hex algorithm and normalized license hashes.
Git commit/tree metadata came from the upstream GitHub API. No peer checkout was reset.
Source cache and package projection remain the existing mechanisms.

[metal-cpp-parity.json](metal-cpp-parity.json) compares all 136 public .hpp files from
the [official ZIP required by MLX](https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip)
against the [official Git release](https://github.com/apple/metal-cpp/tree/f567ed836e4cbb85788c42115a2682bbe68097ee).
Each file is byte-identical. Non-header repository metadata is not claimed ZIP-identical.

Bounded inspection at the MLX pin established:

- `compile.cpp` owns a thread-local compile cache; Metal encoders are thread-local.
- `array.cpp` availability/detach is not a substitute for checked stream synchronization.
- `backend/metal/device.cpp` synchronization waits and checks the retained native error;
  command-buffer error handling poisons/signals dependent events.
- `transforms.cpp` attempts to drain touched streams after eval failure. HYLE still must
  retain every admitted invocation/result through its own terminal notification.
- CPU/Metal/CUDA are separate. Nested downloads must be authenticated before vendor
  configure, not permitted through an outer route flag alone.
- MLX_METAL_JIT embeds source and avoids a relocatable metallib lookup requirement;
  its actual compile/warmup cost belongs to cold prepare and will be measured.

The previous [Q4 native probe](../../correctness/compute-apple-route-review/README.md)
is reused only for the proven native format/lifetime premise. Source-built product,
fixed-thread completion, cancellation/close, multiple callers, packaging and P1 are pending.
