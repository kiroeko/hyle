# Qwen3.8-2.4T BF16 full MoE Graph batch (partial D5 receipt)

Immutable Windows product/test candidate `825c41f8d702e5bb8ea40dc8f0ef2bc203ef63d5`
uses the pinned official `Qwen/Qwen3.8-2.4T-A95B@207bd685` BF16 profile.
The shared test fixture keeps H=8192, 512 router experts, top-10 and expert
width 2048. It keeps 11 routed experts resident so replay 29 can select the
near-tie eleventh expert without replacing a prepared weight bank. It uses
format-valid synthetic bytes, not the full 2.4T checkpoint.

One multi-node Graph runs BF16-to-F32 router scores, Softmax, TopK,
renormalization, ID-driven BF16 grouped gate/up/down projections, weighted
combination, independent shared expert with sigmoid gate, and final addition.
The same three changing inputs run through each route. The test compares the
selected IDs, per-slot routed result, combined result, gated shared result and
final result with an independent CPU oracle. Provider reports exclude HYLE
self-written math; metadata-only `View` steps have no Provider identity.

| Route | Machine | Targeted full-chain test | Full Graph owner suite |
|---|---|---|---|
| CPU | Windows / MSVC 19.50 | PASS, 1/1 | PASS, 69.35 s |
| CUDA | Windows / RTX 4070 / MSVC 19.50 | PASS, 1/1 | PASS, 67.58 s |
| Vulkan | Windows / RTX 4070 / MSVC 19.50 | PASS, 1/1 | PASS, 75.73 s |
| Apple | MacBook Air M4 / AppleClang 21 / MLX 0.32.2 / ggml Metal 0.24.0 | PASS, 1/1 | PASS, 46.41 s |

Windows Graph core `hyle_graph_test` and `hyle_graph_executable_test` passed;
the Mac Graph core target passed. CUDA initially failed at submit with
`CUDA buffer offset does not satisfy the requested type alignment`.
The fixed ggml CUDA region had already declared its 256-byte port alignment;
Graph's workspace planner dropped that value when allocating an internal
root exposed by a metadata View. D5-F20 reopens only the Graph owner and
propagates selected port alignment across physical instances and aliases.
The same CUDA full-chain test and full owner suite passed after the fix.
Vulkan's initial failure was a test decoder reading Int32 ID bits as Float32;
the corrected test reads the raw integer bits and then passed.

The Windows change-impact plan from `6f954ef5` selected Graph,
ggml-CUDA/oneDNN Graph providers, and CPU/CUDA/Vulkan Graph route factors;
the full selector has **not completed** on this partial-profile candidate.
Apple was checked in a
disposable Mac verification checkout fetched from the Windows-pushed commit.
No source/static/shared delivery or P1 is signed here. The later
[Direct full-chain receipt](../qwen38-24t-full-router-direct/README.md)
completed the same MoE chain through all four ordinary Direct routes; neither
partial receipt changes any D5 model cell from `VALIDATION_NOT_RUN`. The
unified Compute + Graph implementation count at this
candidate is **58,043 lines / 228 files**, excluding tests, with the LOC
self-check passing.
