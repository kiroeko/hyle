# Qwen3.8-2.4T semantic roles and checked grouped IDs

Product/test candidate `9e7c579282030db876dac98ca20b6e20e189ce70` covers both
fixed GGUF profiles through ordinary Direct and Graph entries on all four
routes. The shared recipe checks nine attention/GDN projection roles at their
real widths and stored formats, two causal attention positions with Q64/KV4
and depth 256, partial RoPE 64, the Q8_0 MTP 16384-to-8192 projection, and
the K16/V128 GDN state primitive plus 20480-channel K4 convolution.
Synthetic payloads retain full physical matrices; these are not downloaded
2.4T checkpoint weights or whole-model inference results.

Both profiles run changed-input replays. The complete official and quantized
512-way/top-10 MoE recipes also run in these owner suites. Grouped expert IDs
`-1` and `E` now fail through checked boundaries on CPU, CUDA, Metal and Vulkan,
retain the prior published result, and permit a fresh invocation of the same
prepared artifact. No HYLE operator math or shader was added.

## Bounded fixes and findings

- CPU, CUDA and Metal extend their existing index checks to grouped expert
  IDs. Vulkan uses its existing index-check and checked-publication shaders
  around the native ggml region. The original CPU invalid-ID crash is retained
  in the initial logs.
- CUDA admits F32 attention to its existing F32 QK/softmax/PV lowering; its
  previous F16 and BF16 suites continue to pass.
- This pinned Vulkan matrix path rounds operands to F16 while accumulating
  in F32. Attention outputs retain both a full-F32 accuracy check and a separate
  independently rounded reference. The allowance applies only to attention
  products/output gates, not normalization, RoPE or GDN. See D5-F23 in the card.
- Apple/Vulkan Direct chains reuse Programs and create independent invocations
  for dependent calls, as required by retained-result leases. Graph reuses its
  invocation. This receipt does not claim allocation-free Direct invocation
  creation or sign the later complete-cost performance gate.

## Selected checks

The retained Windows and Apple selector plans preserve skipped reasons.
Selected owners/consumers were executed manually in reused route-specific
trees, at Release with warnings as errors. These files are actual command,
cache, CTest and raw-output receipts; they are not a fabricated selector-run
verdict. Windows used MSVC 19.50, CUDA Toolkit 13.3, Vulkan SDK 1.4.357.0 and
RTX 4070 driver **617.14**. Apple used MacBook Air Mac16,12, macOS 27.0
(26A428), AppleClang and the configured native MLX/ggml Metal providers.

| Route | Complete owner/consumer result |
|---|---|
| CPU | ggml Provider 41/41, Direct 80/80, Graph 95/95, executor CTest passed |
| CUDA | ggml Provider 28/28, Graph Provider 8/8, Direct 79/79, Graph 96/96 |
| Apple | ggml Metal Provider 7/7, Direct 109/109, Graph 82/82 |
| Vulkan | Graph 104/104; Direct initially 191/193, then the two corrected tests passed 2/2 |

There were no skipped owner cases. Vulkan's two failures were obsolete
assertions that the last dispatch itself was the managed region. Checked
publication now follows that region. Tests-only candidate `32772681b` requires
both checked status and a retained managed region, and both previously blocked
numeric/replay tests passed. The other 191 Direct cases and complete Graph
result are reused because product code did not change. The initial CTest
failure remains recorded and is not relabeled as a successful full rerun.

Apple's initial targeted batch used a cache with `HYLE_WERROR=OFF`. Its final
configure explicitly enables WERROR, rebuilds the affected owners, and runs
all three suites. Only that final build signs warnings-as-errors acceptance.
Shared input tooling passed 13 tests; closure (`--previous-ref 0e4bb7ed`), docs,
ledger structure and source diff checks passed.

## Scope still open

This remains a partial profile receipt. The official BF16 projection roles
and connected convolution/gate/normalization/GDN state boundary still need
their own completed batch, including official versus converted-GGUF head
ordering. The ledger remains **48 RUN_PASS / 128 VALIDATION_NOT_RUN**.
D5-A5 source/static/shared external delivery and D5-A6 P1 are unrun.

The canonical implementation count is **58,117 lines / 228 files**, excluding
tests. See [receipt.json](receipt.json), the retained raw logs and [LOC](loc.json).
