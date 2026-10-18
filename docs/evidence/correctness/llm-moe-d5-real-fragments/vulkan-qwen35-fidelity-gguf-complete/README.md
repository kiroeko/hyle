# Qwen3.5 fidelity GGUF complete Vulkan acceptance

Immutable tests-only candidate `123c87c1082f99c03ceed4be954231e520f8357f`
completes the ordinary Vulkan Direct and Graph cells for
`qwen35_2b:fidelity_gguf:UD-Q8_K_XL` on the Windows RTX 4070 route.

- The fixed artifact remains a mixed-format tensor inventory: F16 embedding,
  Q8_0 FFN gate/up/down and attention V/output, F16 Gated DeltaNet output and
  attention K/Q, plus F32 convolution/state/norm tensors.
- R01 uses admitted ggml Vulkan F16 row selection with changed-token replay and
  checked failure without publication. R02, R03 and R14 preserve exact Q8_0 or
  F16 weights at the Qwen3.5-2B dimensions and execute fixed mature Vulkan
  regions.
- R04 and R05 retain the exact 8-query-head/2-KV-head x 256 GQA, K4 causal
  convolution and 16 x 128 Gated DeltaNet semantics. Direct and Graph replay
  changed inputs and compare with independent packed-weight, dense-half or
  fragment oracles.
- Every prepared operation and Graph compile report fixes an admitted ggml
  Vulkan region or limited IREE Vulkan dispatch. The candidate adds no HYLE
  math, fallback, provider or hot-path selection.
- R13 is inapplicable to the fixed 320-tensor artifact because its audited
  header and repository tree contain no MTP/NextN tensor or sidecar.

The immutable Windows selector passed docs, closure, Release/WERROR configure,
Vulkan Direct (1/1 CTest executable, 30.89 s; 142/142 tests) and Vulkan Graph
(1/1, 15.48 s; 53/53 tests); 127 unrelated gates were skipped or reused. Before
that formal run, the six focused fidelity/semantic tests in each entry and both
full owner suites passed. D5-A5 source/static/shared delivery and D5-A6 P1
remain route-batch work and are not claimed here. Machine-readable details are
in [receipt.json](receipt.json).

The uniform immutable-candidate report remains **57,920 lines across 228
Compute+Graph implementation files**, excluding tests. See
[compute-graph-loc.json](compute-graph-loc.json).
