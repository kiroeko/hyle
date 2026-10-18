# Qwen3.5 fidelity GGUF complete CPU acceptance

Immutable tests-only candidate `74101b6e2216dfc3dd9fd6879ef942fcad960330`
completes the ordinary CPU Direct and Graph cells for
`qwen35_2b:fidelity_gguf:UD-Q8_K_XL`.

- The fixed artifact is treated as a mixed-format tensor inventory rather than
  as globally Q8. Its embedding is F16, representative FFN and attention
  projections are Q8_0 or F16, and convolution/state/norm tensors remain F32.
- R01, R02, R03, R04, R05, R14 and R15 run at the exact Qwen3.5-2B width and
  head geometry. Direct and Graph prepare once, replay changed inputs, compare
  with independent packed-weight or dense-half oracles, and preserve the last
  published embedding output after an invalid index.
- Every prepared operation and Graph compile report fixes a ggml CPU or oneDNN
  implementation. The candidate adds no HYLE math, fallback, provider or
  hot-path selection.
- R13 is inapplicable to this fixed 320-tensor artifact because its audited
  header and repository tree contain no MTP/NextN tensor or sidecar.

The immutable Windows selector passed docs, closure, the model-input tool,
Release/WERROR configure, CPU Direct and CPU Graph; 127 unrelated gates were
skipped or reused. The first selector invocation lacked the Visual Studio
resource-tool environment and stopped during compiler configure; the same
candidate passed after loading `VsDevCmd`, with no source change. D5-A5
source/static/shared delivery and D5-A6 P1 remain route-batch work and are not
claimed here. Machine-readable details are in [receipt.json](receipt.json).

The uniform implementation-only report remains **57,920 lines across 228
Compute+Graph files**; this package changes tests and evidence only. See
[compute-graph-loc.json](compute-graph-loc.json).
