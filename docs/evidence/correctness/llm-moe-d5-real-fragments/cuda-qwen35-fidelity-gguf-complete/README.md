# Qwen3.5 fidelity GGUF complete CUDA acceptance

Immutable tests-only candidate `d407d9cc7c3f01cffab5b02e1c613c60bc220b0e`
completes the ordinary CUDA Direct and Graph cells for
`qwen35_2b:fidelity_gguf:UD-Q8_K_XL` on the RTX 4070 / SM89 route.

- The fixed artifact remains a mixed-format tensor inventory: F16 embedding,
  Q8_0 FFN gate/up/down and attention V/output, F16 Gated DeltaNet output and
  attention K/Q, plus F32 convolution/state/norm tensors.
- R01 uses the original CCCL checked Gather, including its declared workspace,
  changed-token replay and failure-without-publication contract. R02, R03 and
  R14 preserve exact Q8_0 or F16 weights at the Qwen3.5-2B dimensions.
- R04 and R05 retain the exact 8-query-head/2-KV-head x 256 GQA, K4 causal
  convolution and 16 x 128 Gated DeltaNet semantics. Direct and Graph replay
  changed inputs and compare with independent packed-weight, dense-half or
  fragment oracles.
- Every prepared operation and Graph compile report fixes a ggml CUDA, cuBLAS,
  SGLang/CCCL native or FlashInfer mature implementation. The candidate adds no
  HYLE math, fallback, provider or hot-path selection.
- R13 is inapplicable to the fixed 320-tensor artifact because its audited
  header and repository tree contain no MTP/NextN tensor or sidecar.

The immutable Windows selector passed docs, closure, Release/WERROR configure,
CUDA Direct (1/1 CTest executable, 16.10 s; 29/29 tests) and CUDA Graph (1/1,
18.13 s; 46/46 tests); 127 unrelated gates were skipped or reused. The formal
build fixed `CMAKE_CUDA_ARCHITECTURES=89-real` through `CUDAARCHS`. D5-A5
source/static/shared delivery and D5-A6 P1 remain route-batch work and are not
claimed here. Machine-readable details are in [receipt.json](receipt.json).

The uniform immutable-candidate report remains **57,920 lines across 228
Compute+Graph implementation files**, excluding tests. See
[compute-graph-loc.json](compute-graph-loc.json).
