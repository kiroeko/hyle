# Kimi K3: three profiles, four ordinary routes

The [receipt](receipt.json) qualifies all required fragments of the fixed official
MXFP4/BF16, primary UD-Q4_K_XL and fidelity UD-Q8_K_XL profiles through CPU,
CUDA, Apple and Vulkan Direct/Graph. **24 complete cells are added: the ledger
now has 96 RUN_PASS and 80 VALIDATION_NOT_RUN.** This is correctness and
lifecycle coverage; final source/static/shared delivery, P1 and D6 remain open.

## Executed checks

The new batch passed 6 groups in each of eight entries (48/48), followed by the
complete affected owner suites on candidate `9d8d90e4`. Numbers below are
GTest counts in the retained CTest log order; additional non-GTest contracts
are recorded in each owner receipt. All current runs are failure/skip free.

| Route | GTest suite counts | Result |
|---|---|---|
| cpu | 42, 125, 140 | PASS, zero skip |
| cuda | 127, 142, 29, 8 | PASS, zero skip |
| vulkan | 239, 149, 24, 7 | PASS, zero skip |
| apple | 154, 127, 7 | PASS, zero skip |

F29 only fixes the ggml CPU/CUDA/Vulkan cold admission of compressed MLA:
Q/K depth576, V depth512, query heads96 and one cached KV head. Native
MatMul/Softmax/MatMul lowering already supported these dimensions. Wrong
K/V sequence, head, batch, output or QK depth is rejected; unrelated asymmetric
heads and BF16 descriptions remain outside this narrow domain. Original
attention, independent invocation, lifetime and allocation-failure cases pass.

## Real model semantics and weights

- Dense FFN keeps H7168 and intermediate33792, three native weight matrices,
  original SiTU, residual add and final RMS norm (epsilon1e-5).
- MLA keeps Q-rank1536, KV-rank512, K-B/V-B per-head weights, no RoPE transform,
  original scale1/sqrt(192), sigmoid output gate and output projection. Four
  replays feed the actual compressed cache forward; the oracle advances alone.
- Attention Residual scores normalized values but mixes raw values. Eight
  replays cover empty bank, checkpoint append/prefix restart, ordinary residual
  add and final mixing; retained bank bits are checked exactly. Fixed-capacity
  Concat+checked Gather replaces the erroneous assignment-Scatter fixture.
  It materializes9*7168 F32 values then selects8*7168; these extra consumer
  copies are disclosed, not claimed as a model-specific performance win.
- R01 full163840-row vocabulary/head, KDA, routing and mixed expert/shared MoE
  are covered by retained region receipts and rerun in complete owner suites.
  The stale ledger label3584 is corrected: latent3584, expert FFN3072,
  shared FFN6144 and dense FFN33792 are distinct dimensions.

[All-layer header audit](all-layer-roles.json) confirms20/18/18 remaining-role
signatures; [original converter checks](converter-golden.json) verify full-width
KV-B BF16 split/transpose and all three F32 residual-score fusions exactly.
Large-model payloads are format-valid synthetic nonzero data. These tests do
not claim Kimi checkpoint content validation or whole-model inference.

## Numerical qualification corrections

The initial CPU F32 gate failure is retained. A 5.96e-8 norm difference crosses
the BF16 midpoint -0.748046875, yielding adjacent BF16 codes. SiTU also crosses
a BF16 midpoint before the down projection. Only the two new BF16 full chains
were reclassified to the pre-existing BF16 multi-projection rule:
`abs_error <= 2e-5 + 0.025*abs(reference)`. This threshold was not fitted to
observed errors. F32 input norm keeps its strict rule; BF16 conversion midpoints,
their adjacent F32 values, and residual bank storage use exact comparison.
The oracle never takes actual route outputs as expected values.

Apple initially failed because the independent grouped projection oracle
incorrectly rounded F32 activations to BF16. The executed pinned Metal kernel
`kernel_mul_mv_id_bf16_f32_4` explicitly retains F32 input. The corrected Apple
oracle reflects that native boundary; recipe, inputs, providers and thresholds
are unchanged. Both the failed and corrected runs are retained. Other routes
retain their independently modeled native BF16 dot-product boundary.

## Scope and validation accounting

Region links, immutable candidates, raw-log hashes, reused environment identity,
current cache identities and manual batch drivers are in the receipt. This is
not an automatic selector-run PASS. Pinned dependencies, public API and math
are unchanged; the selected factor suites run serially in existing builds.
Final D5 delivery/performance work and the other10 profiles remain unverified.

The unified [LOC report](loc.json) counts58,606 implementation lines in232
Compute+Graph files, excluding tests. No HYLE operator mathematics was added.
