# GLM: four GGUF profiles, four ordinary routes

The [receipt](receipt.json) qualifies GLM-5.3 and GLM-5.3-Flash primary
UD-Q4_K_XL and fidelity Q8_0 through CPU, CUDA, Apple and Vulkan Direct/Graph.
**32 complete correctness cells are added: 128 RUN_PASS,48 VALIDATION_NOT_RUN.**
Official GLM FP8, four DeepSeek profiles, final source/static/shared deliveries,
P1 and D6 remain open. This is not whole-model inference qualification.

## Executed checks

All affected route/provider suites passed without failures or skips. Counts
below are GTest cases in retained CTest log order; selected executor, source
and MASM contracts also passed.

| Route | GTest suite counts | Result |
|---|---|---|
| CPU | 42,134,149; incremental1,1 | PASS |
| CUDA | 137,152,30,8 | PASS |
| Apple | 164,137,7 | PASS |
| Vulkan | 249,159,24,7 | PASS |

CPU's full suite ran on immutable `b52ff8d1`. Final candidate `7af6eae4` only
adds an existing Q6_K enum mapping and four rare-format MoE recipes/tests;
existing products, programs, inputs and error gates are unchanged. Both CPU
entries passed that incremental test; the full suite was not repeated.
Other routes ran their complete suites at the final candidate. Existing
Release/WERROR builds were reused with `-j2` builds and serial heavy runs.
Apple used the registered MacBook Air; Windows DLLs stayed beside EXEs.

## Complete region boundaries

- All154880 output logits consume the actual normalized NextN decoder hidden
  state for four changing-input replays. The independent oracle advances its
  own decoder trajectory. EH, absorbed MLA, routed/shared MoE, residuals,
  norms and actual hidden/cache feedback remain connected.
- Two independently verified indexer contexts generate distinct I32 seeds.
  Four native replays exercise initial seed, actual state reuse, reset and
  second-context reuse. Reuse inputs deliberately contain invalid placeholders;
  only preceding output makes them valid. DSA indices feed checked compressed
  cache Gather and64-head attention; Flash retains tail/padding checks.
- The [all-layer expert audit](../../../../tests/compute_models/d5_glm_expert_variants.json)
  adds Q4_K/Q4_K/Q6_K and Q5_K/Q5_K/Q6_K gate/up/down combinations for both
  primary profiles. Each uses the existing complete MoE with three changing
  inputs. Fidelity remains all Q8_0. Representative Q4/Q4/Q5 does not substitute
  for rare layers.
- Full vocabulary, dense FFN, role-specific projections, KDA, MLA, DSA/Flash
  pool-tail, weighted mHC and shared/routed MoE retain their region receipts
  and pass in the full suites. Every required region has a receipt mapping.

Stale ledger labels are corrected to dense FFN12288, Q-rank2048/1536,
KV-rank512,64 heads, QK256/V256, RoPE64/0, compressed QK576/512 and V512.
[NextN facts](../../../../tests/compute_models/d5_glm_mtp_facts.json) confirm
neither GGUF NextN layer has mHC parameters; trunk mHC is not inserted there.

## Numerical qualification and costs

Initial CPU candidate `c0a5ffae` wrongly applied the single-MatMul absolute
gate to accumulated quantized decoder error. Both failed logs are retained.
The correction applies two previously fixed gates: head output relative to
its actual activation retains max0.008/RMSE0.002; complete logits relative to
the independently advanced decoder use the existing normalized quantized-chain
`expect_output` gate. No threshold was added or fitted. Actual outputs never
replace the whole-chain oracle; exact index/state checks remain exact.

The correctness consumer reads back and uploads actual hidden at the
decoder/head boundary to stream all vocabulary shards in bounded memory.
Index carry policy and diagnostic state capture use explicit host control;
diagnostic copies/sorting remain disclosed. This is not a zero-copy,
optimal-fusion or performance claim. Payloads are format-valid nonzero synthetic
data tied to pinned headers, not giant-checkpoint content tests.

## Evidence and limits

The receipt records immutable candidates, cache identities, compressed logs
and hashes, first failures, selected factors, drivers and reused regions.
These are manual factor results, not an automatic selector-run PASS. This
batch changes test consumers and facts only; prior F30/F31 fixes are retained.
The [unified LOC report](loc.json) counts58,617 implementation lines in232
Compute+Graph files, excluding tests. No HYLE mathematical kernel was added.
Official FP8, DeepSeek, final delivery/performance gates and D6 are unqualified.
