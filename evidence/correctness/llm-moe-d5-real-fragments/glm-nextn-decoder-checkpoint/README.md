# GLM NextN decoder and sparse-cache consumer checkpoint

Candidate `3fb5e818` passes **10/10 test groups in each of eight ordinary
Direct/Graph entries: 80/80, with no skips**. The [receipt](receipt.json)
preserves candidate identities, logs, source hashes and the first failed CPU run.
Two groups per entry are new; six GLM groups and two full-MoE groups are regressions.

| Route | Direct | Graph |
|---|---:|---:|
| CPU | 10/10 | 10/10 |
| CUDA | 10/10 | 10/10 |
| Apple, MacBook Air | 10/10 | 10/10 |
| Vulkan | 10/10 | 10/10 |

## New coverage

- Four GLM GGUF profiles execute NextN embedding/hidden normalization, EH,
  compressed MLA, residual, norm, routed/shared MoE, residual and head norm.
  Four changing embeddings use each route's actual previous hidden/cache outputs;
  the oracle advances independently. Real role widths and storage are retained.
- Actual DSA top-2048 IDs drive checked native Gather from the compressed KV
  cache and 64-head Attention. Flash expands 512 pools and includes 0/1/2/3 live
  tail tokens. Index selection never goes through a host readback before Gather.
  Both indexer and compressed caches feed actual previous outputs back.

[Exact NextN tensor facts](../../../../tests/compute_models/d5_glm_mtp_facts.json)
distinguish official mixed FP8/BF16/F32 storage from the four GGUF profiles.
Weights in this batch are nonzero synthetic contents with those exact formats
and shapes; this is not whole-model inference with downloaded giant weights.

## First failure and finite correction

At `247403b1`, both CPU entries passed 7/9 groups and failed the two new groups:

1. The single decode query over selected past/current keys was incorrectly
   declared noncausal. The existing mature-provider domain requires causal
   attention. Right-aligned causal masking leaves all these keys visible, so
   correcting the consumer preserves its mathematics without product changes.
2. Flash Q8's upstream quantization swapped the order of experts 2 and 4, while
   selected membership and final numeric outputs passed. Diagnostics incorrectly
   compared expert weights by position. Original TopK/Gather now pairs diagnostic
   IDs and weights in ID order; actual MoE order is untouched. Exact membership
   and all numeric error gates remain unchanged.

The CPU Graph search-budget failure disappears with the valid SDPA request.
No planner budget, provider admission, product math or tolerance was changed.
Diagnostic sorting, gathering and copies remain explicit test costs; this batch
makes no performance claim. The corrected batch also includes the eight-profile
GGUF MoE regression that the first run's misspelled filter did not select.

## Remaining scope

Complete-profile accounting stays **96/176**. MTP cross-iteration index carry/reset
and connection to the full-vocabulary output head remain open, as do official
GLM FP8 domains, DeepSeek remaining chains, final owner gates, three deliveries,
P1 and D6. This receipt does not qualify any of them.

Shared input tests (13/13), closure and docs checks pass. The retained selector
chooses those checks and records 130 skipped/reused gates. Route-specific batch
tests above are the card's additional targeted correctness checks, not a new
full-owner or delivery verdict. Existing build trees were reused with `-j2` and
serial heavy jobs; Windows subprocesses are hidden and DLLs are beside EXEs.

The [unified implementation count](loc.json) remains **58,617 lines / 232 files**,
excluding tests. No product source changed in this batch.
