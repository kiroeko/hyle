# GLM remaining fragments: four-route checkpoint

The [receipt](receipt.json) records **six GLM consumer groups passing in each of
eight ordinary Direct/Graph entries (48/48), with no skips**. This is partial
profile coverage: complete-profile accounting remains96/176. Full MTP, remaining
official FP8 domains, final owner suites, three deliveries, P1 and D6 remain open.

| Route | Qualified candidate | GLM groups, Direct / Graph |
|---|---|---|
| CPU | `264e8633` | 6 / 6 |
| CUDA | `7b0fb4f1` | 6 / 6 |
| Apple | `5c738eaa` | 6 / 6, plus6 Kimi regressions per entry |
| Vulkan | `264e8633` | 6 / 6 |

Later CUDA/Apple evidence reuse is limited to unchanged products, declared
programs, inputs and original reference formulas. Subsequent consumer changes
add Vulkan-specific I32 readback and native multiplication checks. The receipt
retains each candidate, raw-log hashes, prior failures and batch drivers.

## What ran

- Four GGUF dense FFNs retain H6144/4096, intermediate12288, native Q8 weights,
  model-specific gate clamp and residual, with three changing inputs.
- Six NextN input boundaries retain separate embedding/hidden norms,
  embedding-first concatenation and BF16/Q8 EH projection. This does **not**
  qualify the decoder, final head or carried state of a complete MTP step.
- Four GGUF absorbed MLA chains retain64 heads, Q-rank2048/1536, KV-rank512,
  V256, ordinary GLM interleaved RoPE and Flash no-RoPE. Four replays feed the
  actual route cache forward; the reference evolves independently.
- Ordinary DSA keeps32x128, top2048 and more than2052 cache entries. Flash
  keeps513 complete learned K4 pools, selects512, expands2048 token indices,
  then appends0..3 tails and explicit-1 padding. Member sets are checked exactly.
- Flash mHC includes its real fn projection, learned scale/base,20 Sinkhorn
  iterations and four output-feedback steps; final mean is unweighted.

[All-layer role facts](../../../../tests/compute_models/d5_glm_remaining_facts.json)
retain exact pinned official/GGUF shapes and per-role storage. Payloads are
synthetic nonzero values, not downloaded giant checkpoint contents.

## Product fixes and failure attribution

F30 admits64 query heads in the existing ggml CPU/CUDA/Vulkan F32 QK576/V512
attention domain. Existing rank, sequence, head, output and resource checks
remain. CPU provider4/4, CUDA provider9/9 and Vulkan provider13/13 checks pass.

F31 admits dense plain I32 Concat in ggml CUDA. The built runtime accepts I32
CPY/CONCAT but declines DUP; the initial lowering's first-part DUP caused a
cold invocation rejection. It now uses original CPY, including one-input
ownership. Large integers, INT_MIN/MAX, -1, replay and rejected domains pass;
the final three CUDA tensor checks at`f373efce` are retained separately.

Vulkan's initial integer failure was a test readback that interpreted I32 words
as F32. After correcting it, the two remaining issues were:

1. Two permute/Copy calls were incorrectly required to have standalone mature
   arithmetic dispatches. D4 retains this byte-addressed layout adapter. The
   corrected check only allows the same-shape F32 layout copy and still demands
   ggml/IREE for mathematics. Both Softmax materializations and diagnostic output
   copies remain part of consumer cost.
2. Original ggml Vulkan coopmat2 converts F32 MatMul operands to F16 despite F32
   accumulation. The official Flash score now passes both the pre-existing full
   F32 accuracy gate(abs2e-4/rel2e-3) and a strict independent rounded-product
   oracle(abs8e-5/rel4e-5). No measured-output oracle or fitted tolerance is used;
   pool/cache/norm remain strict and selected members remain exact.

Both the initial numerical failures and passing correction are retained.
No HYLE mathematics, kernel, shader, public API or runtime pin was added.

## Validation limits

The selected plan is recorded with118 skipped/reused gates, but this targeted
checkpoint does not claim that all selected owner/delivery gates have run.
Shared input tests13/13, closure and docs checks pass. Heavy jobs ran serially
in existing trees with build parallelism2. Windows DLLs were staged beside EXEs.

The [unified LOC report](loc.json) counts58,617 implementation lines in232
Compute+Graph files, excluding tests. Remaining full-profile and D5-A5/A6 work
continues under the current task card.
