# CUDA C2 candidate qualification

206 frozen typed requests across five actual Providers: 1030 observations,408 CORRECT with complete changed-input outputs,622 explicit declines, zero numerical failures and zero requests without a carrier.192 requests have multiple carriers and14 a single carrier.

All212 actual pair programs independently re-instantiated the native programs with two resident invocation/input/output slots per side, ran both changed payloads, and passed the same full-output oracle. Exact pair-name union checked.

`qualification.zip` preserves all commands, source/SDK inputs, logs and records. Upstream stderr warmup messages interleaved into buffered JSON; only that exact known text was removed while parsing. Original merged log remains unchanged;206 request names and all five unique Provider observations per request prove no rows were lost. `admission-rows.json` and `qualification.json` retain results and archive identity.

Plain Gather is CCCL-only here: ggml public Gather binds its admitted encoded embedding tables, so generic plain F32 Gather is not a second carrier. Quantized embeddings remain their existing single ggml carrier; no artificial conversion for a benchmark.

## Bounded performance result

The one eight-process A/A batch admitted47 of212 pairs. The other165 remain unqualified for a measured winner; their correctness results remain valid. No extra A/A, n escalation or retuned shapes follow. The original212 rows, all raw logs and individual decisions are in `aa-batch.zip` and `aa-summary.json`.

The exact qualified image then completed the preregistered n15/four-strata batch. Instrument qualification and the canonical manifest verification passed. Individual combined verdicts are25 WIN,2 LOSE and20 INCONCLUSIVE; the aggregate performance gate remains **FAIL** because the batch deliberately included losing candidates. Noninferiority results are35 PASS,2 FAIL and10 UNRESOLVED. `pair-verdicts.json`, `p1-receipt.json` and `p1-raw.zip` preserve these results without changing the gate.

Complete same-request candidate coverage yields23 qualified native choices,169 requests using engineering defaults and14 single-carrier requests. Two additional native weighted-RMS pair wins are partial: the third candidate was not qualified, so neither becomes a complete winner rule. `complete-provider-choices.json` records all206 requests and the actual candidate sets. The measured choices apply only to the exact Windows/static Release/i9 caller2 parent0x14/RTX4070 SM89/driver617.14/toolkit13.3/runtime13.4 recipe and original Provider versions. They establish no unrestricted-host, other-device, other-linkage or complete-region WIN.
