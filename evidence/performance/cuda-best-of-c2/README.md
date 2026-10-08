# CUDA C2 candidate qualification

206 frozen typed requests across five actual Providers: 1030 observations,408 CORRECT with complete changed-input outputs,622 explicit declines, zero numerical failures and zero requests without a carrier.192 requests have multiple carriers and14 a single carrier.

All212 actual pair programs independently re-instantiated the native programs with two resident invocation/input/output slots per side, ran both changed payloads, and passed the same full-output oracle. Exact pair-name union checked. No performance winner or AA/P1 qualification is claimed here.

`qualification.zip` preserves all commands, source/SDK inputs, logs and records. Upstream stderr warmup messages interleaved into buffered JSON; only that exact known text was removed while parsing. Original merged log remains unchanged;206 request names and all five unique Provider observations per request prove no rows were lost. `admission-rows.json` and `qualification.json` retain results and archive identity.

Plain Gather is CCCL-only here: ggml public Gather binds its admitted encoded embedding tables, so generic plain F32 Gather is not a second carrier. Quantized embeddings remain their existing single ggml carrier; no artificial conversion for a benchmark.
