# D5 Vulkan Qwen3.5-2B official BF16 R03 partial receipt

Candidate `958b9d69fba9b6bc4479aa3fe693cdab2ef119b0` runs one complete checkpoint tensor through ordinary Vulkan Direct and Graph entry points on an RTX 4070 (driver 610.62, Vulkan SDK 1.4.357.0).

The first run exposed a mature Provider seam defect. Pinned ggml Vulkan admitted rank-2 BF16 small MatMul, selected its `mul_mat_vec` pipeline, and then asserted because that pipeline accepts F32, F16, or Q8_1 activation. The bounded adapter fix widens only a 1–4 row BF16 activation with upstream `ggml_cast`, then uses the same upstream BF16-weight MatMul with FP32 accumulation. It adds no HYLE math, shader, fallback, or runtime search.

The fixture copies seven complete tensors (**41,996,384 bytes**) from the user-supplied, previously verified `Qwen/Qwen3.5-2B@15852e8` safetensors file. The executed projection uses the full BF16 `[2048,2048]` `layers.0.linear_attn.out_proj.weight`; Direct and Graph reuse the prepared artifact for the original and sign-flipped activations. Both outputs match the independent F32-accumulation oracle after BF16 rounding.

- ggml Vulkan Provider checks: 3/3 PASS; IREE Vulkan one-hop checks: 2/2 PASS.
- Vulkan Direct: targeted immutable-candidate run PASS, reported 166 ms; rows 1–4 boundary regression PASS.
- Vulkan Graph: targeted immutable-candidate run PASS, reported 161 ms.
- Full Compute Vulkan and Graph Vulkan owner suites: PASS.
- Source, shared, static, and relocated external consumers: PASS; the consumer runs BF16 Direct, Q4_K Direct, and IQ4_XS Graph.
- Exact model/artifact profile and versioned R03 identity survive prepare/compile.
- No model bytes are committed; the bounded fixture remains external scratch.

This is a **partial R03 receipt**. The two 176-cell ledger entries remain `VALIDATION_NOT_RUN` until every required Qwen3.5 official region has a receipt. It does not claim P1, other routes, or full-model inference. Machine-readable details and tensor hashes are in [receipt.json](receipt.json).
