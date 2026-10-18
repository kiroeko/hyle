# D5 CUDA Qwen3.5-2B official BF16 R03 partial receipt

Candidate `67948310b6eed656e1d84e8dee31b1346fdfb301` runs one complete checkpoint tensor through ordinary CUDA Direct and Graph entry points on an RTX 4070 (SM 8.9, driver 610.62, CUDA Toolkit 13.3.73).

The first run exposed a real Provider seam defect: `ggml CUDA` deliberately leaves ordinary rank-2 dense MatMul to cuBLASLt, while HYLE's cuBLASLt adapter admitted FP16/FP32 only. The bounded fix maps BF16 storage to `CUDA_R_16BF` with FP32 accumulation in the existing mature Provider. It adds no HYLE math, kernel, or runtime search.

The fixture copies seven complete tensors (**41,996,384 bytes**) from the user-supplied, previously verified `Qwen/Qwen3.5-2B@15852e8` safetensors file. The executed projection uses the full BF16 `[2048,2048]` `layers.0.linear_attn.out_proj.weight`; Direct and Graph both bind cuBLASLt 13.6.0 and reuse the prepared artifact for the original and sign-flipped activations. Both outputs match the independent F32-accumulation oracle after BF16 rounding.

- cuBLASLt Provider suite: 6/6 PASS, including BF16 row/column layouts and independent lanes.
- CUDA Direct: targeted immutable-candidate run PASS, reported 176 ms; full owner suite 10/10 PASS.
- CUDA Graph: targeted immutable-candidate run PASS, reported 181 ms; full owner suite 29/29 PASS.
- Compute and Graph source/shared/static/relocated consumers: PASS.
- Exact model/artifact profile and versioned R03 identity survive prepare/compile.
- No model bytes are committed; the bounded fixture remains external scratch.

This is a **partial R03 receipt**. The two 176-cell ledger entries remain `VALIDATION_NOT_RUN` until every required Qwen3.5 official region has a receipt. It does not claim P1, other routes, or full-model inference. Machine-readable details and tensor hashes are in [receipt.json](receipt.json).
