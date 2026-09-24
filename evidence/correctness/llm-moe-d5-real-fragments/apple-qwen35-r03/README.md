# D5 Apple Qwen3.5-2B official BF16 R03 partial receipt

Candidate `78561cec7ff729783bcfe71e604eeb10639ef724` runs one complete checkpoint tensor through ordinary Apple Direct and Graph entry points on the formal MacBook Air `Mac16,12` (Apple M4, macOS 27.0, MLX 0.32.2).

The existing MLX Metal MatMul path accepts the exact BF16 decode shape without a product change. The fixture copies seven complete tensors (**41,996,384 bytes**) from the user-supplied, previously verified `Qwen/Qwen3.5-2B@15852e8` safetensors file. The executed projection uses the full BF16 `[2048,2048]` `layers.0.linear_attn.out_proj.weight`; Direct and Graph reuse the prepared artifact for the original and sign-flipped activations. Both outputs match the independent F32-accumulation oracle after BF16 rounding.

- Apple Direct: targeted immutable-candidate run PASS, reported 315 ms; full owner suite PASS in 42.68 s.
- Apple Graph: targeted immutable-candidate run PASS, reported 37 ms; full owner suite PASS in 6.49 s.
- Source, shared, static, and relocated external consumers: PASS; the consumer runs actual BF16 R03 Direct, Q4_K Direct, and IQ4_XS Graph.
- Exact model/artifact profile and versioned R03 identity survive prepare/compile; replay changes activation without rerouting.
- No HYLE math, Metal kernel, Provider fallback, or product adapter change was needed.
- No model bytes are committed; the bounded fixture remains on the disposable validation host.

This is a **partial R03 receipt**. The two 176-cell ledger entries remain `VALIDATION_NOT_RUN` until every required Qwen3.5 official region has a receipt. It does not claim P1, other routes, or full-model inference. Machine-readable details and tensor hashes are in [receipt.json](receipt.json).
