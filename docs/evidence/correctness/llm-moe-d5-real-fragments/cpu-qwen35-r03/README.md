# D5 CPU Qwen3.5-2B official BF16 R03 partial receipt

Candidate `c847455ab1bed6dde199fad232a699d51bb097bd` runs one complete checkpoint tensor through ordinary CPU Direct and Graph entry points.
The fixture copies seven complete tensors (**41,996,384 bytes**) from the user-supplied, previously verified
`Qwen/Qwen3.5-2B@15852e8` safetensors file. The executed projection uses the full BF16
`[2048,2048]` `layers.0.linear_attn.out_proj.weight`; HYLE binds ggml CPU and reuses the same prepared
artifact for the original activation and a sign-flipped activation. Both results match the independent F32-accumulation oracle after BF16 rounding.

- Direct targeted run: PASS, reported 13 ms; complete owner suite: 10/10 PASS.
- Graph targeted run: PASS, reported 14 ms; complete owner suite: 23/23 PASS.
- Exact model/artifact profile and versioned R03 identity survive prepare/compile.
- No model bytes are committed; the bounded fixture remains external scratch.

This is a **partial R03 receipt**. The two 176-cell ledger entries remain `VALIDATION_NOT_RUN` until every
required Qwen3.5 official region has a receipt. It does not claim P1, other routes, or full-model inference.
Machine-readable details and tensor hashes are in [receipt.json](receipt.json).
