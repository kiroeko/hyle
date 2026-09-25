# Qwen3.8-27B CPU GQA admission audit

Windows canonical baseline `a7ea8b91354143f358743dc84eb9bcfc5842a322`
was audited with the fixed official
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
profile and its exact full-attention geometry: 24 query heads, 4 key/value
heads, head depth 256, partial interleaved rotary width 64, BF16 storage and
F32 accumulation.

The ordinary CPU Direct consumer prepared RMSNorm and RoPE successfully, then
rejected SDPA before execution with:

```text
scaled-dot-product-attention: hyle-native: native SDPA requires equal Q/K/V heads
```

The same executable passed the existing Qwen3.5 8/2x256 GQA consumer. Source
inspection explains the model-specific split: `ggml_cpu_provider.cc` admits
SDPA only when `q[2] == 8 && k[2] == 2`, while the immediately preceding fixed
ggml lowering uses the normal `ggml_mul_mat` batch broadcast for Q/K and V/P
and has no 8/2-only algorithm. Consequently Qwen3.8 falls through to the
equal-head native Provider even though its 24/4 ratio is a valid grouped-query
shape for the same mature ggml program.

This is a Provider admission defect, not missing mathematics. D5-F12 limits
the repair to the existing dense BF16/F16/F32, causal, unmasked, F32-accumulated
SDPA domain: require positive query and KV head counts, equal Q/K/V depth,
`query_heads % kv_heads == 0`, matching K/V tensors and the existing output
shape. It adds 8/2 and 24/4 Provider regression cases plus the ordinary
Qwen3.8 Direct/Graph profile consumers. It does not change the ggml program,
public operation schema, route order, native Provider, algorithm, or any HYLE
kernel.

After that admission boundary was removed, the same real fragment exposed one
adjacent output-gate omission: `attn_output_gate=true` requires Sigmoid followed
by Multiply. Fixed ggml already exposes `ggml_sigmoid`, and its CPU backend uses
the standard `1 / (1 + exp(-x))` implementation, including the required finite,
infinite and NaN behavior. HYLE's ggml unary lowering and admission switch had
simply omitted `UnaryKind::kSigmoid`, so Direct fell through to the frozen
native implementation. The same finite repair therefore maps only Sigmoid to
that mature upstream primitive and adds BF16/special-value coverage; no other
unary kind is opened.
