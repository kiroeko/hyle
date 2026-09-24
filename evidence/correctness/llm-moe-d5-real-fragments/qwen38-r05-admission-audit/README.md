# D5 Qwen3.8-27B R05 grouped-head admission audit

The fixed official profile requires Gated DeltaNet K16/V48×128. HYLE's public schema accepts V heads that are an
integer multiple of K heads, and pinned ggml models this as `v_repeat`. The four HYLE ggml adapters nevertheless
required equal K/V head counts before building the upstream operation.

The tests-only Windows consumer fixes the official model revision, config digest, SHA-256 of the immutable
`model.safetensors.index.json`, R05 identity, full head counts and head depth. Only batch/tokens are reduced to 1.
It supplies nonzero data and an independent double-precision grouped-head oracle.

On product HEAD `9e6d433b336f`, the CPU provider rejects the exact consumer during cold prepare in 5 ms with
`ggml CPU operation domain is not admitted`. No backend execution occurs. This is an adapter reachability defect,
not evidence that upstream lacks the operation. The four provider owners are therefore returned to AUDITED with one
bounded fix: admit the grouped-head subset already implemented by pinned ggml and qualify each native route.

This is a **failure/audit receipt**, not a D5 PASS. No acceptance-ledger cell changes state.
