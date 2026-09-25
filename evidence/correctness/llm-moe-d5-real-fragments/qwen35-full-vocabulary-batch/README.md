# D5 Qwen3.5-2B full-vocabulary BF16 batch checkpoint

Immutable tests-only candidate `ae54c66cead8a9385267b4fb88f5e199bf599be1`
extends the common full-vocabulary consumer to the pinned official
`Qwen/Qwen3.5-2B@15852e8c` BF16 profile. The head retains hidden width
2048 and traverses all 248,320 logical rows in 16,384-row shards. Each of
two changed activations produces a complete global logit vector checked
against the independent row oracle. Gather tests selected global IDs in
early, middle and tail shards. The synthetic sparse BF16 payload is not
checkpoint bytes; the route executes its normal mature Provider call.

The new Qwen3.5-2B profile passed 16/16 targeted checks: four routes ×
Direct/Graph × head/Gather. Windows CPU, CUDA and Vulkan used existing
isolated build trees and the Windows RTX 4070 for GPU checks. The MacBook Air
M4 verification clone was clean, fetched the exact candidate, incrementally
built Apple Direct and Graph, and passed 4/4 targeted checks. CPU and Apple
trees had WERROR off; CUDA and Vulkan had WERROR on. No product code or
Provider math changed.

This is a batch wiring checkpoint. Full owner suites, the complete Windows
selector, external source/static/shared delivery and applicable P1 were
deliberately deferred until a complete profile consumer is stable, as the
[D5 owner card](../../../../docs/implementation/tasks/compute/llm-moe-d5-real-fragment-acceptance.md)
requires. They are not PASS. Global out-of-range Gather rejection and exact
quantized artifact vocabulary formats remain unqualified. No profile cell
changes status: 0/176 `RUN_PASS`, 176/176 `VALIDATION_NOT_RUN`. The canonical
Compute + Graph implementation count is 58,043 lines in 228 files, excluding
tests. See [receipt.json](receipt.json).
