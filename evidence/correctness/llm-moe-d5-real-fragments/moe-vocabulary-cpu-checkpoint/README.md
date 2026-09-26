# Remaining MoE profiles: CPU vocabulary checkpoint

Tests-only candidate `9f88d8b0` adds the other thirteen fixed model profiles
to the existing eight ordinary vocabulary consumers. CPU Direct and Graph
each pass **26 tests** (head and checked gather for every profile). This
checkpoint covers R01 only; it adds no complete-profile ledger cells.

The stored official/GGUF header audit supplies exact tensor names, dtypes,
dimensions and manifest digests in
[profile facts](../../../../tests/compute_models/d5_moe_profile_facts.json).
Kimi retains 163840 rows, DeepSeek 129280 and GLM 154880, with full hidden
widths and per-role BF16/Q8_0 storage. Synthetic nonzero payloads cover every
head row, changed-input replay, global gather boundaries and rejected IDs.
Kimi's vocabulary divides exactly into shards; no zero-row tail is prepared.

CPU validation used the existing Release/WERROR tree, `-j2`, with DLLs beside
the executables. [Receipt and logs](receipt.json) record the actual filtered
run. CUDA/Apple/Vulkan executions, other model regions, external deliveries
and P1 are **unrun**; they remain in the later stable batch. No product or
operator implementation changed. Compute + Graph implementation remains
58,255 lines, excluding tests, as measured in the
[preceding unified report](../qwen-nine-profile-complete-gate/loc.json).
