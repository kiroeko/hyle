# Qwen3.5-2B official Apple Graph acceptance

Immutable candidate `e99c2bf929bbee747dbb8a392ab3ff813299be3f`
completes every required region of the official BF16 Qwen3.5-2B Apple Graph
cell through ordinary Graph construction, cold compile, binding, replay, and
checked completion entry points on the formal Apple M4 machine.

The graph consumers cover R01-R05 and R13-R15 with the same real checkpoint
weights and exact widths as Direct. Each one-node graph checks its compile
report, captures immutable weights/constants in the Program, changes dynamic
inputs or state across invocations, and matches the independent oracle. MLX
keeps its native region behavior; Gated DeltaNet remains a whole ggml Metal
region rather than being split into HYLE kernels.

The focused batch passed 5/5 and the complete Apple Graph owner suite passed
25/25 with WERROR. No product code or route policy changed. Final stable-route
delivery and P1 remain D5-A5/A6 work. Machine-readable details are in
[receipt.json](receipt.json).
