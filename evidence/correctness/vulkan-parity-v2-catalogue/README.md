# Vulkan actual catalogue and state batch

[Current bounded scope](../../../docs/implementation/tasks/compute/vulkan-parity-v2-catalogue-matrix.md)
uses the shared frozen 559/80/34/45/104 demands and model/state/sidecar programs.
This first executable batch adds the 34 non-GGUF packed and 45 plain cases to
both ordinary entrances, including actual N/K, source dtypes, cold native
preparation feeding resident constants, repeat inputs and checked IDs. Common
head windows and explicit expert windows are reported as such; full giant
expert-bank residency is not inferred. The remaining GGUF/parameter/current
branch matrix remains pending until its corresponding evidence exists.

Two old negative official MoE tests are replaced with complete original
program execution. Product changes bind the original grouped F32 accumulator
and compose low-precision convolution with original native casts and ggml
SSM convolution. A concrete N=17408 cold-bound defect is corrected using the
locked original MAT tile32 and split-K reduce tile1024; buffer/workspace limits
remain. No new HYLE math, GLSL, quantization algorithm or performance claim.

Build diagnostics remain in this directory. Native logs/XML/receipts must
record actual outcomes; no unrun case is PASS. WSL/Air, deliveries, defaults
and formal P1 remain parent tasks. Document updates never rebuild or rerun P1.

## First actual Windows batch result

Product `6ed2cdefa`: Direct80/80 and Graph80/80 PASS, no validation-layer errors.
This covers all34 packed, all45 plain and the existing original four official
MoE full programs. Actual N/K, common head row windows, declared expert windows,
native cold preparation feeding GPU constants, two input replays, state/rounding
checks and bad-index retained-result checks are exercised. The original four
MoE profiles do not stand in for newer Pro0813/Flash0731/DSpark qualification.
No unrun559/80/104 or new state/sidecar/platform/delivery/P1 item is PASS.
