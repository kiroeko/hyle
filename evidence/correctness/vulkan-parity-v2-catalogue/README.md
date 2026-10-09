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

## GGUF and current state outcomes

`08764fc4` Direct560/560 and Graph560/560 PASS: all559 GGUF numerical
contracts and a test covering all104 exact identities, original codecs and
actual K/N/head windows/expert windows. No validation-layer errors.

Current state `9ec5c4ee`: Direct16/24 PASS and Graph15/24 PASS; eight/nine
remaining failures are numerical, after the noncausal SDPA and BF16 matvec
admission defects were corrected. The Qwen single-query case was comparing
the actual F32 matvec program against an unconditional F16 coopmat2 oracle.
`d0c6138c` removes only that inapplicable alternate and passes both entrances
with the stricter existing nominal F32 gate. Current totals are Direct17/24
and Graph16/24, with seven/eight unresolved cases, not all-state PASS.

The unresolved cases are retained in `state-numeric-remainder.json` and raw
receipts: Flash mixed three-stage/head, current Pro/Flash attention, official
main/compression and two current GGUF main consumers; Graph additionally has
a complete Pro DSpark numerical failure. Qwen named MTP, exact convolution
state, Qwen norm/carry/NextN, target/confidence/Markov and several DSpark/main
paths did pass their individual tests. These are individual qualifications,
not proof that every current model/state branch is complete.

Next bounded action: capture the first divergent attention/rounding boundary
against the exact original native leaf and the independent source equation.
Use a shared cause to fix only affected current consumers; preserve source
precision and gates. Do not repeat the passed80/560 matrices or all24 state
cases, and do not infer a performance or environment verdict from these errors.
