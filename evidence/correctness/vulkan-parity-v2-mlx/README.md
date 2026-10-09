# Vulkan original MLX packed source batch

Card: [finite audited MLX bindings](../../../docs/implementation/tasks/compute/vulkan-parity-v2-mlx-bindings.md).
Product `1ad27d6c8`, final negative-NaN fidelity correction `1df0cc035`;
source identities are in `source-qualification.json` and the offline receipts.
Windows RTX4070/NVIDIA617.14, locked original IREE compiler/runtime6ef606.
DLLs remain adjacent to each EXE. Heavy builds use2 jobs, GPU tests are serial,
validation layer is enabled, child processes are hidden with fault UI disabled.

## Actual results

| Scope | Raw | Result |
|---|---|---|
| Original native IREE owner | `native-1fbd8ac7-provider.log/.xml` | 7/7 PASS |
| Both ordinary MLX entries | `native-43f56232-{direct,graph}-delta.log/.xml`, final `numeric-1df0cc03-{direct,graph}-numeric.log/.xml` | PASS:39 format/kind/dtype/group combinations +6 mixed affine cases per route; input/weight/scale/bias/IDs changes, independent slots/leases, bad-ID failure and fresh-slot recovery |
| Immutable weight banks and changed A/IDs | `native-1fbd8ac7-{direct,graph}.log/.xml` | Both constant/rebinding IDs PASS; native constant upload runs once per slot, mutable inputs remain charged |
| Empty embedding, feature domain, OOM/cancel/replay/close/zero quota accounting | Original Direct log | PASS |
| All256 MX8 encodings ×three dtypes;42 MX scale/dtype boundaries | Final Direct numeric log | PASS under original matrix arithmetic gate, strict NaN/Inf classification |
| Prior artifact/precision shared seams | Original Direct/Graph logs | Necessary affected subset PASS |
| Full16 model geometries, full catalogue regions, Air, three deliveries, final default/P1 | Not run | UNPROVEN |

The final21 programs are borrowed original MLX tensor sequences, lowered and
scheduled by IREE. A loop expands only the selected experts' current K group;
MX dot precedes source-typed scaling and F32 group accumulation. Embedding
selects rows before its exact typed multiply/add sequence. HYLE adds no GLSL,
quantizer, GEMM, universal executor or per-kernel provider table.

Mixed affine uses existing qualified casts. Immutable banks use a separate
first-submission native copy; changed mutable buffers still copy on execution.
Cold copies, private compressed storage, group workspace and all VM/queue/
completion work remain real and charged. Inputs/native metadata are bound
cold; replay performs no provider search or timing decision.

## Failure causes and practical numeric limits

The original batch's two failed IDs requested reuse of an index-failed slot.
Existing Vulkan/Gather contract preserves that slot's terminal error; creating
a new Invocation from the same Program restores execution. Only the fixture
was corrected; original failure logs are retained.

Additional discriminating tests exposed original IREE conversion of negative
E4M3FN NaN0xff into a finite value. The generated program now protects both
reserved signs using the original format's classification, retaining original
normal conversion. Final raw verifies all256 encodings, rather than a NaN-only
matrix that could hide the difference.

The RTX4070 Vulkan float-controls query reports denorm16=1, denorm32=0 and
signed32=1. A source scale0 produces a2^-127 factor; native F32 multiplication
can flush it. An exact CPU result near1.88e-37 is therefore not a promised
bitwise GPU result. MatMul uses the already established floating tensor gate
(F32 `6e-6 relative +2e-6 absolute`, F16 `0.001 +2e-7`, BF16 `0.008 +2e-7`),
with mandatory NaN/Inf classification. Embedding retains exact typed rounding.
No existing ordinary numeric gate or model-chain threshold was weakened.
The initial artificially exact matrix failures remain historical FAIL; final
PASS is explicitly the applicable native arithmetic contract, not IEEE F32
gradual-underflow identity or full model accuracy proof.

All16 full geometries, complete model state/official branches, portability,
delivery and applicable full-cost performance qualification remain with the
parent goal. No new performance WIN is claimed.
