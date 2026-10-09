# Vulkan original official float/QAT source batch

Card: [finite official bindings](../../../docs/implementation/tasks/compute/vulkan-parity-v2-official-float-bindings.md).
Windows RTX4070/NVIDIA617.14; original IREE compiler/runtime6ef606. All children
are hidden with Windows fault UI disabled, EXEs have adjacent DLLs, validation
layer is enabled, build2/test1 and serial real GPU execution. No P1 sampling.

## Actual qualification

| Scope | Raw | Result |
|---|---|---|
| Original IREE owner and affected prior FP8/precision | `native-a265d1be-*` | Owner7/7, two Direct prior IDs and one Graph prior ID PASS |
| Dynamic FP8 normal/power2, F32/BF16, changed A/weights/scales, N5/129 tails; explicit BF16/F32 weight decode | `native-ee9350ab-{direct,graph}-bf16.log/.xml` | Both ordinary IDs PASS |
| Grouped FP8 and official FP4, A=1/S, duplicate/changed/bad IDs | Same final typed boundary raw | Both ordinary IDs PASS |
| Three QAT variants, all65280 finite BF16 encodings, reverse replay and finite failure | `native-959f5286-{direct,graph}-delta.log/.xml` | Both QAT IDs PASS, exact BF16 numeric gate |
| Encoded block128 real N-tail and changed encoded inputs | Same delta raw | Both encoded IDs PASS |
| Independent slots, retained output, cancel/replay, finite failure/new slot, OOM, close and zero quota accounting | `lifecycle-a5ae3ae4-{direct,graph}.log/.xml` | Both lifecycle IDs PASS |
| Full model geometries/catalogue, state/official complete branches, Air, three deliveries, default/full-cost P1 | Not run in this batch | UNPROVEN |

The public13 precision/operation combinations remain available through ten
native original source entries: five QAT/decode entries, normal/power FP8 F32
cores, grouped FP8/FP4 cores and encoded block128. BF16 dynamic linear/expert
programs use existing qualified native widening, original F32 quant/group
program and native BF16 output conversion. Shared A and distinct per-slot A
retain their original bytes/quantization/scale order.

Original sources are registered DeepSeek/SGLang/ds4 recipes. IREE owns native
matrix execution, loops, allocation and completion. Workspace expands only
the selected current K group, not a full float weight/expert bank. Finite
classification returns a native scalar, read after retirement; error prevents
publication. Existing safe IDs/checked copies preserve bad-index behavior.
Constant banks copy once per slot, mutable inputs retain charged GPU copies.
All conversion/storage/copy/flag synchronization costs remain explicit.

## Source and test corrections

One C++ enum namespace typo was fixed by incremental build. The tail fixture
used floor(N/128), leaving zero scale payload while its logical oracle assumed
the missing tail scale; it was corrected to the public packet's ceil(N/128).
The original raw failures remain unchanged.

Original IREE F32→E4M3FN conversion did not qualify the low-scale QAT boundary.
The registered ds4 original binary nearest/even conversion was translated to
standard tensor/arith/scf without changing its127 values, search or ties. That
restored all finite BF16 QAT samples. No new quantizer, HYLE shader or compiler
pass was introduced, and the exact BF16 test gate was not changed.

A same-stored-input capture then isolated the remaining BF16 linear variant:
F32 normal/power programs had max error0 against the independent reference,
while BF16 variants differed by0.015625/0.125488281. The invalid fused BF16
variants were removed. Existing native casts now preserve BF16 input/output
boundaries around the qualified original F32 program. Final two ordinary IDs
were rerun once per route; QAT/encoded success and prior owner tests were reused.

Offline receipts retain exact compiler commands, source/byte identities and
24 original math schedules plus two original integer finite reductions. The
single-lane integer reductions have no floating arithmetic and do not claim
an inapplicable IEEE32 mode. Installed consumers need neither Python nor IREE
compiler. Licenses and notices are included by the existing package mechanism.

This proves the bounded Windows source batch and ordinary entry/lifetime
seams. It does not prove every model geometry, whole model inference, final
default/best-of, portability, delivery or performance WIN.
