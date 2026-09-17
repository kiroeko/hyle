# R6 Windows native implementation checkpoint

The two Vulkan owners remain AUDITED under
[V1–V7](../../../docs/implementation/tasks/compute/vulkan-v1-four-route-foundation.md).
This checkpoint executes real RTX 4070 GPU mathematics and resource lifetimes;
it is not a route closure, delivery certificate, or immutable P1 candidate.

MSVC 19.50 Release/WERROR built all eight HYLE product units and six embedded
ggml shader variants, using locked Vulkan-Headers 1.4.357 and glslang 15.2.
CPU/CUDA/Apple implementations were outside the selected factor closure. Build
parallelism was two and CTest ran serially. Khronos validation 1.4.357 was explicitly
enabled, including synchronization validation; its DLL was placed beside the EXEs.

## Actual result

- 13 Compute GTests, 3 Graph GTests and both public-header CTests passed.
- MatMul covers non-square/tail dimensions, M1/M7/M16/M65/M129, constant packing,
  changing dense RHS and transposed physical RHS against an independent oracle.
- Separate/packed SiLU, complete FFN, changed input replay, two slots, retained
  results, destroyed source Graph, custom SPIR-V and ready native handoff passed.
- Seven lifecycle cases cover quota/partial initialization, unaccepted submit OOM,
  observation failure until actual native retirement, cancel/abandon/close,
  exported owner tokens, concurrent independent slots and collect.
- Non-coherent flush/invalidate code ran through real Vulkan calls with the
  coherent flag masked in private test metadata. This does not qualify a physical
  non-coherent memory type on this device.
- Actual LastTest.log contains no Validation Error, VUID or SYNC-HAZARD.

No public fault hook or mock provider was added. Boundary failure injection is
test-private; accepted work still executes on the actual GPU. Deliberate GPU hang
or device loss was not induced. Tests do not certify AMD/Intel/MoltenVK performance.

## Bounded corrections

The first build rejected test `getenv` under MSVC WERROR; the fixture now uses
`_dupenv_s` on Windows. The first GPU run then exposed two integration defects:
the generated shader revision used a nonexistent CMake variable, and matrix-matrix
push constants confused reduction span `k_split` with a partition count. Passing 1
computed only one term. The corrected adapter passes K and the exact 64-byte
upstream push layout; the borrowed shader bytes and numerical tolerances are
unchanged. Matvec/custom and all seven lifecycle tests already passed before this
correction. The complete corrected suite passed once; no repeated timing search
or P1 run occurred.

[raw.zip](raw.zip) retains all three attempts, build/CTest/LastTest logs, exact
product-input hashes, controller scripts, layer preparation and the actual selector
plan. [Identity](identity.json) records archive SHA-256
`0f7dd515dceb28a1881a1f619f3bef80902cb5755c42edd5ed11e8339ef01405`.
The successful run verified that its product/test inputs were unchanged throughout.
The plan is READY; broad component/package gates triggered by the central CMake
change remain unexecuted and will be handled with the final product/delivery shape.

Remaining V6/V7 obligations include MoltenVK execution, host instrumentation,
ordinary facade samples, both-system three-delivery/relocation, fixed default-rule
qualification, P0/P1 and final review. This checkpoint does not sign those gates.
