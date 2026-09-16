# R6 Vulkan source, compiler and environment admission

Bounded input to the [R6 SCOPED audit](../../../docs/implementation/tasks/compute/vulkan-v1-four-route-foundation.md).
No product target or native compute result is delivered by this record. Raw commands,
logs, SPIR-V, input identities and the executed compiler-admission script are in
`admission-raw.zip`; `identity.json` binds that archive to the canonical baseline.

## Concrete source findings

The registered independent ggml checkout remains at
`456172ec733a135778adcd32d00e576a58232e45`. Every inspected shader input was compared
with that commit's Git blob; peer files were not edited.

- `include/ggml-vulkan.h` has backend/device/buffer-type discovery but no public
  VkDevice/VkQueue/VkBuffer import/export contract. A full runtime wrapper alone
  would not supply the required native handoff.
- `ggml-vulkan.cpp` owns native pools and traverses/encodes each graph execution.
  `VK_CHECK` throws native system/device-loss exceptions. This is distinct from
  the old CPU ggml owner's abort/threadpool blocker; that owner remains BLOCKED.
- `mul_mm.comp`, `mul_mat_vec.comp` and `swiglu.comp` provide mature matrix/vector
  and fused gate math. FFN is not evidence that unrelated attention/norm fusion
  must be split or reimplemented. The admitted math subset must retain applicable
  mature variants and their numerical/feature contracts.
- Shader generator `process_shaders` invokes cooperative-matrix variants with
  `fp16=true`; `FLOAT_TYPE` then converts loaded F32 values to float16. F32 input
  storage and F32 accumulation alone do not establish full F32 multiply semantics.
  Those variants cannot silently replace a strict F32 baseline or its comparator.

## Actual finite compiler check

Windows glslang 15.2.0 compiled four unmodified-source F32 variants to SPIR-V 1.5
for Vulkan 1.2. Only the include-directive preamble and explicit compile defines
were supplied. The receipt records each command, source hash and output hash.

| Variant | SPIR-V capabilities actually emitted | Result |
|---|---|---|
| F32 tiled matrix multiply | Shader | Compilation PASS |
| F32 matrix-vector, shared-memory reduction | Shader | Compilation PASS |
| F32 matrix-vector, subgroup reduction | Shader, GroupNonUniform, GroupNonUniformArithmetic | Compilation PASS |
| F32 SwiGLU | Shader | Compilation PASS |

The GLSL sources declare 8/16-bit extensions for shared type definitions, but the
three baseline modules do not emit corresponding SPIR-V capabilities. Runtime
admission must inspect the selected artifact's real requirements, limits and
enabled device features. Compilation and capability inspection do not establish
pipeline validity, numerical correctness or a fastest variant.

## Actual environment inventory

- Windows: NVIDIA RTX 4070, driver 610.62, Vulkan API/loader 1.4.341, subgroup 32,
  49,152-byte compute shared memory. KHR/NV cooperative-matrix extensions are
  reported; no HYLE qualification follows from enumeration. The loader reports
  a pre-existing missing OBS layer manifest; no registry/user configuration was
  changed. Khronos validation layer is not currently enumerated.
- Air Mac16,12: Apple M4, MoltenVK 1.4.1, device API 1.4.334, loader 1.4.350.1,
  glslang 16.3.0, subgroup 32, 32,768-byte compute shared memory. No cooperative
  matrix support or validation layer was reported. This is portability admission,
  not an Apple-native performance claim.
- Mini: no Vulkan/MoltenVK tool installation found at the checked locations;
  its R5 native Metal toolchain is unaffected.
- WSL: loader `libvulkan.so.1.3.275` and Mesa ICD manifests exist; `vulkaninfo`
  was not found. No native GPU, software-device or correctness result is asserted.

Remaining bounded decisions are the ordinary/native API and lifetime freeze,
exact supported shader/format domains, validation-layer/runtime preparation and
the finite implementation FIX set. Native product correctness, three deliveries
and P1 remain unperformed for R6. This record does not alter the R5 candidate or
authorize implementation while either Vulkan owner is SCOPED.
