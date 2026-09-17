# R6 MoltenVK and host sanitizers

Executed candidate `ecdd92a7a5e343618b1d3f32fa2e459a641e29aa` on the MacBook Air
Mac16,12 / Apple M4 verification checkout. Windows remains the canonical product
workspace. AppleClang 21.0.0.21000334, Release/WERROR, locked Vulkan-Headers 1.4.357,
glslang 16.3, Vulkan loader 1.4.350 and MoltenVK 1.4.1 / device API 1.4.334.

The native build ran all 13 Compute and 3 Graph GTests and both public-header
CTests successfully. Khronos validation 1.4.357 was explicitly enabled, including
synchronization validation; actual LastTest.log has no VUID or SYNC-HAZARD.
This covers the same full numerical, FFN/custom SPIR-V, lifetime, failure, retained
result, changed binding and concurrent-slot cases as the
[Windows checkpoint](../vulkan-r6-windows-native/README.md).

Three separate configurations then ran those 16 real native GTests under ASan,
UBSan and TSan, all PASS. Each receipt retains 12 instrumented product/generated/
test compilation commands, executable hashes, CMake cache and complete test logs.
The system loader, MoltenVK, validation library and GPU shaders are uninstrumented;
host PASS is not a GPU sanitizer claim. The sanitizer runs disabled the external
validation layer; the separate native run above provides validation-layer evidence.

The first native attempt at `a1329dd2` compiled all six shaders, then exposed the
build script's use of Python 3.10-only `Path.write_text(newline=...)` on macOS's
Python 3.9. The canonical fix writes the same UTF-8 bytes directly. Windows
regeneration proved all six SPIR-V files and both generated C++ files byte-identical
to the previously checked artifacts. No shader, tolerance or runtime API changed.
Both attempts and the three sanitizer runs are retained; no P1 was run.

[Archive](raw.zip), [identity](identity.json), SHA-256
`08c76c0fcfc895f244b6a8901d09f15598138843771e38818e6e8de75ea1d262`.
All 32 entries were rehashed after transfer. The portable validation dylib was
extracted from the official Homebrew bottle; only its private verification copy's
dependency paths and layer JSON were relocated. No global driver/SDK installation
or system configuration was changed.

These are [R6 V6](../../../docs/implementation/tasks/compute/vulkan-v1-four-route-foundation.md)
correctness/host-instrumentation results. Public three-delivery, rule/P1 and final
review remain pending. Physical non-coherent memory, deliberate GPU device-loss,
AMD/Intel and other hardware are not certified by these runs.
