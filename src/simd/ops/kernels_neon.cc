// NEON variant of the dispatched kernels (aarch64 baseline, no extra -march). Only ENTERED
// when best_isa() == kNeon (i.e. an aarch64 build), so it never coexists with the x86 TUs.
// Same shared body as the scalar/sse4/avx2 TUs -- only the backend header (via the target
// macro) and the symbol suffix differ. neon.h backend written+objdump-verified in simd-03;
// this TU is what finally compiles it into the runtime-dispatched path on real hardware.
#define HYLE_SIMD_TARGET_NEON
#define HYLE_KERNEL_SUFFIX neon
#include "src/simd/ops/kernel_impl.h"
