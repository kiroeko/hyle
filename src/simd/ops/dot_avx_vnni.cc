#include "hyle/platform.h"

#if defined(HYLE_ARCH_X64)

#define HYLE_SIMD_TARGET_AVX2
#define HYLE_SIMD_FORCE_AVX_VNNI
#include "hyle/simd/native/ops.h"
#include "src/simd/ops/kernels.h"

namespace hyle::simd::detail {

// AVX-VNNI is orthogonal to the broad AVX2 tier. This TU is compiled with
// VEX-encoded VNNI enabled and is installed into the runtime table only after
// CPUID plus OS-YMM-state gating.
std::int64_t dot_i8_i8_avx_vnni(const std::int8_t* x,
                                const std::int8_t* y,
                                std::size_t n) noexcept {
  return native::dot(x, y, n);
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
