#include "hyle/platform.h"

#if defined(HYLE_ARCH_X64)

#define HYLE_SIMD_TARGET_AVX512
#define HYLE_SIMD_FORCE_AVX512_VNNI
#include "hyle/simd/native/ops.h"
#include "src/simd/ops/kernels.h"

namespace hyle::simd::detail {

std::int64_t dot_i8_i8_avx512_vnni(const std::int8_t* x,
                                   const std::int8_t* y,
                                   std::size_t n) noexcept {
  return native::dot(x, y, n);
}

}  // namespace hyle::simd::detail

#endif
