#include <cstddef>
#include <cstdint>

#include "hyle/simd/native/random.h"
#include "src/simd/random/kernels.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including random/kernel_impl.h"
#endif

#define HYLE_SIMD_RANDOM_CAT_(a, b) a##b
#define HYLE_SIMD_RANDOM_CAT(a, b) HYLE_SIMD_RANDOM_CAT_(a, b)
#define HYLE_SIMD_RANDOM_FN(name) HYLE_SIMD_RANDOM_CAT(name, HYLE_SIMD_VARIANT_SUFFIX)

namespace hyle::simd::detail {

void HYLE_SIMD_RANDOM_FN(random_fill_u64_)(RandomState* state, std::uint64_t* dst,
                                           std::size_t count) noexcept {
  native::random_fill(*state, dst, count);
}
void HYLE_SIMD_RANDOM_FN(random_fill_f32_)(RandomState* state, float* dst,
                                           std::size_t count) noexcept {
  native::random_fill(*state, dst, count);
}
void HYLE_SIMD_RANDOM_FN(random_fill_f64_)(RandomState* state, double* dst,
                                           std::size_t count) noexcept {
  native::random_fill(*state, dst, count);
}

}  // namespace hyle::simd::detail

#undef HYLE_SIMD_RANDOM_FN
#undef HYLE_SIMD_RANDOM_CAT
#undef HYLE_SIMD_RANDOM_CAT_
