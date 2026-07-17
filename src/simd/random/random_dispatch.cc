#include "hyle/simd/random.h"

#include "hyle/simd/runtime.h"
#include "src/simd/dispatch_internal.h"
#include "src/simd/random/kernels.h"

namespace hyle::simd {

void random_fill(RandomState& state, std::uint64_t* dst,
                 std::size_t count) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_SIMD_EXACT_SYMBOL(random_fill_u64)(&state, dst, count);
#else
  runtime().random.fill_u64(&state, dst, count);
#endif
}
void random_fill(RandomState& state, float* dst, std::size_t count) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_SIMD_EXACT_SYMBOL(random_fill_f32)(&state, dst, count);
#else
  runtime().random.fill_f32(&state, dst, count);
#endif
}
void random_fill(RandomState& state, double* dst, std::size_t count) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_SIMD_EXACT_SYMBOL(random_fill_f64)(&state, dst, count);
#else
  runtime().random.fill_f64(&state, dst, count);
#endif
}
}  // namespace hyle::simd
