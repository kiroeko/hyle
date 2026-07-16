#include "hyle/simd/random.h"

#include "hyle/simd/runtime.h"
#include "src/simd/random/kernels.h"

#if defined(HYLE_SIMD_EXACT_TARGET_AVX512)
#define HYLE_RANDOM(name) detail::name##_avx512
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX2)
#define HYLE_RANDOM(name) detail::name##_avx2
#elif defined(HYLE_SIMD_EXACT_TARGET_SSE4)
#define HYLE_RANDOM(name) detail::name##_sse4
#elif defined(HYLE_SIMD_EXACT_TARGET_NEON)
#define HYLE_RANDOM(name) detail::name##_neon
#elif defined(HYLE_SIMD_EXACT_TARGET_SVE)
#define HYLE_RANDOM(name) detail::name##_sve
#elif defined(HYLE_SIMD_EXACT_TARGET_RVV)
#define HYLE_RANDOM(name) detail::name##_rvv
#elif defined(HYLE_SIMD_EXACT_TARGET_SCALAR)
#define HYLE_RANDOM(name) detail::name##_scalar
#endif

namespace hyle::simd {

void random_fill(RandomState& state, std::uint64_t* dst,
                 std::size_t count) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_RANDOM(random_fill_u64)(&state, dst, count);
#else
  runtime().random.fill_u64(&state, dst, count);
#endif
}
void random_fill(RandomState& state, float* dst, std::size_t count) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_RANDOM(random_fill_f32)(&state, dst, count);
#else
  runtime().random.fill_f32(&state, dst, count);
#endif
}
void random_fill(RandomState& state, double* dst, std::size_t count) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_RANDOM(random_fill_f64)(&state, dst, count);
#else
  runtime().random.fill_f64(&state, dst, count);
#endif
}
}  // namespace hyle::simd

#ifdef HYLE_RANDOM
#undef HYLE_RANDOM
#endif
