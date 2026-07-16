#pragma once

#include <cstddef>
#include <cstdint>

namespace hyle::simd {
class RandomState;
}

namespace hyle::simd::detail {

#define HYLE_RANDOM_DECLARE(suffix)                                                   \
  void random_fill_u64_##suffix(RandomState*, std::uint64_t*, std::size_t) noexcept; \
  void random_fill_f32_##suffix(RandomState*, float*, std::size_t) noexcept;         \
  void random_fill_f64_##suffix(RandomState*, double*, std::size_t) noexcept

HYLE_RANDOM_DECLARE(scalar);
HYLE_RANDOM_DECLARE(sse4);
HYLE_RANDOM_DECLARE(avx2);
HYLE_RANDOM_DECLARE(avx512);
HYLE_RANDOM_DECLARE(neon);
HYLE_RANDOM_DECLARE(sve);
HYLE_RANDOM_DECLARE(rvv);

#undef HYLE_RANDOM_DECLARE

}  // namespace hyle::simd::detail
