#pragma once

#include <cstddef>
#include <cstdint>

namespace hyle::simd::detail {

#define HYLE_SCAN_TYPES(X, tier) \
  X(std::uint32_t, u32, tier)    \
  X(std::int32_t, i32, tier)     \
  X(std::uint64_t, u64, tier)    \
  X(std::int64_t, i64, tier)     \
  X(float, f32, tier)            \
  X(double, f64, tier)

#define HYLE_SCAN_DECL(T, tag, tier)                                      \
  void inclusive_scan_##tag##_##tier(const T*, T*, std::size_t) noexcept; \
  void exclusive_scan_##tag##_##tier(const T*, T*, std::size_t,           \
                                      T) noexcept;

HYLE_SCAN_TYPES(HYLE_SCAN_DECL, scalar)
HYLE_SCAN_TYPES(HYLE_SCAN_DECL, sse4)
HYLE_SCAN_TYPES(HYLE_SCAN_DECL, avx2)
HYLE_SCAN_TYPES(HYLE_SCAN_DECL, avx512)
HYLE_SCAN_TYPES(HYLE_SCAN_DECL, neon)
HYLE_SCAN_TYPES(HYLE_SCAN_DECL, sve)
HYLE_SCAN_TYPES(HYLE_SCAN_DECL, rvv)

#undef HYLE_SCAN_DECL
#undef HYLE_SCAN_TYPES

}  // namespace hyle::simd::detail
