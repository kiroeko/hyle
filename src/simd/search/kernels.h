#pragma once

#include <cstddef>
#include <cstdint>

namespace hyle::simd::detail {

#define HYLE_SEARCH_TYPES(X, tier)       \
  X(std::int8_t, i8, tier)               \
  X(std::uint8_t, u8, tier)              \
  X(std::int16_t, i16, tier)             \
  X(std::uint16_t, u16, tier)            \
  X(std::int32_t, i32, tier)             \
  X(std::uint32_t, u32, tier)            \
  X(std::int64_t, i64, tier)             \
  X(std::uint64_t, u64, tier)            \
  X(float, f32, tier)                     \
  X(double, f64, tier)

#define HYLE_SEARCH_DECL(T, tag, tier)                                      \
  std::size_t count_equal_##tag##_##tier(const T*, std::size_t, T) noexcept; \
  std::size_t find_equal_##tag##_##tier(const T*, std::size_t, T) noexcept;  \
  T min_value_##tag##_##tier(const T*, std::size_t) noexcept;                \
  T max_value_##tag##_##tier(const T*, std::size_t) noexcept;                \
  std::size_t index_of_min_##tag##_##tier(const T*, std::size_t) noexcept;   \
  std::size_t index_of_max_##tag##_##tier(const T*, std::size_t) noexcept;   \
  void minmax_value_##tag##_##tier(const T*, std::size_t, T*, T*) noexcept;  \
  bool is_sorted_asc_##tag##_##tier(const T*, std::size_t) noexcept;         \
  bool is_sorted_desc_##tag##_##tier(const T*, std::size_t) noexcept;

HYLE_SEARCH_TYPES(HYLE_SEARCH_DECL, scalar)
HYLE_SEARCH_TYPES(HYLE_SEARCH_DECL, sse4)
HYLE_SEARCH_TYPES(HYLE_SEARCH_DECL, avx2)
HYLE_SEARCH_TYPES(HYLE_SEARCH_DECL, avx512)
HYLE_SEARCH_TYPES(HYLE_SEARCH_DECL, neon)
HYLE_SEARCH_TYPES(HYLE_SEARCH_DECL, sve)
HYLE_SEARCH_TYPES(HYLE_SEARCH_DECL, rvv)

#undef HYLE_SEARCH_DECL
#undef HYLE_SEARCH_TYPES

}  // namespace hyle::simd::detail
