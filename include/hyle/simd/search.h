#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/visibility.h"

// Target-neutral contiguous-range search and reduction operations. A prebuilt
// HYLE library resolves these pointer/count entry points to the best compiled
// ISA once per process. find_equal returns n when no element matches;
// index_of_min/index_of_max return n for an empty range and otherwise the
// first index whose value is the extremum.
//
// Floating-point contract:
// - equality uses IEEE == (NaN never matches);
// - min/max/minmax propagate NaN;
// - index_of_min/index_of_max return the first NaN index when any NaN is
//   present, matching the propagating extrema contract;
// - is_sorted uses strict comparisons, matching std::is_sorted (a NaN does not
//   itself form an inversion).
// Empty min/max ranges return +infinity/-infinity for floating point and
// highest/lowest for integers.
// For n == 0, data may be null. minmax_value requires two valid, distinct
// output pointers even for an empty input range. All operations are O(n), use
// O(1) auxiliary storage, and neither allocate nor lock.

namespace hyle::simd {

#define HYLE_SIMD_SEARCH_DECLARE(T)                                              \
  HYLE_SIMD_API std::size_t count_equal(const T* data, std::size_t n,           \
                                         T value) noexcept;                     \
  HYLE_SIMD_API std::size_t find_equal(const T* data, std::size_t n,            \
                                        T value) noexcept;                      \
  HYLE_SIMD_API T min_value(const T* data, std::size_t n) noexcept;             \
  HYLE_SIMD_API T max_value(const T* data, std::size_t n) noexcept;             \
  HYLE_SIMD_API std::size_t index_of_min(const T* data,                         \
                                         std::size_t n) noexcept;               \
  HYLE_SIMD_API std::size_t index_of_max(const T* data,                         \
                                         std::size_t n) noexcept;               \
  HYLE_SIMD_API void minmax_value(const T* data, std::size_t n, T* out_min,     \
                                  T* out_max) noexcept;                         \
  HYLE_SIMD_API bool is_sorted(const T* data, std::size_t n,                    \
                               bool ascending = true) noexcept;

HYLE_SIMD_SEARCH_DECLARE(std::int8_t)
HYLE_SIMD_SEARCH_DECLARE(std::uint8_t)
HYLE_SIMD_SEARCH_DECLARE(std::int16_t)
HYLE_SIMD_SEARCH_DECLARE(std::uint16_t)
HYLE_SIMD_SEARCH_DECLARE(std::int32_t)
HYLE_SIMD_SEARCH_DECLARE(std::uint32_t)
HYLE_SIMD_SEARCH_DECLARE(std::int64_t)
HYLE_SIMD_SEARCH_DECLARE(std::uint64_t)
HYLE_SIMD_SEARCH_DECLARE(float)
HYLE_SIMD_SEARCH_DECLARE(double)

#undef HYLE_SIMD_SEARCH_DECLARE

}  // namespace hyle::simd
