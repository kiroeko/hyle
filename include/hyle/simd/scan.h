#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/visibility.h"

// Contiguous inclusive and exclusive prefix sums. The exported core accepts
// distinct source/destination ranges or exact in-place aliasing; partial
// overlap is outside the contract. For n == 0, both pointers may be null.
//
// Integer sums are modulo 2^N, including signed types (there is no signed
// overflow UB). Floating-point association is deterministic for a fixed
// selected target, but may differ between ISA widths.
//
// Time O(n), auxiliary space O(1), no allocation and no locking.

namespace hyle::simd {

#define HYLE_SIMD_SCAN_DECLARE(T)                                             \
  HYLE_SIMD_API void inclusive_scan(const T* src, T* dst,                    \
                                     std::size_t n) noexcept;                 \
  HYLE_SIMD_API void exclusive_scan(const T* src, T* dst, std::size_t n,     \
                                     T init) noexcept;                        \
  inline void inclusive_scan(T* data, std::size_t n) noexcept {              \
    inclusive_scan(data, data, n);                                           \
  }                                                                          \
  inline void exclusive_scan(const T* src, T* dst, std::size_t n) noexcept { \
    exclusive_scan(src, dst, n, T{});                                        \
  }                                                                          \
  inline void exclusive_scan(T* data, std::size_t n, T init) noexcept {      \
    exclusive_scan(data, data, n, init);                                     \
  }                                                                          \
  inline void exclusive_scan(T* data, std::size_t n) noexcept {              \
    exclusive_scan(data, data, n, T{});                                      \
  }

HYLE_SIMD_SCAN_DECLARE(std::uint32_t)
HYLE_SIMD_SCAN_DECLARE(std::int32_t)
HYLE_SIMD_SCAN_DECLARE(std::uint64_t)
HYLE_SIMD_SCAN_DECLARE(std::int64_t)
HYLE_SIMD_SCAN_DECLARE(float)
HYLE_SIMD_SCAN_DECLARE(double)

#undef HYLE_SIMD_SCAN_DECLARE

}  // namespace hyle::simd
