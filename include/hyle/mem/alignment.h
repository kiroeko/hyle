#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/core/compiler.h"

// Compile-time alignment helpers (clay alignment.h, generalized). All take a
// power-of-two alignment; bit tricks below are only valid for powers of two, so
// each asserts it. Sizes and pointers both covered -- mem/ rounds block sizes,
// simd/ aligns load/store addresses; PodVector aligns element storage.
//
// clay had only CLAY_ALIGNMENT + align_size() (which min-clamps to alignment, an
// allocation-sizing quirk); HYLE provides the full up/down/is_aligned set with
// no surprising clamp.

namespace hyle::mem {

// 64 covers up to AVX-512 loads/stores. Cache-line spacing is a separate core
// layout contract and is 128 bytes on Apple Silicon.
inline constexpr std::size_t kMaxSimdAlignment = 64;

HYLE_FORCE_INLINE constexpr bool is_pow2(std::size_t a) noexcept {
  return a != 0 && (a & (a - 1)) == 0;
}

// Overflow-aware form for sizes derived from untrusted or runtime input.
// `out` is written only on success.
HYLE_FORCE_INLINE constexpr bool try_align_up(
    std::size_t n, std::size_t a, std::size_t *out) noexcept {
  HYLE_FATAL_ASSERT(is_pow2(a));
  HYLE_FATAL_ASSERT(out != nullptr);
  const std::size_t padding = a - 1;
  if (n > SIZE_MAX - padding)
    return false;
  *out = (n + padding) & ~padding;
  return true;
}

// Smallest multiple of `a` that is >= n.
// An unrepresentable result is a caller bug; use try_align_up() when overflow
// is an expected input condition.
HYLE_FORCE_INLINE constexpr std::size_t align_up(std::size_t n,
                                                 std::size_t a) noexcept {
  std::size_t out = 0;
  HYLE_FATAL_ASSERT(try_align_up(n, a, &out));
  return out;
}

// Largest multiple of `a` that is <= n.
HYLE_FORCE_INLINE constexpr std::size_t align_down(std::size_t n,
                                                   std::size_t a) noexcept {
  HYLE_FATAL_ASSERT(is_pow2(a));
  return n & ~(a - 1);
}

HYLE_FORCE_INLINE constexpr bool is_aligned(std::size_t n,
                                            std::size_t a) noexcept {
  HYLE_FATAL_ASSERT(is_pow2(a));
  return (n & (a - 1)) == 0;
}

namespace detail {
// Overflow-aware address alignment for bump allocators. A synthetic integer input keeps the
// otherwise unreachable UINTPTR_MAX edge directly testable without fabricating an invalid
// pointer. `out` is written only on success.
HYLE_FORCE_INLINE constexpr bool try_align_up_address(
    std::uintptr_t n, std::size_t a, std::uintptr_t* out) noexcept {
  HYLE_FATAL_ASSERT(is_pow2(a));
  HYLE_FATAL_ASSERT(out != nullptr);
  const auto padding = static_cast<std::uintptr_t>(a - 1);
  if (n > UINTPTR_MAX - padding) return false;
  *out = (n + padding) & ~padding;
  return true;
}
}  // namespace detail

// ---- pointer variants -------------------------------------------------------
// Distinct names (not overloads): an integer literal like 0 is a null-pointer
// constant, so overloading size vs pointer is ambiguous (is_aligned(0u,16) could
// mean either). Separate names also read clearer -- the caller states whether a
// size or an address is being aligned.
HYLE_FORCE_INLINE void* align_up_ptr(void* p, std::size_t a) noexcept {
  return reinterpret_cast<void*>(
      align_up(reinterpret_cast<std::uintptr_t>(p), a));
}
HYLE_FORCE_INLINE void* align_down_ptr(void* p, std::size_t a) noexcept {
  return reinterpret_cast<void*>(
      align_down(reinterpret_cast<std::uintptr_t>(p), a));
}
HYLE_FORCE_INLINE bool is_aligned_ptr(const void* p, std::size_t a) noexcept {
  return is_aligned(reinterpret_cast<std::uintptr_t>(p), a);
}

}  // namespace hyle::mem
