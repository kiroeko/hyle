#pragma once

#include <cstddef>
#include <cstdint>

namespace hyle::container::detail {

#define HYLE_DECLARE_BITSET_VARIANT(suffix)                                  \
  void bitset_and_##suffix(std::uint64_t*, const std::uint64_t*,             \
                           std::size_t) noexcept;                             \
  void bitset_or_##suffix(std::uint64_t*, const std::uint64_t*,              \
                          std::size_t) noexcept;                              \
  void bitset_xor_##suffix(std::uint64_t*, const std::uint64_t*,             \
                           std::size_t) noexcept;                             \
  void bitset_and_not_##suffix(std::uint64_t*, const std::uint64_t*,         \
                               std::size_t) noexcept;                         \
  void bitset_not_##suffix(std::uint64_t*, std::size_t) noexcept;            \
  std::size_t bitset_count_##suffix(const std::uint64_t*,                    \
                                    std::size_t) noexcept

HYLE_DECLARE_BITSET_VARIANT(scalar);
#if defined(HYLE_ARCH_X64)
HYLE_DECLARE_BITSET_VARIANT(sse4);
HYLE_DECLARE_BITSET_VARIANT(avx2);
HYLE_DECLARE_BITSET_VARIANT(avx512);
#elif defined(HYLE_ARCH_ARM64)
HYLE_DECLARE_BITSET_VARIANT(neon);
#endif

#undef HYLE_DECLARE_BITSET_VARIANT

}  // namespace hyle::container::detail
