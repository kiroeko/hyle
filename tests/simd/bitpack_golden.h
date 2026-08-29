#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "hyle/simd/bitpack.h"

namespace hyle::simd::test {

template <typename T> constexpr T low_mask(unsigned bits) noexcept {
  constexpr unsigned W = 8 * sizeof(T);
  return bits == W ? std::numeric_limits<T>::max()
                   : static_cast<T>((std::uint64_t{1} << bits) - 1);
}

template <typename T> T sample_value(std::size_t index) noexcept {
  std::uint64_t value =
      (static_cast<std::uint64_t>(index) + 1) * 0x9E3779B97F4A7C15ull;
  value ^= value >> 29;
  value *= 0xBF58476D1CE4E5B9ull;
  value ^= value >> 32;
  return static_cast<T>(value);
}

// FNV-1a digests pin every byte produced for two canonical full blocks plus a
// seven-value tail. They are compact golden data, not a correctness oracle by
// themselves: all-width round-trip, independent u16 11/13 formulas and direct
// cross-tier comparisons remain separate tests.
inline constexpr std::array<std::uint64_t, 8> kU8WireDigest = {
    0x7e8f4a74d7a3f72dull, 0x880330b370e3a1ddull, 0x955117fe8bba75cbull,
    0xf310aa68b48f0b58ull, 0x8b7b7db9070ecb09ull, 0xee45a4b98307d259ull,
    0x9dfce6ed4b54a05cull, 0x96236dc63171771aull};
inline constexpr std::array<std::uint64_t, 16> kU16WireDigest = {
    0x46e824a94900adbdull, 0xbe92892a610a7db8ull, 0x6d93ea7937c7165cull,
    0x98f62616aaa642ceull, 0x78c2e8bd02355edbull, 0x7c203443b5e14be3ull,
    0x3014f29eeae8a633ull, 0x2f5baa79d90fae4eull, 0xaafd7725a3a13223ull,
    0x372325a7d184ef02ull, 0x6a8f92eaaf322d33ull, 0xf79de4b98ddaaa71ull,
    0x46d4d0c8fe026aadull, 0x5593fdfe1f96ac89ull, 0x708dfeac97d0efc2ull,
    0x787aa9170a4effa7ull};
inline constexpr std::array<std::uint64_t, 32> kU32WireDigest = {
    0xe07663ef2f8204faull, 0xe48669ea4d4ca419ull, 0xe828f8a5228dc80eull,
    0x5019394378dfd252ull, 0x859e27265af44127ull, 0x772148b0ba1a1669ull,
    0x8f93f382cddb4794ull, 0x7aeccab8c34c8082ull, 0x68c318baae751e73ull,
    0x19659130c712f125ull, 0xdaf9f5662e21dd52ull, 0x0e1eeb97be25bc19ull,
    0xb73c5f00eb3009bcull, 0xf5466951fd3bd3e3ull, 0xe472f7693710110cull,
    0xf2d112f5f8ad9fe7ull, 0xbdfb26febcc1bc25ull, 0xb177db409f319b0bull,
    0xb7e4d0b39ef58f31ull, 0x907959cf26188aa1ull, 0x295def3e4299f6ccull,
    0xe558cbf82e02af27ull, 0xa42b571e8a75601cull, 0x98ab65448c4ebe65ull,
    0x1c72e1360377de46ull, 0x505f14a47db6d2a7ull, 0xca97b19cb9074b13ull,
    0xd54c51ec3105c818ull, 0xbc62a50ebcc04729ull, 0x363bc84c3ec102fcull,
    0xbcbb93d0b62301e2ull, 0xd3872e703cd91f1eull};

template <typename T> std::uint64_t wire_digest(unsigned bits) {
  constexpr std::size_t kCount = 263;
  std::array<T, kCount> raw{};
  const T mask = low_mask<T>(bits);
  for (std::size_t i = 0; i < raw.size(); ++i)
    raw[i] = static_cast<T>(sample_value<T>(i + 37) & mask);
  std::vector<T> packed(packed_size<T>(raw.size(), bits));
  pack(raw.data(), raw.size(), packed.data(), bits);

  std::uint64_t hash = 14695981039346656037ull;
  for (T word : packed) {
    for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
      hash ^= (static_cast<std::uint64_t>(word) >> (8 * byte)) & 0xFFu;
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

} // namespace hyle::simd::test
