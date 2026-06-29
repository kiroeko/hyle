#pragma once

#include <cstddef>
#include <cstdint>
#include <concepts>
#include <cstring>  // std::memcpy (hash_small small-key load)
#include <string_view>
#include <type_traits>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
#  include <intrin.h>  // _umul128 / __umulh for the 128-bit fold mixer
#endif

#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_CONTAINER_API
#elif defined(HYLE_CONTAINER_BUILD)
#  define HYLE_CONTAINER_API HYLE_EXPORT
#else
#  define HYLE_CONTAINER_API HYLE_IMPORT
#endif

namespace hyle::container {

// XXH3 core for LONGER byte ranges (len > 16). Defined in hash.cc so the 7490-line
// xxhash.h stays out of this header (ABI-clean + no compile-time bloat in every
// consumer). Small inputs (<= 16B) DON'T come here -- hash_bytes() short-circuits
// them to an inline mix below (no call), so the common small-key case pays zero
// seam cost. This is HYLE's default RUNTIME byte hasher.
namespace detail {
HYLE_CONTAINER_API std::uint64_t hash_bytes_xxh3(const void* data, std::size_t len,
                                                 std::uint64_t seed) noexcept;
}  // namespace detail

namespace detail {
// 64-bit integer/pointer mixer. abseil-style: ONE 128-bit multiply of (x^k)*k,
// folded to 64 via hi^lo. Shorter dependency chain than splitmix64 (1 serial
// multiply vs 2 + interleaved shifts) -- the win on find-miss, where the hash is
// a large fraction of cost and no slot is touched. The fold is statistical, NOT a
// bijection: distinct ints CAN collide (~1/2^64). Hashmap correctness never relies
// on bijection -- eq_ confirms real key equality; bijection only saved the rare
// extra eq_. abseil uses this exact mix (absl/hash) in production at scale.
HYLE_FORCE_INLINE std::uint64_t mix64(std::uint64_t x) noexcept {
  constexpr std::uint64_t k = 0x9ddfea08eb382d69ull;  // odd, high entropy
  const std::uint64_t a = x ^ k;
#if defined(__GNUC__) || defined(__clang__)
  // __uint128_t exists on every 64-bit GCC/Clang target (x64 + ARM64).
  // __extension__ silences -Wpedantic's "ISO C++ does not support __int128".
  __extension__ using u128 = unsigned __int128;
  const u128 m = static_cast<u128>(a) * k;
  return static_cast<std::uint64_t>(m) ^ static_cast<std::uint64_t>(m >> 64);
#elif defined(_MSC_VER) && defined(_M_X64)
  std::uint64_t hi;
  const std::uint64_t lo = _umul128(a, k, &hi);
  return lo ^ hi;
#elif defined(_MSC_VER) && defined(_M_ARM64)
  return (a * k) ^ __umulh(a, k);
#else
  // Portable fallback (no 128-bit multiply): splitmix64. Correctness identical;
  // only this rare path pays the longer chain.
  std::uint64_t z = a;
  z ^= z >> 30; z *= 0xbf58476d1ce4e5b9ull;
  z ^= z >> 27; z *= 0x94d049bb133111ebull;
  z ^= z >> 31; return z;
#endif
}

// Inline hash for a SMALL fixed-size byte blob (n <= 16) -- the common map-key case
// (PODs/small structs). absl/F14 do the same: small keys get an inline integer mix, NOT a
// call into the general byte hasher. This avoids the non-inlinable hash_bytes (XXH3) call
// whose fixed cost dominates tiny inputs (measured ~1.85x at 8B). Loads up to two 64-bit
// words (zero-padded; n is constant per type so no length ambiguity) and folds them through
// the same 128-bit-multiply mixer as the integer path -- so both words avalanche.
HYLE_FORCE_INLINE std::uint64_t hash_small(const void* p, std::size_t n) noexcept {
  std::uint64_t lo = 0, hi = 0;
  std::memcpy(&lo, p, n < 8 ? n : 8);
  if (n > 8) std::memcpy(&hi, static_cast<const char*>(p) + 8, n - 8);
  std::uint64_t h = mix64(lo);
  if (n > 8) h = mix64(h ^ hi);
  return h;
}
}  // namespace detail

// Hash an arbitrary byte range. SMALL inputs (len <= 16 -- the common map-key /
// short-string case) take an INLINE mix here (seed + len folded in, no call): the
// non-inlinable XXH3 call's fixed cost dominated tiny inputs (measured ~1.85x at
// 8B), and that whole cost is the wrapper, not the algorithm. LARGER inputs go to
// the out-of-line XXH3 core (detail::hash_bytes_xxh3), where the bytes dominate so
// the one call amortizes and xxhash.h stays hidden from this header. Output is NOT
// bit-identical to pure XXH3 for <= 16B (HYLE's contract is "good 64-bit hash +
// seed-sensitive + deterministic-within-build", not XXH3 byte-compat); nothing
// persists a hash across builds, so this is safe. >16B is unchanged (real XXH3).
HYLE_FORCE_INLINE std::uint64_t hash_bytes(const void* data, std::size_t len,
                                           std::uint64_t seed = 0) noexcept {
  if (len <= 16) {
    // Fold seed + len so different seeds AND different lengths diverge (XXH3 does
    // both). One mix for <=8B, a second folded mix for 9..16B. Avalanche and
    // distinctness are covered by the hash tests (StringView 10k + SmallStruct).
    std::uint64_t lo = 0, hi = 0;
    if (len != 0) {  // memcpy(_,nullptr,0) is UB; len==0 -> pure seed/len mix
      std::memcpy(&lo, data, len < 8 ? len : 8);
      if (len > 8) std::memcpy(&hi, static_cast<const char*>(data) + 8, len - 8);
    }
    std::uint64_t h = detail::mix64(lo ^ seed ^ (len * 0x9e3779b97f4a7c15ull));
    if (len > 8) h = detail::mix64(h ^ hi);
    return h;
  }
  return detail::hash_bytes_xxh3(data, len, seed);
}

// Default hasher. Specialized for the common cheap cases. For SMALL trivially-copyable T
// (<= 16 bytes -- the typical map key) it uses an inline mix (no XXH3 call); larger blobs
// fall back to XXH3 via hash_bytes (where the bytes dominate and the call is amortized).
// Users may specialize Hash<T> for their own types.
template <class T, class = void>
struct Hash {
  static_assert(std::is_trivially_copyable_v<T>,
                "Hash<T>: specialize Hash for non-trivially-copyable T");
  std::uint64_t operator()(const T& v) const noexcept {
    if constexpr (sizeof(T) <= 16) {
      return detail::hash_small(&v, sizeof(T));  // inline, no call
    } else {
      return hash_bytes(&v, sizeof(T));  // XXH3 (bytes amortize the seam)
    }
  }
};

// Integral / enum: inline mix, no XXH3 call.
template <class T>
struct Hash<T, std::enable_if_t<std::is_integral_v<T> || std::is_enum_v<T>>> {
  std::uint64_t operator()(T v) const noexcept {
    return detail::mix64(static_cast<std::uint64_t>(v));
  }
};

// Pointers: mix the address (drop the always-zero low alignment bits first).
template <class T>
struct Hash<T*> {
  std::uint64_t operator()(T* p) const noexcept {
    return detail::mix64(reinterpret_cast<std::uintptr_t>(p));
  }
};

// Strings / byte views: XXH3 over the bytes.
template <>
struct Hash<std::string_view> {
  std::uint64_t operator()(std::string_view s) const noexcept {
    return hash_bytes(s.data(), s.size());
  }
};

// A type usable as a hash key: Hash<T> yields a uint64 for it.
template <class T>
concept Hashable = requires(const T& v) {
  { Hash<T>{}(v) } -> std::convertible_to<std::uint64_t>;
};

}  // namespace hyle::container
