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
// wyhash core for LONG byte ranges (len > kWyhashThreshold). Also out-of-line in
// hash.cc (wyhash.h kept out of this header). Best-of-breed: XXH3 wins short-mid
// strings, wyhash wins long ones -- hash_bytes() picks by length (see below).
HYLE_CONTAINER_API std::uint64_t hash_bytes_wyhash(const void* data, std::size_t len,
                                                   std::uint64_t seed) noexcept;
// Process-local long-input dispatcher. On x86 it may select an AVX2 XXH3 TU for
// sufficiently large inputs; AArch64 uses its baseline NEON XXH3 path after the
// measured crossover. Other CPUs retain the scalar/baseline best-of path.
// This is intentionally an implementation seam rather than a user-selectable ISA
// API: callers describe bytes, platform owns safe feature selection.
HYLE_CONTAINER_API std::uint64_t hash_bytes_long(const void* data, std::size_t len,
                                                 std::uint64_t seed) noexcept;

// Length crossover between the two out-of-line hashers, from the length sweep
// (bench/container/hash_bench.cc, 15-run medians): XXH3 is fastest for 17..~128B
// (wins wyhash 1.1-1.26x), wyhash is fastest for >=~192B (wins 1.45-2.6x; XXH3 has
// a midsize-path pessimum near 256B). 128 is the conservative split -- the tie
// point sits on the XXH3 side, so wyhash only takes over where it clearly wins.
inline constexpr std::size_t kWyhashThreshold = 128;
// Varied-input sweeps put wide XXH3's repeatable *product API* crossover at
// different points for the shipping toolchains: 512B on Linux/GCC 14, but 1KiB
// on Windows/MSVC and macOS/Clang. The latter includes the public dispatch seam;
// using the raw algorithm crossover would make real 512/768B callers slower.
#if defined(__linux__) && !defined(HYLE_ARCH_ARM64)
inline constexpr std::size_t kWideHashThreshold = 512;
#else
inline constexpr std::size_t kWideHashThreshold = 1024;
#endif
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

// TWO-operand mixer: fold(a * b) via hi^lo. This is abseil's `Mix` (absl/hash) --
// it puts two independent 64-bit words into the two OPERANDS of a SINGLE 128-bit
// multiply, so a 9..16B key avalanches both words in ONE mul instead of two serial
// mix64() calls (saves ~one multiply's latency on the small-key hot path). Caller
// XORs high-entropy constants into each operand so neither is trivially zero.
// KNOWN corner (absl-documented, accepted): if one operand becomes exactly 0 the
// product is 0 and the other operand's bits are lost -- 1 in 2^64 per side, and
// only reachable at exactly 16B; the XOR'd constants + overlapping bytes at <16B
// make it practically unhittable. Quality is gated by SmallStructInlineQuality
// (40000 distinct + avalanche>=12 on every bit) and StringView 10k tests.
HYLE_FORCE_INLINE std::uint64_t mix64_2(std::uint64_t a, std::uint64_t b) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __extension__ using u128 = unsigned __int128;
  const u128 m = static_cast<u128>(a) * b;
  return static_cast<std::uint64_t>(m) ^ static_cast<std::uint64_t>(m >> 64);
#elif defined(_MSC_VER) && defined(_M_X64)
  std::uint64_t hi;
  const std::uint64_t lo = _umul128(a, b, &hi);
  return lo ^ hi;
#elif defined(_MSC_VER) && defined(_M_ARM64)
  return (a * b) ^ __umulh(a, b);
#else
  // Portable: fall back to folding two single-operand mixes (longer, rare path).
  return mix64(a) ^ mix64(b + 0x9e3779b97f4a7c15ull);
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
  if (n > 8) {
    // Keep the destination bound explicit even though every production caller
    // instantiates this helper with sizeof(T) <= 16. Besides making the helper
    // robust in isolation, this prevents GCC's inliner from diagnosing an
    // impossible oversized memcpy while cloning Hash<T> for larger T.
    const std::size_t hi_bytes =
        n <= 8 ? 0 : (n < 16 ? n - 8 : 8);
    const auto* bytes = static_cast<const char*>(p);
    const auto* hi_source = n <= 8 ? bytes : bytes + 8;
    std::memcpy(&hi, hi_source, hi_bytes);
  }
  if (n > 8) {
    // 9..16B: fold both words in ONE 128-bit multiply (abseil's Mix), not two serial
    // mix64 calls. XOR distinct high-entropy constants into each operand so neither is
    // trivially zero.
    constexpr std::uint64_t k1 = 0x9ddfea08eb382d69ull;
    constexpr std::uint64_t k2 = 0xff51afd7ed558ccdull;
    return mix64_2(lo ^ k1, hi ^ k2);
  }
  return mix64(lo);
}
}  // namespace detail

// Hash an arbitrary byte range. SMALL inputs (len <= 16 -- the common map-key /
// short-string case) take an INLINE mix here (seed + len folded in, no call): the
// non-inlinable XXH3 call's fixed cost dominated tiny inputs (measured ~1.85x at
// 8B), and that whole cost is the wrapper, not the algorithm. LARGER inputs go to
// the out-of-line XXH3 core (detail::hash_bytes_xxh3), where the bytes dominate so
// the one call amortizes and xxhash.h stays hidden from this header. Output is NOT
// bit-identical to pure XXH3 for <= 16B. This is a PROCESS-LOCAL table hash, like
// absl::Hash: seed-sensitive and repeatable inside one process, but not a persisted
// fingerprint. Its output may change across HYLE versions, builds, or CPU ISA as
// the runtime selects a faster long-input kernel. Never store it or put it on wire.
HYLE_FORCE_INLINE std::uint64_t hash_bytes(const void* data, std::size_t len,
                                           std::uint64_t seed = 0) noexcept {
  HYLE_FATAL_ASSERT(data != nullptr || len == 0);
  if (len <= 16) {
    // Fold seed + len so different seeds AND different lengths diverge (XXH3 does
    // both). One mix for <=8B, a second folded mix for 9..16B. Avalanche and
    // distinctness are covered by the hash tests (StringView 10k + SmallStruct).
    std::uint64_t lo = 0, hi = 0;
    if (len != 0) {  // memcpy(_,nullptr,0) is UB; len==0 -> pure seed/len mix
      std::memcpy(&lo, data, len < 8 ? len : 8);
      if (len > 8) {
        const std::size_t hi_bytes =
            len <= 8 ? 0 : (len < 16 ? len - 8 : 8);
        const auto* bytes = static_cast<const char*>(data);
        const auto* hi_source = len <= 8 ? bytes : bytes + 8;
        std::memcpy(&hi, hi_source, hi_bytes);
      }
    }
    const std::uint64_t s = lo ^ seed ^ (len * 0x9e3779b97f4a7c15ull);
    if (len > 8) {
      // 9..16B: one 128-bit multiply of the two seed/len-folded words (abseil Mix),
      // instead of two serial mix64 -- saves a multiply on this small-string path.
      return detail::mix64_2(s, hi ^ 0xff51afd7ed558ccdull);
    }
    return detail::mix64(s);
  }
  // >16B: best-of-breed by length (runtime `len`). XXH3 for short-mid; the long
  // dispatcher chooses wyhash or a runtime-safe wide XXH3 kernel. All vendor
  // headers stay hidden and hashes remain repeatable for this process.
  if (len <= detail::kWyhashThreshold) return detail::hash_bytes_xxh3(data, len, seed);
  return detail::hash_bytes_long(data, len, seed);
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
    // sizeof(T) is a COMPILE-TIME constant, so this whole ladder resolves at compile
    // time (if constexpr): each Hash<T> instantiation calls exactly ONE hasher with
    // ZERO runtime branch -- the best-of-breed pick is baked in per type.
    if constexpr (sizeof(T) <= 16) {
      return detail::hash_small(&v, sizeof(T));           // inline mix, no call
    } else if constexpr (sizeof(T) <= detail::kWyhashThreshold) {
      return detail::hash_bytes_xxh3(&v, sizeof(T), 0);   // short-mid: XXH3 fastest
    } else {
      return detail::hash_bytes_long(&v, sizeof(T), 0);   // long: runtime best-of
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
