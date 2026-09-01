#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <cstring>  // std::memcpy (hash_small small-key load)
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
#  include <intrin.h>  // _umul128 / __umulh for the 128-bit fold mixer
#endif

#include "hyle/core/compiler.h"
#include "hyle/core/environment.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_CONTAINER_API
#elif defined(HYLE_CONTAINER_BUILD)
#  define HYLE_CONTAINER_API HYLE_EXPORT
#else
#  define HYLE_CONTAINER_API HYLE_IMPORT
#endif

namespace hyle::container {

namespace detail {
// Baseline scalar core for LONG byte ranges. Kept out of line so wyhash.h does
// not leak into every consumer translation unit.
HYLE_CONTAINER_API std::uint64_t hash_bytes_wyhash(const void* data, std::size_t len,
                                                   std::uint64_t seed) noexcept;
// Process-local long-input dispatcher. On x86 it may select an AVX2 XXH3 TU for
// sufficiently large inputs; AArch64 can use its baseline NEON XXH3 path for
// long inputs. Other CPUs retain the scalar/baseline path.
// This is intentionally an implementation seam rather than a user-selectable ISA
// API: callers describe bytes, platform owns safe feature selection.
HYLE_CONTAINER_API std::uint64_t hash_bytes_long(const void* data, std::size_t len,
                                                 std::uint64_t seed) noexcept;
// One best-effort per-process seed used by the fast default string hash. It is
// deliberately non-cryptographic: its job is to invalidate offline collision
// sets across independent executable starts. Each hasher/table caches one word
// of this shared seed, avoiding a global load or exported call on every lookup;
// there is no per-entry state. A fork performed after first use inherits the
// initialized seed; exec creates a new one. Callers needing secret rotation
// must manage a SipHash key.
HYLE_CONTAINER_API std::uint64_t process_hash_seed() noexcept;

// Upper bound of the inline medium-input kernel. Above it the bytes amortize the
// out-of-line product dispatcher; at and below it avoiding that seam wins in
// normal string/table workloads on all three shipping architectures.
inline constexpr std::size_t kWyhashThreshold = 128;
// Varied-input sweeps put wide XXH3's repeatable *product API* crossover at
// different points for the shipping toolchains: 512B on Linux/GCC 14, but 1KiB
// on Windows/MSVC and macOS/Clang. The latter includes the public dispatch seam;
// using the raw algorithm crossover would make real 512/768B callers slower.
#if defined(__linux__) && !HYLE_ARCH_ARM64
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

// Byte views carry a runtime length independently from their backing object's
// static extent. GCC's object-size pass otherwise diagnoses fixed loads in a
// length-guarded inlined branch as an overread when the pointer originated in a
// string literal, even though the branch contract proves the bytes exist. This
// empty compiler barrier forgets only that non-semantic provenance; it emits no
// instruction and keeps the memcpy load unaligned-safe.
HYLE_FORCE_INLINE const std::uint8_t* runtime_bytes(
    const std::uint8_t* data) noexcept {
#if defined(__GNUC__) && !defined(__clang__)
  __asm__ __volatile__("" : "+r"(data));
#endif
  return data;
}

HYLE_FORCE_INLINE std::uint64_t load64(const std::uint8_t* data) noexcept {
  data = runtime_bytes(data);
  std::uint64_t value;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

HYLE_FORCE_INLINE std::uint32_t load32(const std::uint8_t* data) noexcept {
  data = runtime_bytes(data);
  std::uint32_t value;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

// Runtime short-string lengths are mixed with one overlapping load from
// compile-time random data. This is the low-latency shape used by Abseil:
// unlike `len * constant`, it adds no dependent integer multiply before the
// 128-bit avalanche and still gives every length a distinct high-entropy
// state. The table is long enough for an eight-byte load at every len <= 16.
alignas(64) inline constexpr std::uint64_t kSmallHashLengthData[] = {
    0x243f6a8885a308d3ull, 0x13198a2e03707344ull,
    0xa4093822299f31d0ull, 0x082efa98ec4e6c89ull,
    0x452821e638d01377ull,
};

HYLE_FORCE_INLINE std::uint64_t small_hash_length_mix(
    std::size_t len) noexcept {
  HYLE_ASSUME(len <= 16);
  const auto* data = reinterpret_cast<const std::uint8_t*>(
      kSmallHashLengthData);
  return load64(data + len);
}

// 17..32B is still fixed-cost dominated. First and last 16-byte windows cover
// every input byte (overlapping for 17..31B); three independent multiply-folds
// avoid a shared-library/vendor call while retaining a seeded avalanche. This
// is a general byte-hash path, not an interner-only symbol-name special case.
HYLE_FORCE_INLINE std::uint64_t hash_medium_17_32(
    const void* input, std::size_t len, std::uint64_t seed) noexcept {
  const auto* bytes = runtime_bytes(
      static_cast<const std::uint8_t*>(input));
  const std::uint64_t first0 = load64(bytes);
  const std::uint64_t first1 = load64(bytes + 8);
  const std::uint64_t last0 = load64(bytes + len - 16);
  const std::uint64_t last1 = load64(bytes + len - 8);
  const std::uint64_t left =
      mix64_2(first0 ^ 0x9ddfea08eb382d69ull ^ seed,
              first1 ^ (len * 0x9e3779b97f4a7c15ull));
  const std::uint64_t right =
      mix64_2(last0 ^ 0xff51afd7ed558ccdull,
              last1 ^ 0xc4ceb9fe1a85ec53ull ^ (seed << 1));
  return mix64_2(left ^ 0x243f6a8885a308d3ull,
                 right ^ 0x13198a2e03707344ull ^ (seed >> 1));
}

struct Mul128Parts {
  std::uint64_t lo;
  std::uint64_t hi;
};

HYLE_FORCE_INLINE Mul128Parts mul128_parts(std::uint64_t a,
                                           std::uint64_t b) noexcept {
#  if defined(__GNUC__) || defined(__clang__)
  __extension__ using u128 = unsigned __int128;
  const u128 product = static_cast<u128>(a) * b;
  return {static_cast<std::uint64_t>(product),
          static_cast<std::uint64_t>(product >> 64)};
#  elif defined(_MSC_VER) && defined(_M_X64)
  std::uint64_t hi;
  const std::uint64_t lo = _umul128(a, b, &hi);
  return {lo, hi};
#  elif defined(_MSC_VER) && defined(_M_ARM64)
  std::uint64_t hi;
  const std::uint64_t lo = a * b;
  hi = __umulh(a, b);
  return {lo, hi};
#  endif
}

// Dynamic short strings are a different compiler problem from hash_small(): the
// latter sees a compile-time size, while this path receives a runtime length.
// Fixed-width first/last loads avoid a variable-sized memcpy and zero-filled
// temporary. This is the same mature shape used by Abseil/wyhash: overlapping
// windows are intentional and cover every input byte without reading outside the
// range. A single multiply-fold remains the dominant operation for each range.
HYLE_FORCE_INLINE std::uint64_t hash_bytes_small_inline(
    const void* input, std::size_t len, std::uint64_t seed) noexcept {
  constexpr std::uint64_t k1 = 0x9ddfea08eb382d69ull;
  constexpr std::uint64_t k2 = 0xff51afd7ed558ccdull;
  const auto* bytes = runtime_bytes(
      static_cast<const std::uint8_t*>(input));
  const std::uint64_t length_mix = small_hash_length_mix(len);

  if (len >= 8) {
    const std::uint64_t first = load64(bytes);
    const std::uint64_t last = load64(bytes + len - 8);
    return mix64_2(first ^ length_mix ^ seed, last ^ k1);
  }
  if (len >= 4) {
    const std::uint64_t first = load32(bytes);
    const std::uint64_t last = load32(bytes + len - 4);
    return mix64((first | (last << 32)) ^ seed ^ length_mix);
  }
  if (len != 0) {
    // first/middle/last is sufficient for 1..3 bytes and deliberately avoids a
    // length-dependent memcpy. Keep length in the high byte to disambiguate the
    // overlapping samples.
    const std::uint64_t packed =
        static_cast<std::uint64_t>(bytes[0]) |
        (static_cast<std::uint64_t>(bytes[len >> 1]) << 8) |
        (static_cast<std::uint64_t>(bytes[len - 1]) << 16) |
        (static_cast<std::uint64_t>(len) << 24);
    return mix64(packed ^ seed ^ length_mix);
  }
  return mix64(seed ^ k2 ^ length_mix);
}

// Keep the medium-only core inline: at these lengths a shared-library call is
// still a material fraction of the work. The >=48B loop is the published wyhash
// shape (three independent accumulators); long ranges remain out of line so
// ordinary call sites do not inherit a full streaming hasher.
HYLE_FORCE_INLINE std::uint64_t hash_bytes_medium_inline(
    const void* input, std::size_t len, std::uint64_t seed) noexcept {
  constexpr std::uint64_t secret[4] = {
      0x2d358dccaa6c78a5ull, 0x8bb84b93962eacc9ull,
      0x4b33a62ed433d4a3ull, 0x4d5a2da51de1aa47ull};
  const auto mix = [](std::uint64_t left, std::uint64_t right) noexcept {
    const Mul128Parts product = mul128_parts(left, right);
    return product.lo ^ product.hi;
  };

  const auto* bytes = static_cast<const std::uint8_t*>(input);
  seed ^= mix(seed ^ secret[0], secret[1]);
  if (len <= 32) {
    seed = mix(load64(bytes) ^ secret[1], load64(bytes + 8) ^ seed);
    std::uint64_t left = load64(bytes + len - 16) ^ secret[1];
    std::uint64_t right = load64(bytes + len - 8) ^ seed;
    const Mul128Parts folded = mul128_parts(left, right);
    return mix(folded.lo ^ secret[0] ^ len, folded.hi ^ secret[1]);
  }
  std::size_t remaining = len;
  if (remaining >= 48) {
    std::uint64_t second = seed;
    std::uint64_t third = seed;
    do {
      seed = mix(load64(bytes) ^ secret[1], load64(bytes + 8) ^ seed);
      second = mix(load64(bytes + 16) ^ secret[2],
                   load64(bytes + 24) ^ second);
      third = mix(load64(bytes + 32) ^ secret[3],
                  load64(bytes + 40) ^ third);
      bytes += 48;
      remaining -= 48;
    } while (remaining >= 48);
    seed ^= second ^ third;
  }
  while (remaining > 16) {
    seed = mix(load64(bytes) ^ secret[1], load64(bytes + 8) ^ seed);
    bytes += 16;
    remaining -= 16;
  }

  std::uint64_t left = load64(bytes + remaining - 16) ^ secret[1];
  std::uint64_t right = load64(bytes + remaining - 8) ^ seed;
  const Mul128Parts folded = mul128_parts(left, right);
  return mix(folded.lo ^ secret[0] ^ len, folded.hi ^ secret[1]);
}

// Inline hash for a SMALL fixed-size byte blob (n <= 16) -- the common map-key case
// (PODs/small structs). absl/F14 do the same: small keys get an inline integer mix, NOT a
// call into the general byte hasher. This avoids the non-inlinable long-input dispatcher
// whose fixed cost dominates tiny inputs. Loads up to two 64-bit
// words (zero-padded; n is constant per type so no length ambiguity) and folds them through
// the same 128-bit-multiply mixer as the integer path -- so both words avalanche.
HYLE_FORCE_INLINE std::uint64_t hash_small(const void* p, std::size_t n) noexcept {
  std::uint64_t lo = 0, hi = 0;
  std::memcpy(&lo, p, n < 8 ? n : 8);
  if (n > 8) {
    // Keep the destination bound explicit even though every production caller
    // instantiates this helper with sizeof(T) <= 16. Besides making the helper
    // robust in isolation, this prevents GCC's inliner from diagnosing an
    // impossible oversized memcpy while cloning a typed bytewise hasher for
    // larger T.
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
// short-string case) take an inline fixed-size mix; 17..128B use an inline
// multiply stream so a DLL seam cannot dominate symbol/path-sized strings; only
// longer inputs enter the out-of-line product dispatcher. Output is not
// bit-identical to a named vendor algorithm. This is a PROCESS-LOCAL table hash, like
// absl::Hash: seed-sensitive and repeatable inside one process, but not a persisted
// fingerprint. Its output may change across HYLE versions, builds, or CPU ISA as
// the runtime selects a faster long-input kernel. Never store it or put it on wire.
HYLE_FORCE_INLINE std::uint64_t hash_bytes(const void* data, std::size_t len,
                                           std::uint64_t seed = 0) noexcept {
  HYLE_FATAL_ASSERT(data != nullptr || len == 0);
  if (len <= 16) {
    return detail::hash_bytes_small_inline(data, len, seed);
  }
  // >16B: best-of-breed by length (runtime `len`). Medium inputs use the compact
  // inline multiply stream so a DLL call does not dominate them. The long
  // dispatcher chooses wyhash or a runtime-safe wide XXH3 kernel. Vendor headers
  // stay hidden and hashes remain repeatable for this process.
  if (len <= 32) {
#if HYLE_ARCH_ARM64
    return detail::hash_bytes_medium_inline(data, len, seed);
#else
    return detail::hash_medium_17_32(data, len, seed);
#endif
  }
  if (len <= detail::kWyhashThreshold)
    return detail::hash_bytes_medium_inline(data, len, seed);
  return detail::hash_bytes_long(data, len, seed);
}

// SipHash-2-4 with a caller-supplied 128-bit key. This is the explicit hardened
// path for attacker-controlled byte/string keys. The key must be unpredictable
// to the attacker; HYLE does not silently turn a clock/ASLR seed into a claimed
// cryptographic secret. Output is a table hash, not a MAC protocol contract.
HYLE_CONTAINER_API std::uint64_t siphash24(const void* data, std::size_t len,
                                           std::uint64_t key0,
                                           std::uint64_t key1) noexcept;

// Explicit bytewise opt-in for keys whose equality is exactly their object
// representation. Unique object representations exclude padding and alternate
// encodings, but the trait cannot prove that a user-defined Eq compares every
// byte. Callers remain responsible for Eq(a, b) => identical object bytes; types
// with field-wise or otherwise broader equality must use a semantic hasher.
template <class T>
struct BytewiseHash {
  static_assert(std::is_trivially_copyable_v<T>,
                "BytewiseHash<T> requires a trivially-copyable T");
  static_assert(std::has_unique_object_representations_v<T>,
                "BytewiseHash<T> requires a unique object representation; this "
                "does not prove compatibility with a custom equality relation");

  std::uint64_t operator()(const T& v) const noexcept {
    // sizeof(T) is a COMPILE-TIME constant, so this whole ladder resolves at compile
    // time (if constexpr): each BytewiseHash<T> instantiation calls exactly ONE
    // hasher with ZERO runtime branch -- the best-of-breed pick is baked in per
    // type.
    if constexpr (sizeof(T) <= 16) {
      return detail::hash_small(&v, sizeof(T));           // inline mix, no call
    } else if constexpr (sizeof(T) <= 32) {
#if HYLE_ARCH_ARM64
      return detail::hash_bytes_medium_inline(&v, sizeof(T), 0);
#else
      return detail::hash_medium_17_32(&v, sizeof(T), 0);
#endif
    } else if constexpr (sizeof(T) <= detail::kWyhashThreshold) {
      return detail::hash_bytes_medium_inline(&v, sizeof(T), 0);
    } else {
      return detail::hash_bytes_long(&v, sizeof(T), 0);   // long: runtime best-of
    }
  }
};

// Default hasher. A type with a valid noexcept std::hash specialization already
// has a semantic, equality-compatible hashing contract, so bridge it through
// HYLE's final avalanche. Other user-defined types remain unavailable: neither
// trivial copyability nor unique object representations proves compatibility
// with that type's equality relation. Supply a semantic hasher (or specialize
// Hash<T>), or explicitly opt in to BytewiseHash<T> when equality is bytewise.
template <class T, class = void>
struct Hash {
  std::uint64_t operator()(const T& value) const noexcept
    requires requires {
      { std::hash<T>{}(value) } noexcept -> std::convertible_to<std::size_t>;
    }
  {
    return detail::mix64(
        static_cast<std::uint64_t>(std::hash<T>{}(value)));
  }
};

// Integral / enum: inline mix, no out-of-line call.
template <class T>
struct Hash<T, std::enable_if_t<std::is_integral_v<T> || std::is_enum_v<T>>> {
  std::uint64_t operator()(T v) const noexcept {
    return detail::mix64(static_cast<std::uint64_t>(v));
  }
};

// IEEE float/double have one equality-compatible bit representation after
// canonicalizing signed zero. Hash those bits directly so the table hot path is
// the same single inline mix as an integer key. NaNs are not equal and need no
// canonicalization.
template <>
struct Hash<float> {
  std::uint64_t operator()(float v) const noexcept {
    static_assert(std::numeric_limits<float>::is_iec559 &&
                  sizeof(float) == sizeof(std::uint32_t));
    const std::uint32_t bits = v == 0.0f
                                   ? 0u
                                   : std::bit_cast<std::uint32_t>(v);
    return detail::mix64(bits);
  }
};

template <>
struct Hash<double> {
  std::uint64_t operator()(double v) const noexcept {
    static_assert(std::numeric_limits<double>::is_iec559 &&
                  sizeof(double) == sizeof(std::uint64_t));
    const std::uint64_t bits = v == 0.0
                                   ? 0u
                                   : std::bit_cast<std::uint64_t>(v);
    return detail::mix64(bits);
  }
};

// long double has platform-specific width, padding, and encodings. Keep its
// equality semantics delegated to the standard library and only canonicalize
// signed zero before HYLE's final avalanche.
template <>
struct Hash<long double> {
  std::uint64_t operator()(long double v) const noexcept {
    if (v == 0.0L) v = 0.0L;
    return detail::mix64(
        static_cast<std::uint64_t>(std::hash<long double>{}(v)));
  }
};

// Pointers: mix the address (drop the always-zero low alignment bits first).
template <class T>
struct Hash<T*> {
  std::uint64_t operator()(T* p) const noexcept {
    return detail::mix64(reinterpret_cast<std::uintptr_t>(p));
  }
};

namespace detail {

struct StringSpan {
  const char* data;
  std::size_t size;
};

inline StringSpan as_string_span(std::string_view value) noexcept {
  return {value.data(), value.size()};
}

template <class P>
  requires(std::is_pointer_v<P> &&
           std::convertible_to<P, const char*> &&
           std::same_as<std::remove_cv_t<std::remove_pointer_t<P>>, char>)
inline StringSpan as_string_span(const P& value) noexcept {
  HYLE_FATAL_ASSERT(value != nullptr);
  return {value, std::char_traits<char>::length(value)};
}

template <std::size_t N>
inline StringSpan as_string_span(const char (&value)[N]) noexcept {
  static_assert(N != 0);
  HYLE_FATAL_ASSERT(value[N - 1] == '\0');
  return {value, std::char_traits<char>::length(value)};
}

template <class R>
concept MethodStringRange = requires(const R& value) {
  { value.data() } noexcept -> std::convertible_to<const char*>;
  { value.size() } noexcept -> std::convertible_to<std::size_t>;
};

template <MethodStringRange R>
inline StringSpan as_string_span(const R& value) noexcept {
  const auto size = static_cast<std::size_t>(value.size());
  const char* data = value.data();
  HYLE_FATAL_ASSERT(data != nullptr || size == 0);
  return {data, size};
}

template <class R>
concept FieldStringRange = requires(const R& value) {
  { value.data } noexcept -> std::convertible_to<const char*>;
  { value.size } noexcept -> std::convertible_to<std::size_t>;
};

template <class R>
  requires(!MethodStringRange<R> && FieldStringRange<R>)
inline StringSpan as_string_span(const R& value) noexcept {
  const auto size = static_cast<std::size_t>(value.size);
  const char* data = value.data;
  HYLE_FATAL_ASSERT(data != nullptr || size == 0);
  return {data, size};
}

}  // namespace detail

// Public transparent helper for trusted/general-purpose owning-string tables.
// Every default-constructed instance caches the same best-effort process seed
// and feeds it into the byte core (not merely a final XOR), invalidating offline
// collision sets across starts. This is not a cryptographic/adaptive-attack
// promise; use HardenedStringHash with a caller-managed secret for that case.
// Explicit StringHash(seed) selects a reproducible deterministic seed.
// A post-initialization fork inherits the parent's process seed; it is not an
// automatic key-rotation boundary.
//
// Queries may be std::string/string_view, StringView-like {data,size} views, or
// NUL-terminated C strings. Bytes are inspected only during the call and no
// borrowed pointer is retained. Use a sized view for embedded NUL bytes.
struct StringHash {
  using is_transparent = void;

  StringHash() noexcept : seed_(detail::process_hash_seed()) {}
  explicit constexpr StringHash(std::uint64_t seed) noexcept : seed_(seed) {}

  template <class S>
    requires requires(const S& text) { detail::as_string_span(text); }
  std::uint64_t operator()(const S& text) const noexcept {
    const detail::StringSpan span = detail::as_string_span(text);
    return hash_bytes(span.data, span.size, seed_);
  }

 private:
  std::uint64_t seed_;
};

struct StringEqual {
  using is_transparent = void;

  template <class L, class R>
    requires requires(const L& left, const R& right) {
      detail::as_string_span(left);
      detail::as_string_span(right);
    }
  bool operator()(const L& left, const R& right) const noexcept {
    const detail::StringSpan lhs = detail::as_string_span(left);
    const detail::StringSpan rhs = detail::as_string_span(right);
    return lhs.size == rhs.size &&
           (lhs.size == 0 || std::memcmp(lhs.data, rhs.data, lhs.size) == 0);
  }
};

// Explicit SipHash-2-4 policy for attacker-controlled string keys. Callers must
// inject an unpredictable 128-bit key and keep it secret for the table's
// lifetime. The hasher owns only the key; queried character storage is borrowed
// for the duration of operator() exactly like StringHash.
struct HardenedStringHash {
  using is_transparent = void;

  constexpr HardenedStringHash(std::uint64_t key0,
                               std::uint64_t key1) noexcept
      : key0_(key0), key1_(key1) {}

  template <class S>
    requires requires(const S& text) { detail::as_string_span(text); }
  std::uint64_t operator()(const S& text) const noexcept {
    const detail::StringSpan span = detail::as_string_span(text);
    return siphash24(span.data, span.size, key0_, key1_);
  }

 private:
  std::uint64_t key0_;
  std::uint64_t key1_;
};

// Standard string keys/views use the process-seeded fast policy by default.
// Select StringHash + StringEqual explicitly for heterogeneous view lookup, or
// HardenedStringHash + StringEqual for attacker-controlled keys.
template <>
struct Hash<std::string_view> : StringHash {
  using StringHash::StringHash;
};

template <>
struct Hash<std::string> : StringHash {
  using StringHash::StringHash;
};

// Ordered semantic composition for common metadata keys. Each field is hashed
// through its own Hash<T> policy and visited once; no bytes, padding, or object
// representation are inferred. hash_combine is public so user-defined semantic
// hashers can use the same route without specializing std::hash.
inline std::uint64_t hash_combine(std::uint64_t state,
                                  std::uint64_t value) noexcept {
  constexpr std::uint64_t kStep = 0x9e3779b97f4a7c15ull;
  return detail::mix64_2(state ^ kStep, value + kStep);
}

namespace detail {
template <class... T>
inline std::uint64_t hash_tuple_fields(const std::tuple<T...>& value) noexcept {
  std::uint64_t state = mix64(sizeof...(T));
  std::apply(
      [&](const T&... field) {
        ((state = hash_combine(state, Hash<T>{}(field))), ...);
      },
      value);
  return state;
}
}  // namespace detail

template <class First, class Second>
struct Hash<std::pair<First, Second>> {
  std::uint64_t operator()(const std::pair<First, Second>& value) const noexcept
    requires(std::is_nothrow_invocable_r_v<std::uint64_t, Hash<First>,
                                            const First&> &&
             std::is_nothrow_invocable_r_v<std::uint64_t, Hash<Second>,
                                            const Second&>)
  {
    std::uint64_t state = detail::mix64(2);
    state = hash_combine(state, Hash<First>{}(value.first));
    return hash_combine(state, Hash<Second>{}(value.second));
  }
};

template <class... T>
struct Hash<std::tuple<T...>> {
  std::uint64_t operator()(const std::tuple<T...>& value) const noexcept
    requires((std::is_nothrow_invocable_r_v<std::uint64_t, Hash<T>,
                                             const T&> && ...))
  {
    return detail::hash_tuple_fields(value);
  }
};

template <class T, std::size_t N>
struct Hash<std::array<T, N>> {
  std::uint64_t operator()(const std::array<T, N>& value) const noexcept
    requires std::is_nothrow_invocable_r_v<std::uint64_t, Hash<T>, const T&>
  {
    std::uint64_t state = detail::mix64(N);
    for (const T& field : value) {
      state = hash_combine(state, Hash<T>{}(field));
    }
    return state;
  }
};

// A type usable as a hash key: Hash<T> yields a uint64 for it.
template <class T>
concept Hashable = requires(const T& v) {
  { Hash<T>{}(v) } -> std::convertible_to<std::uint64_t>;
};

}  // namespace hyle::container
