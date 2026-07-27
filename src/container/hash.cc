#include "hyle/container/hash.h"

#include "hyle/platform/cpu.h"

// XXH3 lives entirely here: XXH_INLINE_ALL pulls the whole (header-only) xxHash
// implementation into this one translation unit, so the 7490-line header never
// reaches a public HYLE header or a consumer TU.
#define XXH_INLINE_ALL
#include "xxhash.h"

// wyhash (public domain, single header) -- the faster hasher for LONG byte ranges.
// Kept in this TU too (out of the public header), same seam discipline as XXH3.
#include "wyhash.h"

namespace hyle::container::detail {

#if defined(_M_X64) || defined(__x86_64__)
std::uint64_t hash_bytes_xxh3_avx2(const void* data, std::size_t len,
                                   std::uint64_t seed) noexcept;
#endif

// XXH3 core for 17..kWyhashThreshold bytes. Small inputs (<=16B) are handled inline
// in hash.h; long inputs (> kWyhashThreshold) go to wyhash below. XXH3 is fastest in
// the short-mid range (measured: wins wyhash ~1.1-1.26x for 17-96B, tie ~128B).
std::uint64_t hash_bytes_xxh3(const void* data, std::size_t len,
                              std::uint64_t seed) noexcept {
  return XXH3_64bits_withSeed(data, len, seed);
}

// wyhash core for LONG byte ranges (> kWyhashThreshold). Measured to beat XXH3
// 1.45-2.6x for >=192B (XXH3 has a midsize-path pessimum near 256B; wyhash's
// 48B/loop 3-accumulator streaming pulls ahead). Same 64-bit quality class as XXH3
// (both pass SMHasher; wyhash is the default hasher of Go 1.17+/Zig/Nim). `seed`
// feeds wyhash's seed arg; the default secret `_wyp` is used (dynamic-secret salting
// is a security feature we don't need for an in-process hashmap).
std::uint64_t hash_bytes_wyhash(const void* data, std::size_t len,
                                std::uint64_t seed) noexcept {
  return wyhash(data, len, seed, _wyp);
}

std::uint64_t hash_bytes_long(const void* data, std::size_t len,
                              std::uint64_t seed) noexcept {
#if defined(_M_X64) || defined(__x86_64__)
  // The detector is compiled at baseline ISA in platform/cpu.cc. Cache the result
  // once; the branch and out-of-line call are amortized by the platform-tuned
  // long-input crossover.
  static const bool has_avx2 =
      platform::supports(platform::Isa::kAvx2);
  if (len >= kWideHashThreshold && has_avx2) {
    return hash_bytes_xxh3_avx2(data, len, seed);
  }
#elif defined(HYLE_ARCH_ARM64)
  // NEON is part of the AArch64 baseline, so the XXH3 TU already contains its
  // wide product path and needs no runtime feature branch. Varied-input M4
  // sweeps put the first durable crossover at 512B; below it wyhash remains the
  // better long-range choice.
  if (len >= kWideHashThreshold)
    return hash_bytes_xxh3(data, len, seed);
#endif
  return hash_bytes_wyhash(data, len, seed);
}

}  // namespace hyle::container::detail
