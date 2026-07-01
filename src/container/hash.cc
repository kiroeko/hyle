#include "hyle/container/hash.h"

// XXH3 lives entirely here: XXH_INLINE_ALL pulls the whole (header-only) xxHash
// implementation into this one translation unit, so the 7490-line header never
// reaches a public HYLE header or a consumer TU.
#define XXH_INLINE_ALL
#include "xxhash.h"

// wyhash (public domain, single header) -- the faster hasher for LONG byte ranges.
// Kept in this TU too (out of the public header), same seam discipline as XXH3.
#include "wyhash.h"

namespace hyle::container::detail {

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

}  // namespace hyle::container::detail
