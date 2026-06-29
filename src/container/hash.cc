#include "hyle/container/hash.h"

// XXH3 lives entirely here: XXH_INLINE_ALL pulls the whole (header-only) xxHash
// implementation into this one translation unit, so the 7490-line header never
// reaches a public HYLE header or a consumer TU.
#define XXH_INLINE_ALL
#include "xxhash.h"

namespace hyle::container::detail {

// XXH3 core for len > 16 (small inputs are handled inline in hash.h, never reach
// here). Kept out-of-line so xxhash.h stays out of every consumer TU.
std::uint64_t hash_bytes_xxh3(const void* data, std::size_t len,
                              std::uint64_t seed) noexcept {
  return XXH3_64bits_withSeed(data, len, seed);
}

}  // namespace hyle::container::detail
