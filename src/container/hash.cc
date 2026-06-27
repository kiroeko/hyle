#include "hyle/container/hash.h"

// XXH3 lives entirely here: XXH_INLINE_ALL pulls the whole (header-only) xxHash
// implementation into this one translation unit, so the 7490-line header never
// reaches a public HYLE header or a consumer TU.
#define XXH_INLINE_ALL
#include "xxhash.h"

namespace hyle::container {

std::uint64_t hash_bytes(const void* data, std::size_t len,
                         std::uint64_t seed) noexcept {
  return XXH3_64bits_withSeed(data, len, seed);
}

}  // namespace hyle::container
