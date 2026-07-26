#include <cstddef>
#include <cstdint>

// Force xxHash's 256-bit implementation into this isolated TU. The baseline
// dispatcher calls it only after platform's CPUID+XGETBV gate confirms AVX2.
#define XXH_INLINE_ALL
#define XXH_VECTOR XXH_AVX2
#include "xxhash.h"

namespace hyle::container::detail {

std::uint64_t hash_bytes_xxh3_avx2(const void* data, std::size_t len,
                                   std::uint64_t seed) noexcept {
  return XXH3_64bits_withSeed(data, len, seed);
}

}  // namespace hyle::container::detail
