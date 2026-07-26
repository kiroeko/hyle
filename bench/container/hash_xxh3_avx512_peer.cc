#include <cstddef>
#include <cstdint>

#define XXH_INLINE_ALL
#define XXH_VECTOR XXH_AVX512
#include "xxhash.h"

std::uint64_t hyle_bench_xxh3_avx512(const void* data, std::size_t size,
                                     std::uint64_t seed) noexcept {
  return XXH3_64bits_withSeed(data, size, seed);
}
