#pragma once

// Philox4x32-10 equations and constants from Random123 v1.14.0.
// Copyright 2010-2012 D. E. Shaw Research. BSD-3-Clause; see LICENSE.
// HYLE adapts only the fixed counter block and integer addressing, without
// mutable generator state or the upstream portability macro framework.
#include <cstdint>

#if defined(__CUDACC__)
#define HYLE_RANDOM123_INLINE __host__ __device__ __forceinline__
#else
#define HYLE_RANDOM123_INLINE inline
#endif

namespace hyle::borrowed::random123_counter {
struct Words final {
  std::uint32_t x0, x1, x2, x3;
};

HYLE_RANDOM123_INLINE Words philox(Words value, std::uint32_t key0,
                                   std::uint32_t key1) noexcept {
  for (unsigned round = 0; round < 10; ++round) {
    const auto first = std::uint64_t{0xd2511f53U} * value.x0;
    const auto second = std::uint64_t{0xcd9e8d57U} * value.x2;
    value = {static_cast<std::uint32_t>(second >> 32) ^ value.x1 ^ key0,
             static_cast<std::uint32_t>(second),
             static_cast<std::uint32_t>(first >> 32) ^ value.x3 ^ key1,
             static_cast<std::uint32_t>(first)};
    key0 += 0x9e3779b9U;
    key1 += 0xbb67ae85U;
  }
  return value;
}

HYLE_RANDOM123_INLINE Words block(const std::uint32_t *identity,
                                   std::uint64_t index) noexcept {
  return philox({static_cast<std::uint32_t>(index),
                 static_cast<std::uint32_t>(index >> 32), identity[2],
                 identity[3]}, identity[0], identity[1]);
}

HYLE_RANDOM123_INLINE std::uint32_t lane(Words value, unsigned index) noexcept {
  switch (index) {
  case 0: return value.x0;
  case 1: return value.x1;
  case 2: return value.x2;
  default: return value.x3;
  }
}

HYLE_RANDOM123_INLINE std::uint32_t word(const std::uint32_t *identity,
                                         std::uint64_t index) noexcept {
  return lane(block(identity, index / 4), static_cast<unsigned>(index % 4));
}
} // namespace hyle::borrowed::random123_counter

#undef HYLE_RANDOM123_INLINE
