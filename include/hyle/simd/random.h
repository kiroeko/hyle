#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dispatch.h"  // HYLE_SIMD_API

// Runtime-safe random generation. Unlike native::VectorXoshiro, this header exposes no
// target-dependent vector type: RandomState has one fixed layout and the fill functions have a
// pointer/POD ABI, so a prebuilt HYLE library can select the best compiled kernel at runtime.
//
// The state contains eight jump-separated xoshiro256++ streams. Output is round-robin by stream,
// and every ISA updates the same eight logical streams in the same order. Therefore a given seed
// and sequence of fill calls is bit-identical across scalar/SSE4/AVX2/AVX-512/NEON.
//
// This generator is NOT cryptographically secure. Do not use it for keys, tokens or nonces.

namespace hyle::simd {

namespace detail {

struct RandomStateAccess;

constexpr std::uint64_t random_rotl(std::uint64_t x, int k) noexcept {
  return (x << k) | (x >> (64 - k));
}

constexpr std::uint64_t random_splitmix64(std::uint64_t& state) noexcept {
  std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

constexpr std::uint64_t random_xoshiro_step(std::uint64_t& s0, std::uint64_t& s1,
                                             std::uint64_t& s2,
                                             std::uint64_t& s3) noexcept {
  const std::uint64_t result = random_rotl(s0 + s3, 23) + s0;
  const std::uint64_t t = s1 << 17;
  s2 ^= s0;
  s3 ^= s1;
  s1 ^= s2;
  s0 ^= s3;
  s2 ^= t;
  s3 = random_rotl(s3, 45);
  return result;
}

constexpr void random_xoshiro_jump(std::uint64_t state[4]) noexcept {
  constexpr std::uint64_t kJump[4] = {
      0x180ec6d33cfd0abaull, 0xd5a61266f0c9392cull,
      0xa9582618e03fc9aaull, 0x39abdc4529b1661cull};
  std::uint64_t out[4] = {};
  for (std::size_t i = 0; i < 4; ++i) {
    const std::uint64_t word = kJump[i];
    for (int bit = 0; bit < 64; ++bit) {
      if ((word & (std::uint64_t{1} << bit)) != 0) {
        out[0] ^= state[0];
        out[1] ^= state[1];
        out[2] ^= state[2];
        out[3] ^= state[3];
      }
      (void)random_xoshiro_step(state[0], state[1], state[2], state[3]);
    }
  }
  for (std::size_t i = 0; i < 4; ++i) state[i] = out[i];
}

}  // namespace detail

class alignas(64) RandomState {
 public:
  static constexpr std::uint32_t kLayoutVersion = 1;
  static constexpr std::size_t kStreams = 8;

  explicit constexpr RandomState(
      std::uint64_t seed = 0xd1b54a32d192ed03ull) noexcept {
    std::uint64_t scalar[4];
    for (std::uint64_t& word : scalar) word = detail::random_splitmix64(seed);
    for (std::size_t stream = 0; stream < kStreams; ++stream) {
      for (std::size_t word = 0; word < 4; ++word) state_[word][stream] = scalar[word];
      detail::random_xoshiro_jump(scalar);
    }
  }

 private:
  friend struct detail::RandomStateAccess;
  std::uint64_t state_[4][kStreams]{};
  [[maybe_unused]] std::uint32_t layout_version_ = kLayoutVersion;
  std::uint32_t cursor_ = 0;
  [[maybe_unused]] std::uint64_t reserved_[7]{};
};

static_assert(sizeof(RandomState) == 320);
static_assert(alignof(RandomState) == 64);

namespace detail {

// Internal bridge for separately compiled native kernels. RandomState remains
// opaque to callers so its C++ layout is not part of the public source contract.
struct RandomStateAccess {
  static std::uint64_t (&words(RandomState& state) noexcept)[4][RandomState::kStreams] {
    return state.state_;
  }
  static std::uint32_t& cursor(RandomState& state) noexcept { return state.cursor_; }
};

}  // namespace detail

HYLE_SIMD_API void random_fill(RandomState& state, std::uint64_t* dst,
                               std::size_t count) noexcept;
HYLE_SIMD_API void random_fill(RandomState& state, float* dst,
                               std::size_t count) noexcept;
HYLE_SIMD_API void random_fill(RandomState& state, double* dst,
                               std::size_t count) noexcept;

}  // namespace hyle::simd
