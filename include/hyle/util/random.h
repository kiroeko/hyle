#pragma once

#include <cstdint>

#include "hyle/macros.h"

#if defined(_MSC_VER)
#  include <intrin.h>  // _umul128 for the 64x64->128 multiply in Lemire's map
#endif

// Fast PRNG (clay random.h used xoroshiro128+). HYLE uses xoshiro256++ -- the
// recommended general-purpose successor (better statistical quality, same speed
// class, 256-bit state). NOT cryptographic. Header-only inline: an RNG is a hot
// path, so every call must inline to a few instructions with no DLL hop.
//
// Improvements over clay:
//  - xoshiro256++ (clay: xoroshiro128+, whose low bits are weak).
//  - SplitMix64 seeding from a single u64 (avoids the all-zero state that breaks
//    xoshiro; clay left seeding to the caller).
//  - UNBIASED bounded range via Lemire's method (clay used `% (max+1-min)`,
//    which is biased unless the range divides 2^64).
//  - jump(): advance 2^128 steps for non-overlapping parallel streams (give each
//    sched worker its own).

namespace hyle::util {

namespace detail {
HYLE_FORCE_INLINE constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept {
  return (x << k) | (x >> (64 - k));
}
// SplitMix64: maps one seed to a well-distributed stream; used to fill the
// xoshiro state so a single user seed never yields the forbidden all-zero state.
HYLE_FORCE_INLINE constexpr std::uint64_t splitmix64(std::uint64_t& s) noexcept {
  std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// 64x64 -> 128 multiply, returning hi and lo halves. The core of Lemire's
// nearly-divisionless bounded-int map (absl/numpy/rust rand all use this).
// MSVC: _umul128 intrinsic; GCC/Clang: __uint128_t (lowered to one `mul`).
HYLE_FORCE_INLINE std::uint64_t mul128_hi_lo(std::uint64_t a, std::uint64_t b,
                                             std::uint64_t& lo) noexcept {
#if defined(_MSC_VER)
  std::uint64_t hi;
  lo = _umul128(a, b, &hi);
  return hi;
#else
  __extension__ using u128 = unsigned __int128;  // silence -Wpedantic
  const u128 p = static_cast<u128>(a) * b;
  lo = static_cast<std::uint64_t>(p);
  return static_cast<std::uint64_t>(p >> 64);
#endif
}
}  // namespace detail

class Rng {
 public:
  // Seed from a single u64 (SplitMix64-expanded into the 256-bit state).
  explicit constexpr Rng(std::uint64_t seed = 0xD1B54A32D192ED03ull) noexcept {
    for (std::uint64_t& w : s_) w = detail::splitmix64(seed);
  }

  // Next 64-bit value. xoshiro256++: the "++" output scrambler (rotl(s0+s3,23)
  // + s0) gives good high AND low bits.
  HYLE_FORCE_INLINE std::uint64_t next_u64() noexcept {
    const std::uint64_t result = detail::rotl(s_[0] + s_[3], 23) + s_[0];
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = detail::rotl(s_[3], 45);
    return result;
  }

  std::uint32_t next_u32() noexcept {
    return static_cast<std::uint32_t>(next_u64() >> 32);  // high bits
  }
  bool next_bool() noexcept { return (next_u64() >> 63) != 0; }

  // Double in [0, 1): top 53 bits scaled (the canonical construction).
  double next_double() noexcept {
    return static_cast<double>(next_u64() >> 11) * 0x1.0p-53;
  }
  float next_float() noexcept {
    return static_cast<float>(next_u64() >> 40) * 0x1.0p-24f;
  }

  // Unbiased integer in [lo, hi] inclusive via Lemire's NEARLY-DIVISIONLESS map
  // (Lemire 2019; the method absl/numpy/rust rand ship). The map x -> hi(x*n) is
  // unbiased once a rejection zone of size 2^64 % n is excluded. The trick that
  // makes it "nearly divisionless": the common-case check uses `n` itself as a
  // cheap OVERESTIMATE of the threshold (one 128-bit multiply, ZERO divisions);
  // only when lo(x*n) < n -- probability < n/2^64, essentially never for normal
  // ranges -- do we compute the exact `2^64 % n` (one division) and resample.
  // The old form was modulo-rejection (two idivs EVERY call, ~15-40 cyc each);
  // this pays one `mul` on the hot path instead.
  std::uint64_t range_u64(std::uint64_t lo, std::uint64_t hi) noexcept {
    HYLE_FATAL_ASSERT(lo <= hi);
    std::uint64_t span = hi - lo;
    if (span == UINT64_MAX) return next_u64();  // full range, no rejection
    const std::uint64_t n = span + 1;
    std::uint64_t low;
    std::uint64_t m = detail::mul128_hi_lo(next_u64(), n, low);  // m = hi(x*n)
    if (low < n) {  // rare: only then is a divide needed to get the exact threshold
      const std::uint64_t threshold = (UINT64_MAX - n + 1) % n;  // == 2^64 % n
      while (low < threshold) {
        m = detail::mul128_hi_lo(next_u64(), n, low);
      }
    }
    return lo + m;  // hi(x*n) is uniform in [0, n)
  }
  std::uint32_t range_u32(std::uint32_t lo, std::uint32_t hi) noexcept {
    return static_cast<std::uint32_t>(range_u64(lo, hi));
  }
  // Double in [lo, hi).
  double range_double(double lo, double hi) noexcept {
    return lo + (hi - lo) * next_double();
  }

  // Unbiased Fisher-Yates shuffle of n elements via a swap callback.
  // swap(i, j) exchanges elements i and j (lets the caller own element storage).
  template <class Swap>
  void shuffle(std::uint64_t n, Swap&& swap) {
    for (std::uint64_t i = n; i > 1; --i) {
      std::uint64_t j = range_u64(0, i - 1);
      if (j != i - 1) swap(i - 1, j);
    }
  }

  // Advance 2^128 steps: two RNGs that then diverge produce non-overlapping
  // streams (give each parallel worker a jumped copy).
  void jump() noexcept {
    static constexpr std::uint64_t kJump[4] = {
        0x180ec6d33cfd0abaull, 0xd5a61266f0c9392cull, 0xa9582618e03fc9aaull,
        0x39abdc4529b1661cull};
    std::uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (std::uint64_t jw : kJump) {
      for (int b = 0; b < 64; ++b) {
        if (jw & (std::uint64_t{1} << b)) {
          s0 ^= s_[0];
          s1 ^= s_[1];
          s2 ^= s_[2];
          s3 ^= s_[3];
        }
        next_u64();
      }
    }
    s_[0] = s0;
    s_[1] = s1;
    s_[2] = s2;
    s_[3] = s3;
  }

 private:
  std::uint64_t s_[4];
};

// Thread-local default generator: a per-thread Rng so the common case needs no
// instance and no locking. Seeded from the address of a stack local captured at
// first use -- distinct per thread (different stacks), no extra includes.
inline Rng& thread_rng() noexcept {
  static thread_local Rng rng = [] {
    std::uint64_t marker;  // its address differs per thread's stack
    return Rng(reinterpret_cast<std::uint64_t>(&marker) ^
               0x2545F4914F6CDD1Dull);
  }();
  return rng;
}

}  // namespace hyle::util
