#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>

#include "hyle/core/compiler.h"

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

struct RngState {
  std::uint64_t words[4];
};

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
#if defined(_MSC_VER) && defined(_M_X64)
  std::uint64_t hi;
  lo = _umul128(a, b, &hi);
  return hi;
#elif defined(_MSC_VER) && defined(_M_ARM64)
  lo = a * b;
  return __umulh(a, b);
#elif defined(__GNUC__) || defined(__clang__)
  __extension__ using u128 = unsigned __int128;  // silence -Wpedantic
  const u128 p = static_cast<u128>(a) * b;
  lo = static_cast<std::uint64_t>(p);
  return static_cast<std::uint64_t>(p >> 64);
#else
  // Portable 32-bit limb fallback for uncommon compilers without a native
  // 64x64->128 type/intrinsic. Shipping 64-bit toolchains take a branch above.
  const std::uint64_t a0 = static_cast<std::uint32_t>(a);
  const std::uint64_t a1 = a >> 32;
  const std::uint64_t b0 = static_cast<std::uint32_t>(b);
  const std::uint64_t b1 = b >> 32;
  const std::uint64_t p00 = a0 * b0;
  const std::uint64_t p01 = a0 * b1;
  const std::uint64_t p10 = a1 * b0;
  const std::uint64_t p11 = a1 * b1;
  const std::uint64_t middle =
      (p00 >> 32) + static_cast<std::uint32_t>(p01) +
      static_cast<std::uint32_t>(p10);
  lo = (middle << 32) | static_cast<std::uint32_t>(p00);
  return p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32);
#endif
}

inline std::atomic<std::uint64_t> g_thread_seed_sequence{
    0xA0761D6478BD642Full};

// One relaxed RMW and one best-effort std::random_device seed per thread, only
// at its first thread_rng() call. The C++ standard does not require
// random_device to be backed by OS entropy, and the inline sequence is not a
// cross-DSO stream-identity service. Together with an address salt these inputs
// make accidental stream reuse unlikely; they do not guarantee unique or
// disjoint streams. None of this makes xoshiro cryptographically secure.
inline std::uint64_t next_thread_seed() noexcept {
  constexpr std::uint64_t kStep = 0x9E3779B97F4A7C15ull;
  const std::uint64_t sequence =
      g_thread_seed_sequence.fetch_add(kStep, std::memory_order_relaxed);
  std::uint64_t entropy = 0;
  try {
    std::random_device source;
    entropy = static_cast<std::uint64_t>(source());
    entropy = (entropy << 32) ^ static_cast<std::uint64_t>(source());
  } catch (...) {
    // random_device construction/call is allowed to throw. thread_rng() is a
    // noexcept convenience path, so retain a varying but lower-entropy seed.
  }
  const auto process_salt = static_cast<std::uint64_t>(
      reinterpret_cast<std::uintptr_t>(&g_thread_seed_sequence));
  return sequence ^ detail::rotl(entropy, 17) ^ process_salt;
}
}  // namespace detail

class Rng {
 public:
  // UniformRandomBitGenerator surface. This lets Rng compose directly with
  // std::shuffle and standard distributions instead of requiring HYLE-specific
  // adapters. The named next_* helpers remain the convenient typed surface.
  using result_type = std::uint64_t;
  static constexpr result_type min() noexcept { return 0; }
  static constexpr result_type max() noexcept {
    return std::numeric_limits<result_type>::max();
  }

  // Seed from a single u64 (SplitMix64-expanded into the 256-bit state).
  explicit constexpr Rng(
      std::uint64_t seed = 0xD1B54A32D192ED03ull) noexcept {
    reseed(seed);
  }

  constexpr void reseed(std::uint64_t seed) noexcept {
    for (std::uint64_t& word : state_.words) {
      word = detail::splitmix64(seed);
    }
  }

  static constexpr bool valid_state(const RngState& state) noexcept {
    return (state.words[0] | state.words[1] | state.words[2] |
            state.words[3]) != 0;
  }

  RngState state() const noexcept { return state_; }

  void set_state(const RngState& state) noexcept {
    HYLE_FATAL_ASSERT(valid_state(state));
    state_ = state;
  }

  static Rng from_state(const RngState& state) noexcept {
    Rng rng;
    rng.set_state(state);
    return rng;
  }

  // Next 64-bit value. xoshiro256++: the "++" output scrambler (rotl(s0+s3,23)
  // + s0) gives good high AND low bits.
  HYLE_FORCE_INLINE std::uint64_t next_u64() noexcept {
    auto& s = state_.words;
    const std::uint64_t result = detail::rotl(s[0] + s[3], 23) + s[0];
    const std::uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = detail::rotl(s[3], 45);
    return result;
  }

  HYLE_FORCE_INLINE result_type operator()() noexcept { return next_u64(); }

  // Standard engines expose discard(n). It is intentionally O(n); use jump()
  // or long_jump() for the fixed 2^128 / 2^192 parallel-stream leaps.
  void discard(std::uint64_t count) noexcept {
    while (count-- != 0) (void)next_u64();
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
    HYLE_FATAL_ASSERT(lo <= hi);
    const std::uint32_t span = hi - lo;
    if (span == UINT32_MAX) return next_u32();
    const std::uint32_t n = span + 1;
    std::uint64_t product =
        static_cast<std::uint64_t>(next_u32()) * n;
    std::uint32_t low = static_cast<std::uint32_t>(product);
    if (low < n) {
      const std::uint32_t threshold =
          static_cast<std::uint32_t>(0u - n) % n;
      while (low < threshold) {
        product = static_cast<std::uint64_t>(next_u32()) * n;
        low = static_cast<std::uint32_t>(product);
      }
    }
    return lo + static_cast<std::uint32_t>(product >> 32);
  }
  // Double in [lo, hi).
  double range_double(double lo, double hi) noexcept {
    HYLE_FATAL_ASSERT(std::isfinite(lo));
    HYLE_FATAL_ASSERT(std::isfinite(hi));
    HYLE_FATAL_ASSERT(lo <= hi);
    if (lo == hi) return lo;
    const double value = std::lerp(lo, hi, next_double());
    // Floating rounding may otherwise produce the exclusive upper endpoint.
    return value < hi ? value : std::nextafter(hi, lo);
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

  // Natural C++ range spelling. Keep the count+swap overload as the storage-
  // agnostic callback primitive, but do not make ordinary users
  // manufacture index callbacks just to shuffle a contiguous/random-access
  // range. This preserves HYLE's fast Lemire map instead of routing through a
  // standard-library implementation with different bounded-draw costs.
  template <std::random_access_iterator Iterator>
  void shuffle(Iterator first, Iterator last) {
    const auto distance = last - first;
    HYLE_FATAL_ASSERT(distance >= 0);
    shuffle(static_cast<std::uint64_t>(distance),
            [&](std::uint64_t left, std::uint64_t right) {
              using Difference = std::iter_difference_t<Iterator>;
              std::iter_swap(first + static_cast<Difference>(left),
                             first + static_cast<Difference>(right));
            });
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
          s0 ^= state_.words[0];
          s1 ^= state_.words[1];
          s2 ^= state_.words[2];
          s3 ^= state_.words[3];
        }
        next_u64();
      }
    }
    state_.words[0] = s0;
    state_.words[1] = s1;
    state_.words[2] = s2;
    state_.words[3] = s3;
  }

  // Advance 2^192 steps. Use this to split a seed into large groups, then
  // jump() within each group for per-worker 2^128 substreams.
  void long_jump() noexcept {
    static constexpr std::uint64_t kLongJump[4] = {
        0x76e15d3efefdcbbfull, 0xc5004e441c522fb3ull,
        0x77710069854ee241ull, 0x39109bb02acbe635ull};
    std::uint64_t next[4] = {};
    for (std::uint64_t jump_word : kLongJump) {
      for (int bit = 0; bit < 64; ++bit) {
        if (jump_word & (std::uint64_t{1} << bit)) {
          for (int lane = 0; lane < 4; ++lane) {
            next[lane] ^= state_.words[lane];
          }
        }
        next_u64();
      }
    }
    for (int lane = 0; lane < 4; ++lane) {
      state_.words[lane] = next[lane];
    }
  }

 private:
  RngState state_;
};

// Thread-local default generator: a per-thread Rng so the common case needs no
// instance and no steady-state locking. Default streams are convenience-only:
// use an explicit seed plus jump()/long_jump() when replay or deliberate stream
// partitioning matters.
inline Rng& thread_rng() noexcept {
  static thread_local Rng rng{detail::next_thread_seed()};
  return rng;
}

}  // namespace hyle::util
