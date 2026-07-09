#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "hyle/simd/vec.h"

// random.h -- vectorized pseudo-random generation (simd-09-rng). N parallel xoshiro256**
// streams held one-per-lane: the state update is per-lane independent (Xor/ShiftLeft/rotate),
// so one vector step advances all lanes at once. Blackman-Vigna xoshiro256** is a public,
// well-studied generator (period 2^256-1, passes BigCrush); HYLE OWNS the algorithm at the
// generic layer (composed from Set/Add/Xor/ShiftLeft<k>/Rol<k>/ConvertTo) rather than
// vendoring Highway's contrib/random D-tag idiom -- "borrow the algorithm contract, not the
// library abstraction" (simd-design), same call as intdiv/cmp128.
//
// Streams are decorrelated by SEEDING each lane from a distinct SplitMix64 draw and then
// applying xoshiro's Jump() (2^128 steps) per lane index, so the N lanes are guaranteed
// non-overlapping subsequences (the standard parallel-stream construction). This makes the
// generator reproducible for a given seed AND safe to run many lanes in parallel.
//
// NOT cryptographically secure -- for Monte Carlo / dropout / init / sampling, not keys.

namespace hyle::simd {

namespace detail {
// Scalar SplitMix64: seeds the xoshiro state (the standard bootstrap, avoids all-zero state).
class SplitMix64 {
 public:
  constexpr explicit SplitMix64(std::uint64_t state) noexcept : state_(state) {}
  constexpr std::uint64_t operator()() noexcept {
    std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
 private:
  std::uint64_t state_;
};

// Scalar xoshiro256** -- used only to compute per-lane Jump()ed seed states at construction.
class Xoshiro {
 public:
  constexpr explicit Xoshiro(std::uint64_t seed) noexcept : s_{} {
    SplitMix64 sm{seed};
    for (auto& e : s_) e = sm();
  }
  constexpr std::array<std::uint64_t, 4> state() const noexcept { return {s_[0], s_[1], s_[2], s_[3]}; }
  // Jump: equivalent to 2^128 Next() calls -- non-overlapping subsequence per stream.
  constexpr void Jump() noexcept {
    static constexpr std::uint64_t kJump[4] = {0x180ec6d33cfd0abaull, 0xd5a61266f0c9392cull,
                                               0xa9582618e03fc9aaull, 0x39abdc4529b1661cull};
    std::uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (std::uint64_t jw : kJump)
      for (int b = 0; b < 64; ++b) {
        if (jw & (std::uint64_t{1} << b)) { s0 ^= s_[0]; s1 ^= s_[1]; s2 ^= s_[2]; s3 ^= s_[3]; }
        Next();
      }
    s_[0] = s0; s_[1] = s1; s_[2] = s2; s_[3] = s3;
  }
  constexpr std::uint64_t Next() noexcept {
    const std::uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }
 private:
  static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept { return (x << k) | (x >> (64 - k)); }
  std::uint64_t s_[4];
};
}  // namespace detail

// N parallel xoshiro256** streams (N = native u64 lanes). Build once with a seed, then pull
// vectors of random bits via operator()/Next(), or f64 in [0,1) via Uniform().
class VectorXoshiro {
  using D = ScalableTag<std::uint64_t>;
  using VU64 = VFromD<D>;

 public:
  explicit VectorXoshiro(std::uint64_t seed) noexcept {
    const D d;
    const std::size_t n = Lanes(d);
    // Per-lane state seeded by an independent Jump()ed scalar xoshiro (non-overlapping streams).
    alignas(64) std::uint64_t s0[D::kMaxLanes], s1[D::kMaxLanes], s2[D::kMaxLanes], s3[D::kMaxLanes];
    detail::Xoshiro x{seed};
    for (std::size_t lane = 0; lane < n; ++lane) {
      const auto st = x.state();
      s0[lane] = st[0]; s1[lane] = st[1]; s2[lane] = st[2]; s3[lane] = st[3];
      x.Jump();  // advance 2^128 so the next lane is a disjoint subsequence
    }
    state0_ = Load(d, s0); state1_ = Load(d, s1); state2_ = Load(d, s2); state3_ = Load(d, s3);
  }

  // One vectorized xoshiro256** step: advances all lanes, returns N random u64 (one per stream).
  VU64 operator()() noexcept { return Next(); }
  VU64 Next() noexcept {
    const D d;
    // result = rotl(s0 + s3, 23) + s0
    const VU64 result = Add(Rol<23>(Add(state0_, state3_)), state0_);
    const VU64 t = ShiftLeft<17>(state1_);
    state2_ = Xor(state2_, state0_);
    state3_ = Xor(state3_, state1_);
    state1_ = Xor(state1_, state2_);
    state0_ = Xor(state0_, state3_);
    state2_ = Xor(state2_, t);
    state3_ = Rol<45>(state3_);
    (void)d;
    return result;
  }

  // f64 in [0,1). Uses the bit-pattern trick (NO int->float convert, which isn't native SIMD
  // for u64->f64 pre-AVX512): take the top 52 bits as a mantissa, OR in the exponent of 1.0 to
  // form a double in [1,2), then subtract 1.0 -> [0,1). Uniform spacing 2^-52, all integer +
  // BitCast + one f64 Sub (all backends have these).
  VFromD<ScalableTag<double>> Uniform() noexcept {
    const ScalableTag<double> df;
    const ScalableTag<std::uint64_t> du;
    const VU64 mant = ShiftRight<12>(Next());              // top 52 bits
    const VU64 one_bits = Or(mant, Set(du, 0x3FF0000000000000ull));  // exponent of 1.0
    return Sub(BitCast(df, one_bits), Set(df, 1.0));       // [1,2) - 1 = [0,1)
  }

 private:
  VU64 state0_, state1_, state2_, state3_;
};

}  // namespace hyle::simd
