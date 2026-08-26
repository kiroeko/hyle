#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/core/compiler.h"
#include "hyle/simd/random.h"
#include "hyle/simd/vec.h"

namespace hyle::simd::native {
namespace detail {

static HYLE_FORCE_INLINE std::uint64_t random_step_stream(
    RandomState& state, std::size_t stream) noexcept {
  auto& words = ::hyle::simd::detail::RandomStateAccess::words(state);
  return ::hyle::simd::detail::random_xoshiro_step(
      words[0][stream], words[1][stream], words[2][stream], words[3][stream]);
}

static HYLE_FORCE_INLINE std::uint64_t random_step_one(RandomState& state) noexcept {
  auto& cursor = ::hyle::simd::detail::RandomStateAccess::cursor(state);
  const std::size_t stream = cursor;
  const std::uint64_t result = random_step_stream(state, stream);
  cursor = static_cast<std::uint32_t>((stream + 1) % RandomState::kStreams);
  return result;
}

static HYLE_FORCE_INLINE void random_block(RandomState& state,
                                           std::uint64_t* out) noexcept {
  const CappedTag<std::uint64_t, RandomState::kStreams> d;
  const std::size_t width = Lanes(d);
  auto& words = ::hyle::simd::detail::RandomStateAccess::words(state);
  std::size_t offset = 0;
  for (; offset + width <= RandomState::kStreams; offset += width) {
    auto s0 = LoadU(d, words[0] + offset);
    auto s1 = LoadU(d, words[1] + offset);
    auto s2 = LoadU(d, words[2] + offset);
    auto s3 = LoadU(d, words[3] + offset);
    const auto result = Add(Rol<23>(Add(s0, s3)), s0);
    const auto t = ShiftLeft<17>(s1);
    s2 = Xor(s2, s0);
    s3 = Xor(s3, s1);
    s1 = Xor(s1, s2);
    s0 = Xor(s0, s3);
    s2 = Xor(s2, t);
    s3 = Rol<45>(s3);
    StoreU(result, d, out + offset);
    StoreU(s0, d, words[0] + offset);
    StoreU(s1, d, words[1] + offset);
    StoreU(s2, d, words[2] + offset);
    StoreU(s3, d, words[3] + offset);
  }
  for (; offset < RandomState::kStreams; ++offset)
    out[offset] = random_step_stream(state, offset);
}

}  // namespace detail

static HYLE_FORCE_INLINE void random_fill(RandomState& state, std::uint64_t* dst,
                                          std::size_t count) noexcept {
  while (count != 0 &&
         ::hyle::simd::detail::RandomStateAccess::cursor(state) != 0) {
    *dst++ = detail::random_step_one(state);
    --count;
  }
  while (count >= RandomState::kStreams) {
    detail::random_block(state, dst);
    dst += RandomState::kStreams;
    count -= RandomState::kStreams;
  }
  while (count-- != 0) *dst++ = detail::random_step_one(state);
}

static HYLE_FORCE_INLINE void random_fill(RandomState& state, double* dst,
                                          std::size_t count) noexcept {
  alignas(64) std::uint64_t bits[RandomState::kStreams];
  while (count >= RandomState::kStreams) {
    ::hyle::simd::native::random_fill(state, bits, RandomState::kStreams);
    for (std::size_t i = 0; i < RandomState::kStreams; ++i)
      dst[i] = static_cast<double>(bits[i] >> 11) * 0x1.0p-53;
    dst += RandomState::kStreams;
    count -= RandomState::kStreams;
  }
  for (std::size_t i = 0; i < count; ++i)
    dst[i] = static_cast<double>(detail::random_step_one(state) >> 11) * 0x1.0p-53;
}

static HYLE_FORCE_INLINE void random_fill(RandomState& state, float* dst,
                                          std::size_t count) noexcept {
  alignas(64) std::uint64_t bits[RandomState::kStreams];
  while (count >= RandomState::kStreams) {
    ::hyle::simd::native::random_fill(state, bits, RandomState::kStreams);
    for (std::size_t i = 0; i < RandomState::kStreams; ++i)
      dst[i] = static_cast<float>(bits[i] >> 40) * 0x1.0p-24f;
    dst += RandomState::kStreams;
    count -= RandomState::kStreams;
  }
  for (std::size_t i = 0; i < count; ++i)
    dst[i] = static_cast<float>(detail::random_step_one(state) >> 40) * 0x1.0p-24f;
}

// Register-returning generator for expert fixed-width target code. SVE/RVV vector types are
// sizeless and cannot be class members; scalable targets use native::random_fill, whose state
// lives in ordinary memory. The target id is part of the class specialization so fixed-width
// target definitions can coexist without an ODR collision.
#if !defined(HYLE_SIMD_TARGET_SVE) && !defined(HYLE_SIMD_TARGET_RVV)
#if defined(HYLE_SIMD_TARGET_AVX512)
static constexpr int kTargetId = 4;
#elif defined(HYLE_SIMD_TARGET_AVX2)
static constexpr int kTargetId = 3;
#elif defined(HYLE_SIMD_TARGET_SSE4)
static constexpr int kTargetId = 2;
#elif defined(HYLE_SIMD_TARGET_NEON)
static constexpr int kTargetId = 5;
#elif defined(HYLE_SIMD_TARGET_SVE)
static constexpr int kTargetId = 6;
#elif defined(HYLE_SIMD_TARGET_RVV)
static constexpr int kTargetId = 7;
#else
static constexpr int kTargetId = 1;
#endif

template <int TargetId>
class BasicVectorXoshiro {
  using D = ScalableTag<std::uint64_t>;
  using VU64 = VFromD<D>;

 public:
  explicit BasicVectorXoshiro(std::uint64_t seed) noexcept {
    RandomState seeded(seed);
    const D d;
    auto& words = ::hyle::simd::detail::RandomStateAccess::words(seeded);
    state0_ = LoadU(d, words[0]);
    state1_ = LoadU(d, words[1]);
    state2_ = LoadU(d, words[2]);
    state3_ = LoadU(d, words[3]);
  }

  VU64 operator()() noexcept { return Next(); }
  VU64 Next() noexcept {
    const VU64 result = Add(Rol<23>(Add(state0_, state3_)), state0_);
    const VU64 t = ShiftLeft<17>(state1_);
    state2_ = Xor(state2_, state0_);
    state3_ = Xor(state3_, state1_);
    state1_ = Xor(state1_, state2_);
    state0_ = Xor(state0_, state3_);
    state2_ = Xor(state2_, t);
    state3_ = Rol<45>(state3_);
    return result;
  }

  VFromD<ScalableTag<double>> Uniform() noexcept {
    const ScalableTag<double> df;
    const ScalableTag<std::uint64_t> du;
    const VU64 mantissa = ShiftRight<12>(Next());
    return Sub(BitCast(df, Or(mantissa, Set(du, 0x3ff0000000000000ull))),
               Set(df, 1.0));
  }

 private:
  VU64 state0_, state1_, state2_, state3_;
};

using VectorXoshiro = BasicVectorXoshiro<kTargetId>;
#endif

}  // namespace hyle::simd::native
