// Direct-Highway per-target implementation of HYLE's fixed eight-stream xoshiro
// sequence. Highway owns register lowering; HYLE owns state, stream order and
// float mapping.

#include <cstddef>
#include <cstdint>

#include "hwy/highway.h"

#include "hyle/simd/random.h"
#include "src/simd/random/kernels.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including random/kernel_impl.h"
#endif

#define HYLE_RANDOM_CAT_(a, b) a##b
#define HYLE_RANDOM_CAT(a, b) HYLE_RANDOM_CAT_(a, b)
#define HYLE_RANDOM_FN(name) HYLE_RANDOM_CAT(name, HYLE_SIMD_VARIANT_SUFFIX)

namespace hyle::simd::detail {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

HWY_INLINE std::uint64_t step_stream(RandomState &state,
                                     std::size_t stream) noexcept {
  auto &words = RandomStateAccess::words(state);
  return random_xoshiro_step(words[0][stream], words[1][stream],
                             words[2][stream], words[3][stream]);
}

HWY_INLINE std::uint64_t step_one(RandomState &state) noexcept {
  auto &cursor = RandomStateAccess::cursor(state);
  const std::size_t stream = cursor;
  const std::uint64_t result = step_stream(state, stream);
  cursor = static_cast<std::uint32_t>((stream + 1) % RandomState::kStreams);
  return result;
}

template <bool kStore>
HWY_INLINE void update_block(RandomState &state,
                             std::uint64_t *output) noexcept {
  const hn::CappedTag<std::uint64_t, RandomState::kStreams> d;
  const std::size_t lanes = hn::Lanes(d);
  auto &words = RandomStateAccess::words(state);
  std::size_t offset = 0;
  for (; offset + lanes <= RandomState::kStreams; offset += lanes) {
    auto s0 = hn::LoadU(d, words[0] + offset);
    auto s1 = hn::LoadU(d, words[1] + offset);
    auto s2 = hn::LoadU(d, words[2] + offset);
    auto s3 = hn::LoadU(d, words[3] + offset);
    if constexpr (kStore) {
      const auto result = hn::Add(hn::RotateRight<41>(hn::Add(s0, s3)), s0);
      hn::StoreU(result, d, output + offset);
    }
    const auto shifted = hn::ShiftLeft<17>(s1);
    s2 = hn::Xor(s2, s0);
    s3 = hn::Xor(s3, s1);
    s1 = hn::Xor(s1, s2);
    s0 = hn::Xor(s0, s3);
    s2 = hn::Xor(s2, shifted);
    s3 = hn::RotateRight<19>(s3);
    hn::StoreU(s0, d, words[0] + offset);
    hn::StoreU(s1, d, words[1] + offset);
    hn::StoreU(s2, d, words[2] + offset);
    hn::StoreU(s3, d, words[3] + offset);
  }
  for (; offset < RandomState::kStreams; ++offset) {
    const std::uint64_t value = step_stream(state, offset);
    if constexpr (kStore)
      output[offset] = value;
  }
}

HWY_INLINE void fill_u64(RandomState &state, std::uint64_t *dst,
                         std::size_t count) noexcept {
  while (count != 0 && RandomStateAccess::cursor(state) != 0) {
    *dst++ = step_one(state);
    --count;
  }
  while (count >= RandomState::kStreams) {
    update_block<true>(state, dst);
    dst += RandomState::kStreams;
    count -= RandomState::kStreams;
  }
  while (count-- != 0)
    *dst++ = step_one(state);
}

HWY_INLINE void advance_state(RandomState &state, std::size_t count) noexcept {
  while (count != 0 && RandomStateAccess::cursor(state) != 0) {
    (void)step_one(state);
    --count;
  }
  while (count >= RandomState::kStreams) {
    update_block<false>(state, nullptr);
    count -= RandomState::kStreams;
  }
  while (count-- != 0)
    (void)step_one(state);
}

} // namespace

void HYLE_RANDOM_FN(random_fill_u64_)(RandomState *state, std::uint64_t *dst,
                                      std::size_t count) noexcept {
  fill_u64(*state, dst, count);
}

void HYLE_RANDOM_FN(random_fill_f32_)(RandomState *state, float *dst,
                                      std::size_t count) noexcept {
  alignas(64) std::uint64_t bits[RandomState::kStreams];
  while (count >= RandomState::kStreams) {
    fill_u64(*state, bits, RandomState::kStreams);
    for (std::size_t i = 0; i < RandomState::kStreams; ++i)
      dst[i] = static_cast<float>(bits[i] >> 40) * 0x1.0p-24f;
    dst += RandomState::kStreams;
    count -= RandomState::kStreams;
  }
  for (std::size_t i = 0; i < count; ++i)
    dst[i] = static_cast<float>(step_one(*state) >> 40) * 0x1.0p-24f;
}

void HYLE_RANDOM_FN(random_fill_f64_)(RandomState *state, double *dst,
                                      std::size_t count) noexcept {
  alignas(64) std::uint64_t bits[RandomState::kStreams];
  while (count >= RandomState::kStreams) {
    fill_u64(*state, bits, RandomState::kStreams);
    for (std::size_t i = 0; i < RandomState::kStreams; ++i)
      dst[i] = static_cast<double>(bits[i] >> 11) * 0x1.0p-53;
    dst += RandomState::kStreams;
    count -= RandomState::kStreams;
  }
  for (std::size_t i = 0; i < count; ++i)
    dst[i] = static_cast<double>(step_one(*state) >> 11) * 0x1.0p-53;
}

void HYLE_RANDOM_FN(random_advance_)(RandomState *state,
                                     std::size_t count) noexcept {
  advance_state(*state, count);
}

} // namespace hyle::simd::detail

#undef HYLE_RANDOM_FN
#undef HYLE_RANDOM_CAT
#undef HYLE_RANDOM_CAT_
