#include "hyle/simd/random.h"

#include "src/simd/dispatch_internal.h"
#include "src/simd/random/kernels.h"
#include "src/simd/runtime_internal.h"

namespace hyle::simd {

namespace {

void xoshiro_long_jump(std::uint64_t state[4]) noexcept {
  constexpr std::uint64_t kLongJump[4] = {
      0x76e15d3efefdcbbfull, 0xc5004e441c522fb3ull, 0x77710069854ee241ull,
      0x39109bb02acbe635ull};
  std::uint64_t out[4] = {};
  for (std::size_t i = 0; i < 4; ++i) {
    const std::uint64_t word = kLongJump[i];
    for (int bit = 0; bit < 64; ++bit) {
      if ((word & (std::uint64_t{1} << bit)) != 0) {
        out[0] ^= state[0];
        out[1] ^= state[1];
        out[2] ^= state[2];
        out[3] ^= state[3];
      }
      (void)detail::random_xoshiro_step(state[0], state[1], state[2], state[3]);
    }
  }
  for (std::size_t i = 0; i < 4; ++i)
    state[i] = out[i];
}

} // namespace

void RandomState::advance(std::size_t count) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_SIMD_EXACT_SYMBOL(random_advance)(this, count);
#else
  runtime().random.advance(this, count);
#endif
}

RandomState RandomState::split() noexcept {
  RandomState child = *this;
  for (std::size_t stream = 0; stream < kStreams; ++stream) {
    std::uint64_t lane[4] = {state_[0][stream], state_[1][stream],
                             state_[2][stream], state_[3][stream]};
    xoshiro_long_jump(lane);
    for (std::size_t word = 0; word < 4; ++word)
      state_[word][stream] = lane[word];
  }
  return child;
}

void random_fill(RandomState &state, std::uint64_t *dst,
                 std::size_t count) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_SIMD_EXACT_SYMBOL(random_fill_u64)(&state, dst, count);
#else
  runtime().random.fill_u64(&state, dst, count);
#endif
}
void random_fill(RandomState &state, float *dst, std::size_t count) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_SIMD_EXACT_SYMBOL(random_fill_f32)(&state, dst, count);
#else
  runtime().random.fill_f32(&state, dst, count);
#endif
}
void random_fill(RandomState &state, double *dst, std::size_t count) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_SIMD_EXACT_SYMBOL(random_fill_f64)(&state, dst, count);
#else
  runtime().random.fill_f64(&state, dst, count);
#endif
}
} // namespace hyle::simd
