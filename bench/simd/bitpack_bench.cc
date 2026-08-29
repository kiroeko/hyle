// SIMD-10 bounded bitpack/random integration instrument.
//
// The candidate is the exact shipping target body instantiated in this TU. The
// bitpack peer directly composes locked Highway Pack16 over the same canonical
// fixed-128 layout; the random peer independently spells the same eight-stream
// xoshiro state transition with direct Highway ops. Both sides therefore share
// the ISA, bytes/sequence, pointer/count contract and resource accounting. The
// common parity engine supplies placement replicas, reverse order/layout, A/A
// calibration and CPU pinning. Public/body rows are informational only.

#include <nanobench.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "hyle/core/compiler.h"
#include "hyle/simd/bitpack.h"
#include "hyle/simd/random.h"
#include "op_parity_engine.h"

#define HWY_COMPILE_ONLY_STATIC 1
#define HYLE_SIMD_VARIANT_SUFFIX parity
#include "src/simd/bitpack/kernel_impl.h"
#include "src/simd/random/kernel_impl.h"
#undef HYLE_SIMD_VARIANT_SUFFIX
#undef HWY_COMPILE_ONLY_STATIC

namespace nb = ankerl::nanobench;
namespace hs = hyle::simd;
namespace hn = hwy::HWY_NAMESPACE;

namespace {

using T = std::uint16_t;

constexpr std::size_t kResidentCount = 4096;
constexpr std::size_t kStreamingCount = 1u << 20;
constexpr std::size_t kPeerLogicalLanes = 8;
constexpr std::size_t kPeerBlockValues = 128;

template <unsigned kBits>
HWY_INLINE void peer_pack_width(const T* raw, std::size_t count,
                                T* packed) noexcept {
  const hn::CappedTag<T, kPeerLogicalLanes> d;
  const std::size_t blocks = count / kPeerBlockValues;
  for (std::size_t block = 0; block < blocks; ++block) {
    hn::Pack16<kBits>().Pack(d, raw + block * kPeerBlockValues,
                             packed + block * kBits * kPeerLogicalLanes);
  }
  const std::size_t block_raw = blocks * kPeerBlockValues;
  const std::size_t block_packed = blocks * kBits * kPeerLogicalLanes;
  hs::detail::bitstream_pack(raw + block_raw, count - block_raw,
                             packed + block_packed, kBits);
}

template <unsigned kBits>
HWY_INLINE void peer_unpack_width(const T* packed, std::size_t count,
                                  T* raw) noexcept {
  const hn::CappedTag<T, kPeerLogicalLanes> d;
  const std::size_t blocks = count / kPeerBlockValues;
  for (std::size_t block = 0; block < blocks; ++block) {
    hn::Pack16<kBits>().Unpack(
        d, packed + block * kBits * kPeerLogicalLanes,
        raw + block * kPeerBlockValues);
  }
  const std::size_t block_raw = blocks * kPeerBlockValues;
  const std::size_t block_packed = blocks * kBits * kPeerLogicalLanes;
  hs::detail::bitstream_unpack(packed + block_packed, count - block_raw,
                               raw + block_raw, kBits);
}

HYLE_NOINLINE void candidate_pack(const T* raw, std::size_t count, T* packed,
                                  unsigned bits) noexcept {
  hs::detail::pack_u16_parity(raw, count, packed, bits);
}

HYLE_NOINLINE void candidate_unpack(const T* packed, std::size_t count, T* raw,
                                    unsigned bits) noexcept {
  hs::detail::unpack_u16_parity(packed, count, raw, bits);
}

HYLE_NOINLINE void peer_pack(const T* raw, std::size_t count, T* packed,
                             unsigned bits) noexcept {
  if (bits == 11) {
    peer_pack_width<11>(raw, count, packed);
  } else {
    peer_pack_width<13>(raw, count, packed);
  }
}

HYLE_NOINLINE void peer_unpack(const T* packed, std::size_t count, T* raw,
                               unsigned bits) noexcept {
  if (bits == 11) {
    peer_unpack_width<11>(packed, count, raw);
  } else {
    peer_unpack_width<13>(packed, count, raw);
  }
}

HWY_INLINE std::uint64_t peer_step_stream(hs::RandomState& state,
                                          std::size_t stream) noexcept {
  auto& words = hs::detail::RandomStateAccess::words(state);
  return hs::detail::random_xoshiro_step(
      words[0][stream], words[1][stream], words[2][stream],
      words[3][stream]);
}

HWY_INLINE std::uint64_t peer_step_one(hs::RandomState& state) noexcept {
  auto& cursor = hs::detail::RandomStateAccess::cursor(state);
  const std::size_t stream = cursor;
  const std::uint64_t result = peer_step_stream(state, stream);
  cursor = static_cast<std::uint32_t>(
      (stream + 1) % hs::RandomState::kStreams);
  return result;
}

HWY_INLINE void peer_update_block(hs::RandomState& state,
                                  std::uint64_t* output) noexcept {
  const hn::CappedTag<std::uint64_t, hs::RandomState::kStreams> d;
  const std::size_t lanes = hn::Lanes(d);
  auto& words = hs::detail::RandomStateAccess::words(state);
  std::size_t offset = 0;
  for (; offset + lanes <= hs::RandomState::kStreams; offset += lanes) {
    auto s0 = hn::LoadU(d, words[0] + offset);
    auto s1 = hn::LoadU(d, words[1] + offset);
    auto s2 = hn::LoadU(d, words[2] + offset);
    auto s3 = hn::LoadU(d, words[3] + offset);
    hn::StoreU(hn::Add(hn::RotateRight<41>(hn::Add(s0, s3)), s0), d,
               output + offset);
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
  for (; offset < hs::RandomState::kStreams; ++offset) {
    output[offset] = peer_step_stream(state, offset);
  }
}

HWY_INLINE void peer_fill_u64(hs::RandomState& state, std::uint64_t* dst,
                              std::size_t count) noexcept {
  while (count != 0 && hs::detail::RandomStateAccess::cursor(state) != 0) {
    *dst++ = peer_step_one(state);
    --count;
  }
  while (count >= hs::RandomState::kStreams) {
    peer_update_block(state, dst);
    dst += hs::RandomState::kStreams;
    count -= hs::RandomState::kStreams;
  }
  while (count-- != 0) *dst++ = peer_step_one(state);
}

HYLE_NOINLINE void peer_random_fill_f32(hs::RandomState* state, float* dst,
                                        std::size_t count) noexcept {
  alignas(64) std::uint64_t bits[hs::RandomState::kStreams];
  while (count >= hs::RandomState::kStreams) {
    peer_fill_u64(*state, bits, hs::RandomState::kStreams);
    for (std::size_t i = 0; i < hs::RandomState::kStreams; ++i) {
      dst[i] = static_cast<float>(bits[i] >> 40) * 0x1.0p-24f;
    }
    dst += hs::RandomState::kStreams;
    count -= hs::RandomState::kStreams;
  }
  for (std::size_t i = 0; i < count; ++i) {
    dst[i] = static_cast<float>(peer_step_one(*state) >> 40) * 0x1.0p-24f;
  }
}

HYLE_NOINLINE void candidate_random_fill_f32(hs::RandomState* state,
                                             float* dst,
                                             std::size_t count) noexcept {
  hs::detail::random_fill_f32_parity(state, dst, count);
}

HYLE_NOINLINE void peer_add(const float* x, const float* y, float* dst,
                            std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t lanes = hn::Lanes(d);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    hn::StoreU(hn::Add(hn::LoadU(d, x + i), hn::LoadU(d, y + i)), d,
               dst + i);
  }
  for (; i < n; ++i) dst[i] = x[i] + y[i];
}

std::size_t epoch_iterations(std::size_t n) {
  return n == kResidentCount ? 4096 : 8;
}

template <class Candidate, class Peer>
void run_pair(const std::string& title, std::size_t n, Candidate candidate,
              Peer peer) {
  nb::Bench bench;
  bench.title(title)
      .unit("value")
      .relative(true)
      .batch(n)
      .minEpochIterations(epoch_iterations(n));
  HYLE_RUN_PAIR(
      bench,
      [&](auto rep) {
        hyle_bench::stagger(rep);
        candidate();
        hyle_bench::mark(rep);
      },
      [&](auto rep) {
        hyle_bench::stagger(rep);
        peer();
        hyle_bench::mark(rep);
      });
}

template <class Public, class Body>
void run_public_info(const std::string& title, std::size_t n,
                     Public public_call, Body body_call) {
  nb::Bench bench;
  bench.title(title)
      .unit("value")
      .relative(true)
      .batch(n)
      .minEpochIterations(epoch_iterations(n));
  HYLE_RUN_INFO_PAIR(
      bench,
      [&](auto rep) {
        hyle_bench::stagger(rep);
        public_call();
        hyle_bench::mark(rep);
      },
      [&](auto rep) {
        hyle_bench::stagger(rep);
        body_call();
        hyle_bench::mark(rep);
      });
}

void run_anchor() {
  std::vector<float> x(kResidentCount), y(kResidentCount), output(kResidentCount);
  for (std::size_t i = 0; i < kResidentCount; ++i) {
    x[i] = static_cast<float>(i & 31u);
    y[i] = static_cast<float>((i * 3) & 63u);
  }
  const float* x_data = x.data();
  const float* y_data = y.data();
  float* output_data = output.data();
  run_pair("Add f32 (4096)", kResidentCount,
           [&] {
             peer_add(x_data, y_data, output_data, kResidentCount);
             nb::doNotOptimizeAway(output);
           },
           [&] {
             peer_add(x_data, y_data, output_data, kResidentCount);
             nb::doNotOptimizeAway(output);
           });
}

void run_bitpack(std::size_t n, unsigned bits) {
  std::vector<T> raw(n);
  const T mask = static_cast<T>((std::uint32_t{1} << bits) - 1);
  for (std::size_t i = 0; i < n; ++i) {
    raw[i] = static_cast<T>((i * 40503u + i / 7u + 19u) & mask);
  }
  const std::size_t words = hs::packed_size<T>(n, bits);
  std::vector<T> packed(words), peer_packed(words), output(n), peer_output(n);
  candidate_pack(raw.data(), n, packed.data(), bits);
  peer_pack(raw.data(), n, peer_packed.data(), bits);
  candidate_unpack(packed.data(), n, output.data(), bits);
  peer_unpack(packed.data(), n, peer_output.data(), bits);
  if (packed != peer_packed || output != raw || peer_output != raw) {
    std::fputs("SIMD-10 qualification mismatch\n", stderr);
    std::abort();
  }

  const T* raw_data = raw.data();
  T* packed_data = packed.data();
  T* output_data = output.data();
  const std::string suffix = " b" + std::to_string(bits) + " (" +
                             std::to_string(n) + ")";

  run_pair("bitpack pack u16" + suffix, n,
           [&] {
             candidate_pack(raw_data, n, packed_data, bits);
             nb::doNotOptimizeAway(packed);
           },
           [&] {
             peer_pack(raw_data, n, packed_data, bits);
             nb::doNotOptimizeAway(packed);
           });
  run_pair("bitpack unpack u16" + suffix, n,
           [&] {
             candidate_unpack(packed_data, n, output_data, bits);
             nb::doNotOptimizeAway(output);
           },
           [&] {
             peer_unpack(packed_data, n, output_data, bits);
             nb::doNotOptimizeAway(output);
           });

  if (bits == 11) {
    run_public_info("public/body bitpack pack" + suffix, n,
                    [&] {
                      hs::pack(raw_data, n, packed_data, bits);
                      nb::doNotOptimizeAway(packed);
                    },
                    [&] {
                      candidate_pack(raw_data, n, packed_data, bits);
                      nb::doNotOptimizeAway(packed);
                    });
    run_public_info("public/body bitpack unpack" + suffix, n,
                    [&] {
                      hs::unpack(packed_data, n, output_data, bits);
                      nb::doNotOptimizeAway(output);
                    },
                    [&] {
                      candidate_unpack(packed_data, n, output_data, bits);
                      nb::doNotOptimizeAway(output);
                    });
  }
}

void run_random(std::size_t n) {
  constexpr std::uint64_t kSeed = 0x8b8b8b8b12345678ull;
  hs::RandomState candidate_check{kSeed};
  hs::RandomState peer_check{kSeed};
  std::vector<float> candidate_output(n), peer_output(n);
  candidate_random_fill_f32(&candidate_check, candidate_output.data(), n);
  peer_random_fill_f32(&peer_check, peer_output.data(), n);
  if (candidate_output != peer_output) {
    std::fputs("SIMD-10 random qualification mismatch\n", stderr);
    std::abort();
  }

  hs::RandomState state{kSeed};
  std::vector<float> output(n);
  hs::RandomState* state_ptr = &state;
  float* output_data = output.data();
  const std::string suffix = " (" + std::to_string(n) + ")";

  run_pair("random fill f32" + suffix, n,
           [&] {
             candidate_random_fill_f32(state_ptr, output_data, n);
             nb::doNotOptimizeAway(output);
           },
           [&] {
             peer_random_fill_f32(state_ptr, output_data, n);
             nb::doNotOptimizeAway(output);
           });
  run_public_info("public/body random fill f32" + suffix, n,
                  [&] {
                    hs::random_fill(*state_ptr, output_data, n);
                    nb::doNotOptimizeAway(output);
                  },
                  [&] {
                    candidate_random_fill_f32(state_ptr, output_data, n);
                    nb::doNotOptimizeAway(output);
                  });
}

}  // namespace

int main() {
  hyle_bench::pin_and_boost();
  std::printf(
      "SIMD-10 bitpack/random parity: shipping target body vs direct Highway\n");
  run_anchor();
  for (const std::size_t n : {kResidentCount, kStreamingCount}) {
    run_bitpack(n, 11);
    run_bitpack(n, 13);
    run_random(n);
  }
}
