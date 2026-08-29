#pragma once

#include <cstddef>
#include <cstdint>

#include "src/simd/bitpack/scalar_layout.h"

// bitstream.h (internal) -- the scalar value-order codec used only for
// canonical-v1 tails. Full blocks are composed separately from fixed 128-bit
// logical Highway lanes. This file also owns the minimal lane-wise scalar
// spelling used when Highway's one-lane fallback cannot instantiate Pack8/16;
// its bytes are the same fixed-128 format, not a second wire layout.
//
// LAYOUT: values are laid LSB-first into a stream of W-bit words (W =
// 8*sizeof(T)). Value i occupies bits [i*bits, i*bits+bits); a field may
// straddle two words (bits <= W guarantees it spans at most two). A 64-bit
// accumulator carries the straddle so no shift is ever by >= its operand width
// (the off>0 guard keeps W-off in [1, W-1]).

namespace hyle::simd::detail {

// packed word count for `count` values at `bits` each -- ceil(count*bits / W).
// Matches hyle::simd::packed_size<T> (kept local so this header stays
// dependency-free).
template <typename T>
constexpr std::size_t bitstream_words(std::size_t count,
                                      unsigned bits) noexcept {
  constexpr unsigned W = 8 * sizeof(T);
  return (count / W) * bits + ((count % W) * bits + (W - 1)) / W;
}

template <typename T>
inline void bitstream_pack(const T *raw, std::size_t count, T *packed,
                           unsigned bits) noexcept {
  constexpr unsigned W = 8 * sizeof(T);
  const std::size_t nwords = bitstream_words<T>(count, bits);
  for (std::size_t i = 0; i < nwords; ++i)
    packed[i] = T{0};
  const std::uint64_t mask =
      (bits >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << bits) - 1);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint64_t v = static_cast<std::uint64_t>(raw[i]) & mask;
    const std::size_t bitpos = i * static_cast<std::size_t>(bits);
    const std::size_t word = bitpos / W;
    const unsigned off = static_cast<unsigned>(bitpos % W);
    packed[word] = static_cast<T>(packed[word] | static_cast<T>(v << off));
    if (off + bits >
        W) // straddle: off>0 here, so W-off is in [1, W-1] (safe shift)
      packed[word + 1] =
          static_cast<T>(packed[word + 1] | static_cast<T>(v >> (W - off)));
  }
}

template <typename T>
inline void bitstream_unpack(const T *packed, std::size_t count, T *raw,
                             unsigned bits) noexcept {
  constexpr unsigned W = 8 * sizeof(T);
  const std::uint64_t mask =
      (bits >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << bits) - 1);
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t bitpos = i * static_cast<std::size_t>(bits);
    const std::size_t word = bitpos / W;
    const unsigned off = static_cast<unsigned>(bitpos % W);
    std::uint64_t acc = static_cast<std::uint64_t>(packed[word]) >> off;
    if (off + bits > W) // straddle (off>0, W-off in [1, W-1])
      acc |= static_cast<std::uint64_t>(packed[word + 1]) << (W - off);
    raw[i] = static_cast<T>(acc & mask);
  }
}

// Highway's fixed-width bit-pack operates lane-wise: for a 128-bit logical tag,
// each logical lane independently packs W input vectors into `bits` output
// vectors. Spell that mapping directly for the no-ISA floor; real SIMD targets
// continue to instantiate Highway itself.
template <typename T>
inline void canonical_scalar_pack(const T *raw, std::size_t count, T *packed,
                                  unsigned bits) noexcept {
  constexpr std::size_t W = 8 * sizeof(T);
  constexpr std::size_t kLogicalLanes = 16 / sizeof(T);
  constexpr std::size_t kBlockValues = W * kLogicalLanes;
  const std::size_t blocks = count / kBlockValues;

  for (std::size_t block = 0; block < blocks; ++block) {
    T *block_out = packed + block * bits * kLogicalLanes;
    for (std::size_t i = 0; i < bits * kLogicalLanes; ++i)
      block_out[i] = T{0};
    const T *block_in = raw + block * kBlockValues;
    if constexpr (sizeof(T) < sizeof(std::uint32_t)) {
      for (const ScalarLayoutSegment segment : scalar_layout<T>(bits)) {
        const std::uint64_t mask = (std::uint64_t{1} << segment.length) - 1;
        for (std::size_t lane = 0; lane < kLogicalLanes; ++lane) {
          const std::uint64_t value =
              (block_in[segment.row * kLogicalLanes + lane] >>
               segment.input_shift) &
              mask;
          const std::size_t out = segment.output_word * kLogicalLanes + lane;
          block_out[out] = static_cast<T>(
              block_out[out] | static_cast<T>(value << segment.output_shift));
        }
      }
    } else {
      // Pack32 uses Highway's generic lane-wise bitstream layout.
      for (std::size_t lane = 0; lane < kLogicalLanes; ++lane) {
        for (std::size_t row = 0; row < W; ++row) {
          const std::uint64_t value = block_in[row * kLogicalLanes + lane];
          const std::size_t bitpos = row * static_cast<std::size_t>(bits);
          const std::size_t word = bitpos / W;
          const unsigned offset = static_cast<unsigned>(bitpos % W);
          const std::size_t out = word * kLogicalLanes + lane;
          block_out[out] =
              static_cast<T>(block_out[out] | static_cast<T>(value << offset));
          if (offset + bits > W) {
            const std::size_t next = (word + 1) * kLogicalLanes + lane;
            block_out[next] = static_cast<T>(
                block_out[next] | static_cast<T>(value >> (W - offset)));
          }
        }
      }
    }
  }

  const std::size_t block_raw = blocks * kBlockValues;
  const std::size_t block_packed = blocks * bits * kLogicalLanes;
  bitstream_pack(raw + block_raw, count - block_raw, packed + block_packed,
                 bits);
}

template <typename T>
inline void canonical_scalar_unpack(const T *packed, std::size_t count, T *raw,
                                    unsigned bits) noexcept {
  constexpr std::size_t W = 8 * sizeof(T);
  constexpr std::size_t kLogicalLanes = 16 / sizeof(T);
  constexpr std::size_t kBlockValues = W * kLogicalLanes;
  const std::size_t blocks = count / kBlockValues;

  for (std::size_t block = 0; block < blocks; ++block) {
    const T *block_in = packed + block * bits * kLogicalLanes;
    T *block_out = raw + block * kBlockValues;
    for (std::size_t i = 0; i < kBlockValues; ++i)
      block_out[i] = T{0};
    if constexpr (sizeof(T) < sizeof(std::uint32_t)) {
      for (const ScalarLayoutSegment segment : scalar_layout<T>(bits)) {
        const std::uint64_t mask = (std::uint64_t{1} << segment.length) - 1;
        for (std::size_t lane = 0; lane < kLogicalLanes; ++lane) {
          const std::uint64_t value =
              (block_in[segment.output_word * kLogicalLanes + lane] >>
               segment.output_shift) &
              mask;
          const std::size_t out = segment.row * kLogicalLanes + lane;
          block_out[out] = static_cast<T>(
              block_out[out] | static_cast<T>(value << segment.input_shift));
        }
      }
    } else {
      const std::uint64_t mask = (std::uint64_t{1} << bits) - 1;
      for (std::size_t lane = 0; lane < kLogicalLanes; ++lane) {
        for (std::size_t row = 0; row < W; ++row) {
          const std::size_t bitpos = row * static_cast<std::size_t>(bits);
          const std::size_t word = bitpos / W;
          const unsigned offset = static_cast<unsigned>(bitpos % W);
          std::uint64_t value = static_cast<std::uint64_t>(
                                    block_in[word * kLogicalLanes + lane]) >>
                                offset;
          if (offset + bits > W) {
            value |= static_cast<std::uint64_t>(
                         block_in[(word + 1) * kLogicalLanes + lane])
                     << (W - offset);
          }
          block_out[row * kLogicalLanes + lane] = static_cast<T>(value & mask);
        }
      }
    }
  }

  const std::size_t block_raw = blocks * kBlockValues;
  const std::size_t block_packed = blocks * bits * kLogicalLanes;
  bitstream_unpack(packed + block_packed, count - block_raw, raw + block_raw,
                   bits);
}

} // namespace hyle::simd::detail
