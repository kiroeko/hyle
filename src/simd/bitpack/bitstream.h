#pragma once

#include <cstddef>
#include <cstdint>

// bitstream.h (internal) -- the pure scalar contiguous bit-stream codec that is BOTH the scalar
// floor (bitpack_scalar.cc, whole-array) AND the sub-block tail helper inside every SIMD tier
// (bitpack_body.inc). Keeping ONE codec used by both is what makes the same-tier round trip hold:
// a SIMD tier packs the whole N*B blocks with HYLE Pack<kBits> and the leftover tail with
// THIS codec; unpack mirrors the split, so unpack(pack(x)) == x. (The packed bytes are an in-
// memory codec, not a portable wire format -- see bitpack.h.)
//
// LAYOUT: values are laid LSB-first into a stream of W-bit words (W = 8*sizeof(T)). Value i
// occupies bits [i*bits, i*bits+bits); a field may straddle two words (bits <= W guarantees it
// spans at most two). A 64-bit accumulator carries the straddle so no shift is ever by >= its
// operand width (the off>0 guard keeps W-off in [1, W-1]).

namespace hyle::simd::detail {

// packed word count for `count` values at `bits` each -- ceil(count*bits / W). Matches
// hyle::simd::packed_size<T> (kept local so this header stays dependency-free).
template <typename T>
constexpr std::size_t bitstream_words(std::size_t count, unsigned bits) noexcept {
  constexpr unsigned W = 8 * sizeof(T);
  return (count * bits + (W - 1)) / W;
}

template <typename T>
inline void bitstream_pack(const T* raw, std::size_t count, T* packed, unsigned bits) noexcept {
  constexpr unsigned W = 8 * sizeof(T);
  const std::size_t nwords = bitstream_words<T>(count, bits);
  for (std::size_t i = 0; i < nwords; ++i) packed[i] = T{0};
  const std::uint64_t mask = (bits >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << bits) - 1);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint64_t v = static_cast<std::uint64_t>(raw[i]) & mask;
    const std::size_t bitpos = i * static_cast<std::size_t>(bits);
    const std::size_t word = bitpos / W;
    const unsigned off = static_cast<unsigned>(bitpos % W);
    packed[word] = static_cast<T>(packed[word] | static_cast<T>(v << off));
    if (off + bits > W)  // straddle: off>0 here, so W-off is in [1, W-1] (safe shift)
      packed[word + 1] = static_cast<T>(packed[word + 1] | static_cast<T>(v >> (W - off)));
  }
}

template <typename T>
inline void bitstream_unpack(const T* packed, std::size_t count, T* raw, unsigned bits) noexcept {
  constexpr unsigned W = 8 * sizeof(T);
  const std::uint64_t mask = (bits >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << bits) - 1);
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t bitpos = i * static_cast<std::size_t>(bits);
    const std::size_t word = bitpos / W;
    const unsigned off = static_cast<unsigned>(bitpos % W);
    std::uint64_t acc = static_cast<std::uint64_t>(packed[word]) >> off;
    if (off + bits > W)  // straddle (off>0, W-off in [1, W-1])
      acc |= static_cast<std::uint64_t>(packed[word + 1]) << (W - off);
    raw[i] = static_cast<T>(acc & mask);
  }
}

}  // namespace hyle::simd::detail
