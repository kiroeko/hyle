// bitpack_body.inc -- the SHARED body of the Highway bit_pack borrow cores. Included by each ISA
// tier TU (bitpack_highway_avx2.cc / _avx512.cc / _neon.cc) AFTER it defines HYLE_BITPACK_TIER
// (the symbol suffix: avx2 / avx512 / neon) and includes the Highway headers. NOT a standalone TU
// (no include guard on purpose -- textually pasted once per tier with a different suffix, exactly
// like vqsort_body.inc / the math borrow TUs).
//
// Each tier compiles at its own -march (via hyle_set_highway_isa / hyle_set_avx512_bitpack_isa in
// CMake), so hn:: / Pack8/16/32 lower to that tier's width. The runtime `bits` is folded to the
// compile-time kBits template arg via an index_sequence over 1..B (B = 8*sizeof(T)); only kBits in
// [1,B] are instantiated, so Pack8<9> etc never exist.
//
// SPLIT: the whole N*B-value BLOCKS (N = Lanes(d)) go through Highway's Pack<kBits>::Pack (which
// emits kBits*N packed elements per block); the sub-block TAIL (< N*B values) goes through the
// scalar bit-stream codec (bitstream.h) -- the SAME codec the scalar floor uses, so a same-host
// (same-tier) round trip is exact. The tail's packed offset is nblocks*N*bits elements, which
// equals nblocks*N*kBits (runtime bits == the folded kBits), so it lands right after the blocks.

#include "src/simd/bitpack/kernels.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "src/simd/bitpack/bitstream.h"

namespace hn = hwy::HWY_NAMESPACE;

namespace hyle::simd::detail {

// VENDOR-BUG SIDESTEP: Highway's hand-written Pack16<11> and Pack16<13> specializations do NOT
// round-trip full-range 11-/13-bit values (probed on MSVC AVX2 AND the portable EMU128 target, so
// it is the ALGORITHM, not codegen). Highway's own bit_pack_test masks inputs with `& kBits` (the
// literal width number, e.g. `& 11`) instead of `& ((1<<kBits)-1)`, so it never feeds values wide
// enough to expose it -- the bug ships untested upstream. All Pack8 widths, all Pack32 widths (the
// generic unroller), and the other 14 Pack16 widths verify clean. For the two broken widths we
// route the WHOLE array through the scalar bit-stream codec (bitstream.h -- correctness oracle,
// round-trip tested), by forcing nblocks=0 so nothing hits the Highway block path. pack + unpack
// share this predicate, so the layout stays self-consistent.
template <typename T>
constexpr bool bitpack_width_borrowable(unsigned bits) noexcept {
  if constexpr (sizeof(T) == 2)
    return bits != 11 && bits != 13;
  else
    return true;
}

#define HYLE_BITPACK_PASTE2(a, b) a##b
#define HYLE_BITPACK_PASTE(a, b) HYLE_BITPACK_PASTE2(a, b)
// Builds e.g. pack_u8_avx2 -- HYLE_BITPACK_TIER is a macro so it needs the two-level paste.
#define HYLE_BITPACK_SYM(stem, tag) HYLE_BITPACK_PASTE(stem##_##tag##_, HYLE_BITPACK_TIER)

// Select the right Highway packer class by lane byte size (Pack8/Pack16/Pack32). if constexpr
// discards the untaken branches from instantiation, so Pack16<kBits> is never instantiated for a
// 1-byte T (whose kBits only reaches 8). One Pack() call handles a block of B*Lanes(d) raw values.
template <std::size_t kBits, class D, typename T>
HWY_INLINE void bitpack_pack_one(D d, const T* raw, T* packed) {
  if constexpr (sizeof(T) == 1)
    hn::Pack8<kBits>().Pack(d, raw, packed);
  else if constexpr (sizeof(T) == 2)
    hn::Pack16<kBits>().Pack(d, raw, packed);
  else
    hn::Pack32<kBits>().Pack(d, raw, packed);
}

template <std::size_t kBits, class D, typename T>
HWY_INLINE void bitpack_unpack_one(D d, const T* packed, T* raw) {
  if constexpr (sizeof(T) == 1)
    hn::Pack8<kBits>().Unpack(d, packed, raw);
  else if constexpr (sizeof(T) == 2)
    hn::Pack16<kBits>().Unpack(d, packed, raw);
  else
    hn::Pack32<kBits>().Unpack(d, packed, raw);
}

// Fold: for the matching kBits (== bits), run the whole block loop with a compile-time packed
// stride of kBits*N. `nblocks` blocks of N*B raw values each. Only kBits in [1,B] instantiated.
template <typename T, class D, std::size_t... Ks>
HWY_INLINE void bitpack_pack_blocks(unsigned bits, D d, std::size_t N, std::size_t nblocks,
                                    const T* raw, T* packed, std::index_sequence<Ks...>) {
  constexpr std::size_t B = 8 * sizeof(T);
  (void)(((bits == Ks + 1) ? (([&] {
            const std::size_t raw_step = N * B;
            const std::size_t packed_step = N * (Ks + 1);
            for (std::size_t b = 0; b < nblocks; ++b)
              bitpack_pack_one<Ks + 1>(d, raw + b * raw_step, packed + b * packed_step);
          }()),
           true)
                           : false) ||
         ...);
}

template <typename T, class D, std::size_t... Ks>
HWY_INLINE void bitpack_unpack_blocks(unsigned bits, D d, std::size_t N, std::size_t nblocks,
                                      const T* packed, T* raw, std::index_sequence<Ks...>) {
  constexpr std::size_t B = 8 * sizeof(T);
  (void)(((bits == Ks + 1) ? (([&] {
            const std::size_t raw_step = N * B;
            const std::size_t packed_step = N * (Ks + 1);
            for (std::size_t b = 0; b < nblocks; ++b)
              bitpack_unpack_one<Ks + 1>(d, packed + b * packed_step, raw + b * raw_step);
          }()),
           true)
                           : false) ||
         ...);
}

#define HYLE_BITPACK_DEFINE(T, tag)                                                          \
  void HYLE_BITPACK_SYM(pack, tag)(const T* raw, std::size_t count, T* packed, unsigned bits) \
      noexcept {                                                                             \
    constexpr std::size_t B = 8 * sizeof(T);                                                 \
    const hn::ScalableTag<T> d;                                                              \
    const std::size_t N = hn::Lanes(d);                                                      \
    const std::size_t block_size = N * B;                                                    \
    const std::size_t nblocks =                                                              \
        bitpack_width_borrowable<T>(bits) ? (count / block_size) : std::size_t{0};           \
    const std::size_t block_raw = nblocks * block_size;                                      \
    const std::size_t block_packed = nblocks * N * bits;                                     \
    bitpack_pack_blocks<T>(bits, d, N, nblocks, raw, packed,                                 \
                           std::make_index_sequence<B>{});                                   \
    bitstream_pack<T>(raw + block_raw, count - block_raw, packed + block_packed, bits);      \
  }                                                                                          \
  void HYLE_BITPACK_SYM(unpack, tag)(const T* packed, std::size_t count, T* raw,             \
                                     unsigned bits) noexcept {                               \
    constexpr std::size_t B = 8 * sizeof(T);                                                 \
    const hn::ScalableTag<T> d;                                                              \
    const std::size_t N = hn::Lanes(d);                                                      \
    const std::size_t block_size = N * B;                                                    \
    const std::size_t nblocks =                                                              \
        bitpack_width_borrowable<T>(bits) ? (count / block_size) : std::size_t{0};           \
    const std::size_t block_raw = nblocks * block_size;                                      \
    const std::size_t block_packed = nblocks * N * bits;                                     \
    bitpack_unpack_blocks<T>(bits, d, N, nblocks, packed, raw,                               \
                             std::make_index_sequence<B>{});                                 \
    bitstream_unpack<T>(packed + block_packed, count - block_raw, raw + block_raw, bits);    \
  }

HYLE_BITPACK_TYPES(HYLE_BITPACK_DEFINE)

#undef HYLE_BITPACK_DEFINE
#undef HYLE_BITPACK_SYM
#undef HYLE_BITPACK_PASTE
#undef HYLE_BITPACK_PASTE2

}  // namespace hyle::simd::detail
