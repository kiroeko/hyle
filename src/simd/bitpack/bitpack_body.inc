// bitpack_body.inc -- shared HYLE SIMD bit-pack implementation. Each HYLE target TU defines
// HYLE_SIMD_TARGET_* and HYLE_SIMD_VARIANT_SUFFIX before including this file. It is deliberately
// multi-included, once per ISA tier, so ScalableTag selects that TU's native vector width.
// Scalable product TUs define HYLE_BITPACK_FIXED_128_LAYOUT: their logical block is frozen at
// 128 bits even when the physical SVE/RVV vector is wider. This makes the in-memory format
// independent of SVE VL / RVV VLEN and compatible with the NEON layout.
//
// Full blocks contain B * Lanes(d) raw values (B = 8 * sizeof(T)) and are packed by the migrated
// Pack8/Pack16/Pack32 specializations. The remaining sub-block tail uses the scalar bit stream;
// the packed offset is nblocks * Lanes(d) * bits, exactly after the vector block region.

#include "src/simd/bitpack/kernels.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "hyle/simd/vec.h"
#include "src/simd/bitpack/bitpack_hyle-inl.h"
#include "src/simd/bitpack/bitpack_hyle_u16-inl.h"
#include "src/simd/bitpack/bitstream.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including bitpack_body.inc"
#endif

namespace hyle::simd::detail {

#define HYLE_BITPACK_PASTE2(a, b) a##b
#define HYLE_BITPACK_PASTE(a, b) HYLE_BITPACK_PASTE2(a, b)
#define HYLE_BITPACK_SYM(stem) HYLE_BITPACK_PASTE(stem, HYLE_SIMD_VARIANT_SUFFIX)

// kBits remains compile-time-specialized. The corrected u16 11/13 layouts supersede the upstream
// implementations; all other widths use the direct semantic migration.
template <std::size_t kBits, class D, typename T>
HYLE_FORCE_INLINE void bitpack_pack_one(D d, const T* raw, T* packed) noexcept {
  if constexpr (sizeof(T) == 1) {
    Pack8<kBits>().Pack(d, raw, packed);
  } else if constexpr (sizeof(T) == 2 && (kBits == 11 || kBits == 13)) {
    bitpack_hyle_pack_u16<kBits>(d, raw, packed);
  } else if constexpr (sizeof(T) == 2) {
    Pack16<kBits>().Pack(d, raw, packed);
  } else {
    Pack32<kBits>().Pack(d, raw, packed);
  }
}

template <std::size_t kBits, class D, typename T>
HYLE_FORCE_INLINE void bitpack_unpack_one(D d, const T* packed, T* raw) noexcept {
  if constexpr (sizeof(T) == 1) {
    Pack8<kBits>().Unpack(d, packed, raw);
  } else if constexpr (sizeof(T) == 2 && (kBits == 11 || kBits == 13)) {
    bitpack_hyle_unpack_u16<kBits>(d, packed, raw);
  } else if constexpr (sizeof(T) == 2) {
    Pack16<kBits>().Unpack(d, packed, raw);
  } else {
    Pack32<kBits>().Unpack(d, packed, raw);
  }
}

template <typename T, class D, std::size_t... Ks>
HYLE_FORCE_INLINE void bitpack_pack_blocks(unsigned bits, D d, std::size_t lanes,
                                           std::size_t nblocks, const T* raw, T* packed,
                                           std::index_sequence<Ks...>) noexcept {
  constexpr std::size_t B = 8 * sizeof(T);
  (void)(((bits == Ks + 1) ? (([&] {
            const std::size_t raw_step = lanes * B;
            const std::size_t packed_step = lanes * (Ks + 1);
            for (std::size_t b = 0; b < nblocks; ++b)
              bitpack_pack_one<Ks + 1>(d, raw + b * raw_step, packed + b * packed_step);
          }()), true) : false) || ...);
}

template <typename T, class D, std::size_t... Ks>
HYLE_FORCE_INLINE void bitpack_unpack_blocks(unsigned bits, D d, std::size_t lanes,
                                             std::size_t nblocks, const T* packed, T* raw,
                                             std::index_sequence<Ks...>) noexcept {
  constexpr std::size_t B = 8 * sizeof(T);
  (void)(((bits == Ks + 1) ? (([&] {
            const std::size_t raw_step = lanes * B;
            const std::size_t packed_step = lanes * (Ks + 1);
            for (std::size_t b = 0; b < nblocks; ++b)
              bitpack_unpack_one<Ks + 1>(d, packed + b * packed_step, raw + b * raw_step);
          }()), true) : false) || ...);
}

#if defined(HYLE_BITPACK_FIXED_128_LAYOUT)
// The serialized format is a sequence of 128-value logical blocks. A full scalable register
// contains `blocks_per_vector` such blocks, but PackN expects its input as B row-vectors. Transpose
// block-major -> row-major before PackN and undo the transpose afterwards. This uses every physical
// lane without changing a byte of the fixed layout.
template <std::size_t kBits, typename T>
HYLE_FORCE_INLINE void bitpack_pack_multiblock_one(const T* raw, std::size_t nblocks,
                                                   T* packed) noexcept {
  constexpr std::size_t B = 8 * sizeof(T);
  constexpr std::size_t kLogicalLanes = 16 / sizeof(T);
  constexpr std::size_t kMaxLanes = ScalableTag<T>::kMaxLanes;
  const ScalableTag<T> full;
  const CappedTag<T, kLogicalLanes> fixed;
  const std::size_t lanes = Lanes(full);
  const std::size_t blocks_per_vector = lanes / kLogicalLanes;
  alignas(64) T raw_rows[B * kMaxLanes];
  alignas(64) T packed_rows[B * kMaxLanes];

  std::size_t block = 0;
  for (; block + blocks_per_vector <= nblocks; block += blocks_per_vector) {
    for (std::size_t row = 0; row < B; ++row)
      for (std::size_t group = 0; group < blocks_per_vector; ++group)
        for (std::size_t lane = 0; lane < kLogicalLanes; ++lane)
          raw_rows[row * lanes + group * kLogicalLanes + lane] =
              raw[(block + group) * 128 + row * kLogicalLanes + lane];

    bitpack_pack_one<kBits>(full, raw_rows, packed_rows);

    for (std::size_t row = 0; row < kBits; ++row)
      for (std::size_t group = 0; group < blocks_per_vector; ++group)
        for (std::size_t lane = 0; lane < kLogicalLanes; ++lane)
          packed[(block + group) * (kBits * kLogicalLanes) + row * kLogicalLanes + lane] =
              packed_rows[row * lanes + group * kLogicalLanes + lane];
  }

  for (; block < nblocks; ++block)
    bitpack_pack_one<kBits>(fixed, raw + block * 128,
                           packed + block * (kBits * kLogicalLanes));
}

template <std::size_t kBits, typename T>
HYLE_FORCE_INLINE void bitpack_unpack_multiblock_one(const T* packed, std::size_t nblocks,
                                                     T* raw) noexcept {
  constexpr std::size_t B = 8 * sizeof(T);
  constexpr std::size_t kLogicalLanes = 16 / sizeof(T);
  constexpr std::size_t kMaxLanes = ScalableTag<T>::kMaxLanes;
  const ScalableTag<T> full;
  const CappedTag<T, kLogicalLanes> fixed;
  const std::size_t lanes = Lanes(full);
  const std::size_t blocks_per_vector = lanes / kLogicalLanes;
  alignas(64) T packed_rows[B * kMaxLanes];
  alignas(64) T raw_rows[B * kMaxLanes];

  std::size_t block = 0;
  for (; block + blocks_per_vector <= nblocks; block += blocks_per_vector) {
    for (std::size_t row = 0; row < kBits; ++row)
      for (std::size_t group = 0; group < blocks_per_vector; ++group)
        for (std::size_t lane = 0; lane < kLogicalLanes; ++lane)
          packed_rows[row * lanes + group * kLogicalLanes + lane] =
              packed[(block + group) * (kBits * kLogicalLanes) +
                     row * kLogicalLanes + lane];

    bitpack_unpack_one<kBits>(full, packed_rows, raw_rows);

    for (std::size_t row = 0; row < B; ++row)
      for (std::size_t group = 0; group < blocks_per_vector; ++group)
        for (std::size_t lane = 0; lane < kLogicalLanes; ++lane)
          raw[(block + group) * 128 + row * kLogicalLanes + lane] =
              raw_rows[row * lanes + group * kLogicalLanes + lane];
  }

  for (; block < nblocks; ++block)
    bitpack_unpack_one<kBits>(fixed, packed + block * (kBits * kLogicalLanes),
                             raw + block * 128);
}

template <typename T, std::size_t... Ks>
HYLE_FORCE_INLINE void bitpack_pack_multiblock(unsigned bits, const T* raw,
                                               std::size_t nblocks, T* packed,
                                               std::index_sequence<Ks...>) noexcept {
  (void)(((bits == Ks + 1)
              ? (bitpack_pack_multiblock_one<Ks + 1>(raw, nblocks, packed), true)
              : false) ||
          ...);
}

template <typename T, std::size_t... Ks>
HYLE_FORCE_INLINE void bitpack_unpack_multiblock(unsigned bits, const T* packed,
                                                 std::size_t nblocks, T* raw,
                                                 std::index_sequence<Ks...>) noexcept {
  (void)(((bits == Ks + 1)
              ? (bitpack_unpack_multiblock_one<Ks + 1>(packed, nblocks, raw), true)
              : false) ||
          ...);
}
#endif

#define HYLE_BITPACK_DEFINE(T, tag)                                                        \
  void HYLE_BITPACK_SYM(pack_##tag##_)(const T* raw, std::size_t count, T* packed,         \
                                       unsigned bits) noexcept {                            \
    constexpr std::size_t B = 8 * sizeof(T);                                               \
    /* Scalable vectors remain native sizeless values; only their active logical lanes */   \
    /* are capped. Loads/stores and all D-tag operations therefore execute 128 bits. */     \
    HYLE_BITPACK_TAG(T) d;                                                                  \
    const std::size_t lanes = Lanes(d);                                                     \
    const std::size_t block_size = lanes * B;                                               \
    const std::size_t nblocks = count / block_size;                                         \
    const std::size_t block_raw = nblocks * block_size;                                     \
    const std::size_t block_packed = nblocks * lanes * bits;                                \
    bitpack_pack_blocks<T>(bits, d, lanes, nblocks, raw, packed, std::make_index_sequence<B>{}); \
    bitstream_pack<T>(raw + block_raw, count - block_raw, packed + block_packed, bits);    \
  }                                                                                         \
  void HYLE_BITPACK_SYM(unpack_##tag##_)(const T* packed, std::size_t count, T* raw,       \
                                         unsigned bits) noexcept {                          \
    constexpr std::size_t B = 8 * sizeof(T);                                               \
    HYLE_BITPACK_TAG(T) d;                                                                  \
    const std::size_t lanes = Lanes(d);                                                     \
    const std::size_t block_size = lanes * B;                                               \
    const std::size_t nblocks = count / block_size;                                         \
    const std::size_t block_raw = nblocks * block_size;                                     \
    const std::size_t block_packed = nblocks * lanes * bits;                                \
    bitpack_unpack_blocks<T>(bits, d, lanes, nblocks, packed, raw, std::make_index_sequence<B>{}); \
    bitstream_unpack<T>(packed + block_packed, count - block_raw, raw + block_raw, bits);  \
  }

#if defined(HYLE_BITPACK_FIXED_128_LAYOUT)
#define HYLE_BITPACK_TAG(T) CappedTag<T, 16 / sizeof(T)>
#else
#define HYLE_BITPACK_TAG(T) ScalableTag<T>
#endif

HYLE_BITPACK_TYPES(HYLE_BITPACK_DEFINE)

#if defined(HYLE_BITPACK_FIXED_128_LAYOUT)
#define HYLE_BITPACK_DEFINE_MULTIBLOCK(T, tag)                                      \
  void HYLE_BITPACK_SYM(pack_##tag##_multiblock_)(                                  \
      const T* raw, std::size_t count, T* packed, unsigned bits) noexcept {          \
    constexpr std::size_t B = 8 * sizeof(T);                                        \
    constexpr std::size_t kLogicalLanes = 16 / sizeof(T);                           \
    const std::size_t nblocks = count / 128;                                        \
    const std::size_t block_raw = nblocks * 128;                                    \
    const std::size_t block_packed = nblocks * kLogicalLanes * bits;                \
    bitpack_pack_multiblock<T>(bits, raw, nblocks, packed,                          \
                               std::make_index_sequence<B>{});                       \
    bitstream_pack<T>(raw + block_raw, count - block_raw, packed + block_packed,     \
                      bits);                                                         \
  }                                                                                  \
  void HYLE_BITPACK_SYM(unpack_##tag##_multiblock_)(                                \
      const T* packed, std::size_t count, T* raw, unsigned bits) noexcept {          \
    constexpr std::size_t B = 8 * sizeof(T);                                        \
    constexpr std::size_t kLogicalLanes = 16 / sizeof(T);                           \
    const std::size_t nblocks = count / 128;                                        \
    const std::size_t block_raw = nblocks * 128;                                    \
    const std::size_t block_packed = nblocks * kLogicalLanes * bits;                \
    bitpack_unpack_multiblock<T>(bits, packed, nblocks, raw,                        \
                                 std::make_index_sequence<B>{});                     \
    bitstream_unpack<T>(packed + block_packed, count - block_raw, raw + block_raw,   \
                        bits);                                                       \
  }
HYLE_BITPACK_TYPES(HYLE_BITPACK_DEFINE_MULTIBLOCK)
#undef HYLE_BITPACK_DEFINE_MULTIBLOCK
#endif

#undef HYLE_BITPACK_TAG
#undef HYLE_BITPACK_DEFINE
#undef HYLE_BITPACK_SYM
#undef HYLE_BITPACK_PASTE
#undef HYLE_BITPACK_PASTE2

}  // namespace hyle::simd::detail
