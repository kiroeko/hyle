// Direct-Highway canonical bitpack kernels. Highway owns target lowering and
// Pack8/16/32; HYLE owns the fixed logical-block wire contract and whole-buffer
// dispatch boundary.

#include <cstddef>
#include <cstdint>
#include <utility>

#include "hwy/contrib/bit_pack/bit_pack-inl.h"
#include "hwy/highway.h"

#include "src/simd/bitpack/bitstream.h"
#include "src/simd/bitpack/kernels.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including bitpack/kernel_impl.h"
#endif

#define HYLE_BITPACK_CAT_(a, b) a##b
#define HYLE_BITPACK_CAT(a, b) HYLE_BITPACK_CAT_(a, b)
#define HYLE_BITPACK_FN(name) HYLE_BITPACK_CAT(name, HYLE_SIMD_VARIANT_SUFFIX)

namespace hyle::simd::detail {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

template <std::size_t kBits, class D, typename T>
HWY_INLINE void pack_one(D d, const T *raw, T *packed) noexcept {
  if constexpr (sizeof(T) == 1) {
    hn::Pack8<kBits>().Pack(d, raw, packed);
  } else if constexpr (sizeof(T) == 2) {
    hn::Pack16<kBits>().Pack(d, raw, packed);
  } else {
    hn::Pack32<kBits>().Pack(d, raw, packed);
  }
}

template <std::size_t kBits, class D, typename T>
HWY_INLINE void unpack_one(D d, const T *packed, T *raw) noexcept {
  if constexpr (sizeof(T) == 1) {
    hn::Pack8<kBits>().Unpack(d, packed, raw);
  } else if constexpr (sizeof(T) == 2) {
    hn::Pack16<kBits>().Unpack(d, packed, raw);
  } else {
    hn::Pack32<kBits>().Unpack(d, packed, raw);
  }
}

template <std::size_t kBits, typename T>
HWY_INLINE void pack_width(const T *raw, std::size_t count,
                           T *packed) noexcept {
  constexpr std::size_t W = 8 * sizeof(T);
  constexpr std::size_t kLogicalLanes = 16 / sizeof(T);
  constexpr std::size_t kBlockValues = W * kLogicalLanes;
  static_assert(kBlockValues == 128);

  const hn::CappedTag<T, kLogicalLanes> fixed;
  const std::size_t blocks = count / kBlockValues;
  for (std::size_t block = 0; block < blocks; ++block)
    pack_one<kBits>(fixed, raw + block * kBlockValues,
                    packed + block * kBits * kLogicalLanes);

  const std::size_t block_raw = blocks * kBlockValues;
  const std::size_t block_packed = blocks * kBits * kLogicalLanes;
  bitstream_pack(raw + block_raw, count - block_raw, packed + block_packed,
                 kBits);
}

template <std::size_t kBits, typename T>
HWY_INLINE void unpack_width(const T *packed, std::size_t count,
                             T *raw) noexcept {
  constexpr std::size_t W = 8 * sizeof(T);
  constexpr std::size_t kLogicalLanes = 16 / sizeof(T);
  constexpr std::size_t kBlockValues = W * kLogicalLanes;
  static_assert(kBlockValues == 128);

  const hn::CappedTag<T, kLogicalLanes> fixed;
  const std::size_t blocks = count / kBlockValues;
  for (std::size_t block = 0; block < blocks; ++block)
    unpack_one<kBits>(fixed, packed + block * kBits * kLogicalLanes,
                      raw + block * kBlockValues);

  const std::size_t block_raw = blocks * kBlockValues;
  const std::size_t block_packed = blocks * kBits * kLogicalLanes;
  bitstream_unpack(packed + block_packed, count - block_raw, raw + block_raw,
                   kBits);
}
template <typename T, std::size_t... Ks>
HWY_INLINE void pack_canonical(const T *raw, std::size_t count, T *packed,
                               unsigned bits,
                               std::index_sequence<Ks...>) noexcept {
  (void)(((bits == Ks + 1) ? (pack_width<Ks + 1>(raw, count, packed), true)
                           : false) ||
         ...);
}

template <typename T, std::size_t... Ks>
HWY_INLINE void unpack_canonical(const T *packed, std::size_t count, T *raw,
                                 unsigned bits,
                                 std::index_sequence<Ks...>) noexcept {
  (void)(((bits == Ks + 1) ? (unpack_width<Ks + 1>(packed, count, raw), true)
                           : false) ||
         ...);
}

} // namespace

#if HWY_TARGET == HWY_SCALAR
#define HYLE_BITPACK_DEFINE(T, tag)                                            \
  void HYLE_BITPACK_FN(pack_##tag##_)(const T *raw, std::size_t count,         \
                                      T *packed, unsigned bits) noexcept {     \
    canonical_scalar_pack(raw, count, packed, bits);                           \
  }                                                                            \
  void HYLE_BITPACK_FN(unpack_##tag##_)(const T *packed, std::size_t count,    \
                                        T *raw, unsigned bits) noexcept {      \
    canonical_scalar_unpack(packed, count, raw, bits);                         \
  }
#else
#define HYLE_BITPACK_DEFINE(T, tag)                                            \
  void HYLE_BITPACK_FN(pack_##tag##_)(const T *raw, std::size_t count,         \
                                      T *packed, unsigned bits) noexcept {     \
    pack_canonical(raw, count, packed, bits,                                   \
                   std::make_index_sequence<8 * sizeof(T)>{});                 \
  }                                                                            \
  void HYLE_BITPACK_FN(unpack_##tag##_)(const T *packed, std::size_t count,    \
                                        T *raw, unsigned bits) noexcept {      \
    unpack_canonical(packed, count, raw, bits,                                 \
                     std::make_index_sequence<8 * sizeof(T)>{});               \
  }
#endif

HYLE_BITPACK_TYPES(HYLE_BITPACK_DEFINE)
#undef HYLE_BITPACK_DEFINE

} // namespace hyle::simd::detail

#undef HYLE_BITPACK_FN
#undef HYLE_BITPACK_CAT
#undef HYLE_BITPACK_CAT_
