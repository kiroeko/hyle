/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

// Finite AppleClang/NEON HYLE winners retained by simd-v3.  The functions use
// locked Highway target-native values but keep only the five qualified bodies;
// this is not another portable register API.

#if !defined(HYLE_SIMD_TARGET_NEON)
#error "hyle_winners_neon-inl.h requires HYLE_SIMD_TARGET_NEON"
#endif

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#include <arm_neon.h>

#include "hyle/core/compiler.h"
#include "hwy/cache_control.h"
#include "hwy/highway.h"

namespace hyle::simd::provider::hyle_winner::neon {

namespace hn = ::hwy::HWY_NAMESPACE;

using I16 = hn::Vec128<std::int16_t>;
using I32 = hn::Vec128<std::int32_t>;
using U64 = hn::Vec128<std::uint64_t>;
using MI32 = hn::Mask128<std::int32_t>;
using MU64 = hn::Mask128<std::uint64_t>;

namespace detail {

alignas(16) inline constexpr std::uint32_t kMaskWeights32[4] = {1, 2, 4,
                                                                 8};

HYLE_FORCE_INLINE unsigned mask_bits(MI32 mask) noexcept {
  return static_cast<unsigned>(vaddvq_u32(
      vandq_u32(mask.raw, vld1q_u32(kMaskWeights32))));
}

template <bool kExpand>
consteval auto make_i32_indices() {
  std::array<std::array<std::uint8_t, 16>, 16> table{};
  for (std::size_t bits = 0; bits < table.size(); ++bits) {
    auto& row = table[bits];
    if constexpr (kExpand) {
      row.fill(0xFF);
      std::size_t packed = 0;
      for (std::size_t lane = 0; lane < 4; ++lane) {
        if ((bits & (std::size_t{1} << lane)) == 0) continue;
        for (std::size_t byte = 0; byte < 4; ++byte) {
          row[lane * 4 + byte] =
              static_cast<std::uint8_t>(packed * 4 + byte);
        }
        ++packed;
      }
    } else {
      std::size_t packed = 0;
      for (std::size_t lane = 0; lane < 4; ++lane) {
        if ((bits & (std::size_t{1} << lane)) == 0) continue;
        for (std::size_t byte = 0; byte < 4; ++byte) {
          row[packed * 4 + byte] =
              static_cast<std::uint8_t>(lane * 4 + byte);
        }
        ++packed;
      }
      for (std::size_t lane = 0; lane < 4; ++lane) {
        if ((bits & (std::size_t{1} << lane)) != 0) continue;
        for (std::size_t byte = 0; byte < 4; ++byte) {
          row[packed * 4 + byte] =
              static_cast<std::uint8_t>(lane * 4 + byte);
        }
        ++packed;
      }
    }
  }
  return table;
}

inline constexpr auto kCompressI32 = make_i32_indices<false>();
inline constexpr auto kExpandI32 = make_i32_indices<true>();

HYLE_FORCE_INLINE I32 compress_i32(I32 value, MI32 mask) noexcept {
  const unsigned bits = mask_bits(mask);
  const uint8x16_t bytes = vreinterpretq_u8_s32(value.raw);
  return I32{vreinterpretq_s32_u8(
      vqtbl1q_u8(bytes, vld1q_u8(kCompressI32[bits].data())))};
}

}  // namespace detail

HYLE_FORCE_INLINE I16 demote_i32_to_i16(I32 a, I32 b) noexcept {
  return I16{vcombine_s16(vqmovn_s32(a.raw), vqmovn_s32(b.raw))};
}

HYLE_FORCE_INLINE I32 expand_i32(I32 packed, MI32 mask) noexcept {
  const unsigned bits = detail::mask_bits(mask);
  const uint8x16_t bytes = vreinterpretq_u8_s32(packed.raw);
  return I32{vreinterpretq_s32_u8(
      vqtbl1q_u8(bytes, vld1q_u8(detail::kExpandI32[bits].data())))};
}

HYLE_FORCE_INLINE I32 load_expand_i32(MI32 mask,
                                      const std::int32_t* packed) noexcept {
  return expand_i32(I32{vld1q_s32(packed)}, mask);
}

HYLE_FORCE_INLINE std::size_t compress_blended_store_i32(
    I32 value, MI32 mask, std::int32_t* out) noexcept {
  vst1q_s32(out, detail::compress_i32(value, mask).raw);
  return static_cast<std::size_t>(std::popcount(detail::mask_bits(mask)));
}

HYLE_FORCE_INLINE U64 compress_not_u64(U64 value, MU64 mask) noexcept {
  const uint64x2_t mask0 = vdupq_n_u64(vgetq_lane_u64(mask.raw, 0));
  const uint64x2_t mask1 = vdupq_n_u64(vgetq_lane_u64(mask.raw, 1));
  const uint64x2_t swap = vbicq_u64(mask0, mask1);
  const uint8x16_t bytes = vreinterpretq_u8_u64(value.raw);
  return U64{vreinterpretq_u64_u8(vbslq_u8(
      vreinterpretq_u8_u64(swap), vextq_u8(bytes, bytes, 8), bytes))};
}

// AppleClang's locked Highway Stream adds a write-prefetch that the qualified
// large working-set workload proved materially harmful.  Plain Store is the
// retained NEON body.  FlushStream is currently a no-op on AArch64, but keeping
// it paired with the store preserves the lifecycle contract and permits the
// x86 route to use its required sfence.
template <class V, class D>
HYLE_FORCE_INLINE void stream(V value, D d,
                              hn::TFromD<D>* aligned) noexcept {
  hn::Store(value, d, aligned);
}

HYLE_FORCE_INLINE void flush_stream() noexcept { ::hwy::FlushStream(); }

}  // namespace hyle::simd::provider::hyle_winner::neon
