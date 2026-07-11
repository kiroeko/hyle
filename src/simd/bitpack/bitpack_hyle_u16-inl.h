#pragma once

#include <cstddef>
#include <cstdint>

namespace hyle::simd::detail {

// Highway's 11/13-bit layouts are compact and fast, but the 13-bit packer scatters rawF with
// 0x7C00<<3 (=0xE000 after 16-bit truncation), not the intended 0x8000. Its inverse also reads
// several groups from the wrong vectors and shifts scatter bits before masking.
template <class D>
HWY_INLINE void bitpack_hyle_pack_u16_13(D d, const std::uint16_t* raw,
                                         std::uint16_t* packed) {
  using V = hn::Vec<decltype(d)>;
  const std::size_t n = hn::Lanes(d);
  const V r0 = hn::LoadU(d, raw + 0 * n);
  const V r1 = hn::LoadU(d, raw + 1 * n);
  const V r2 = hn::LoadU(d, raw + 2 * n);
  const V r3 = hn::LoadU(d, raw + 3 * n);
  const V r4 = hn::LoadU(d, raw + 4 * n);
  const V r5 = hn::LoadU(d, raw + 5 * n);
  const V r6 = hn::LoadU(d, raw + 6 * n);
  const V r7 = hn::LoadU(d, raw + 7 * n);
  const V r8 = hn::LoadU(d, raw + 8 * n);
  const V r9 = hn::LoadU(d, raw + 9 * n);
  const V rA = hn::LoadU(d, raw + 10 * n);
  const V rB = hn::LoadU(d, raw + 11 * n);
  const V rC = hn::LoadU(d, raw + 12 * n);
  const V rD = hn::LoadU(d, raw + 13 * n);
  const V rE = hn::LoadU(d, raw + 14 * n);
  const V rF = hn::LoadU(d, raw + 15 * n);

  const V lo8 = hn::Set(d, 0x00FFu);
  const V p0 = hn::OrAnd(hn::ShiftLeft<8>(r1), r0, lo8);
  const V p1 = hn::OrAnd(hn::ShiftLeft<8>(r3), r2, lo8);
  const V p2 = hn::OrAnd(hn::ShiftLeft<8>(r5), r4, lo8);
  const V p3 = hn::OrAnd(hn::ShiftLeft<8>(r7), r6, lo8);
  const V p4 = hn::OrAnd(hn::ShiftLeft<8>(r9), r8, lo8);
  const V p5 = hn::OrAnd(hn::ShiftLeft<8>(rB), rA, lo8);
  const V p6 = hn::OrAnd(hn::ShiftLeft<8>(rD), rC, lo8);
  const V p7 = hn::OrAnd(hn::ShiftLeft<8>(rF), rE, lo8);
  hn::StoreU(p0, d, packed + 0 * n); hn::StoreU(p1, d, packed + 1 * n);
  hn::StoreU(p2, d, packed + 2 * n); hn::StoreU(p3, d, packed + 3 * n);
  hn::StoreU(p4, d, packed + 4 * n); hn::StoreU(p5, d, packed + 5 * n);
  hn::StoreU(p6, d, packed + 6 * n); hn::StoreU(p7, d, packed + 7 * n);

  V mask = hn::Set(d, 0x03E0u);
  V p8 = hn::OrAnd(hn::ShiftRight<8>(r0), hn::ShiftRight<3>(r5), mask);
  V p9 = hn::OrAnd(hn::ShiftRight<8>(r1), hn::ShiftRight<3>(r6), mask);
  V pA = hn::OrAnd(hn::ShiftRight<8>(r2), hn::ShiftRight<3>(r7), mask);
  V pB = hn::OrAnd(hn::ShiftRight<8>(r3), hn::ShiftRight<3>(r8), mask);
  V pC = hn::OrAnd(hn::ShiftRight<8>(r4), hn::ShiftRight<3>(r9), mask);
  mask = hn::Set(d, 0x7C00u);
  p8 = hn::OrAnd(p8, hn::ShiftLeft<2>(rA), mask);
  p9 = hn::OrAnd(p9, hn::ShiftLeft<2>(rB), mask);
  pA = hn::OrAnd(pA, hn::ShiftLeft<2>(rC), mask);
  pB = hn::OrAnd(pB, hn::ShiftLeft<2>(rD), mask);
  pC = hn::OrAnd(pC, hn::ShiftLeft<2>(rE), mask);
  mask = hn::Set(d, 0x8000u);
  p8 = hn::OrAnd(p8, hn::ShiftLeft<7>(rF), mask);
  p9 = hn::OrAnd(p9, hn::ShiftLeft<6>(rF), mask);
  pA = hn::OrAnd(pA, hn::ShiftLeft<5>(rF), mask);
  pB = hn::OrAnd(pB, hn::ShiftLeft<4>(rF), mask);
  pC = hn::OrAnd(pC, hn::ShiftLeft<3>(rF), mask);
  hn::StoreU(p8, d, packed + 8 * n);  hn::StoreU(p9, d, packed + 9 * n);
  hn::StoreU(pA, d, packed + 10 * n); hn::StoreU(pB, d, packed + 11 * n);
  hn::StoreU(pC, d, packed + 12 * n);
}

template <unsigned kBits, class D>
HWY_INLINE void bitpack_hyle_pack_u16(D d, const std::uint16_t* raw,
                                      std::uint16_t* packed) {
  static_assert(kBits == 11 || kBits == 13);
  if constexpr (kBits == 11) hn::Pack16<11>().Pack(d, raw, packed);
  else bitpack_hyle_pack_u16_13(d, raw, packed);
}

template <class D>
HWY_INLINE void bitpack_hyle_unpack_u16_11(D d, const std::uint16_t* packed_in,
                                           std::uint16_t* raw) {
  using V = hn::Vec<decltype(d)>;
  const std::size_t n = hn::Lanes(d);
  const V p0 = hn::LoadU(d, packed_in + 0 * n);
  const V p1 = hn::LoadU(d, packed_in + 1 * n);
  const V p2 = hn::LoadU(d, packed_in + 2 * n);
  const V p3 = hn::LoadU(d, packed_in + 3 * n);
  const V p4 = hn::LoadU(d, packed_in + 4 * n);
  const V p5 = hn::LoadU(d, packed_in + 5 * n);
  const V p6 = hn::LoadU(d, packed_in + 6 * n);
  const V p7 = hn::LoadU(d, packed_in + 7 * n);
  const V p8 = hn::LoadU(d, packed_in + 8 * n);
  const V p9 = hn::LoadU(d, packed_in + 9 * n);
  const V pA = hn::LoadU(d, packed_in + 10 * n);

  const V lo8 = hn::Set(d, 0x00FFu);
  const V d0 = hn::And(p0, lo8);
  const V d1 = hn::ShiftRight<8>(p0);
  const V d2 = hn::And(p1, lo8);
  const V d3 = hn::ShiftRight<8>(p1);
  const V d4 = hn::And(p2, lo8);
  const V d5 = hn::ShiftRight<8>(p2);
  const V d6 = hn::And(p3, lo8);
  const V d7 = hn::ShiftRight<8>(p3);
  const V d8 = hn::And(p4, lo8);
  const V d9 = hn::ShiftRight<8>(p4);
  const V dA = hn::And(p5, lo8);
  const V dB = hn::ShiftRight<8>(p5);
  const V dC = hn::And(p6, lo8);
  const V dD = hn::ShiftRight<8>(p6);
  const V dE = hn::And(p7, lo8);
  const V dF = hn::ShiftRight<8>(p7);

  const V hi3 = hn::Set(d, 0x0700u);
  const V r0 = hn::OrAnd(d0, hn::ShiftLeft<8>(p8), hi3);
  const V r1 = hn::OrAnd(d1, hn::ShiftLeft<8>(p9), hi3);
  const V r2 = hn::OrAnd(d2, hn::ShiftLeft<8>(pA), hi3);
  const V r3 = hn::OrAnd(d3, hn::ShiftLeft<5>(p8), hi3);
  const V r4 = hn::OrAnd(d4, hn::ShiftLeft<5>(p9), hi3);
  const V r5 = hn::OrAnd(d5, hn::ShiftLeft<5>(pA), hi3);
  const V r6 = hn::OrAnd(d6, hn::ShiftLeft<2>(p8), hi3);
  const V r7 = hn::OrAnd(d7, hn::ShiftLeft<2>(p9), hi3);
  const V r8 = hn::OrAnd(d8, hn::ShiftLeft<2>(pA), hi3);
  const V r9 = hn::OrAnd(d9, hn::ShiftRight<1>(p8), hi3);
  const V rA = hn::OrAnd(dA, hn::ShiftRight<1>(p9), hi3);
  const V rB = hn::OrAnd(dB, hn::ShiftRight<1>(pA), hi3);
  const V rC = hn::OrAnd(dC, hn::ShiftRight<4>(p8), hi3);
  const V rD = hn::OrAnd(dD, hn::ShiftRight<4>(p9), hi3);
  const V rE = hn::OrAnd(dE, hn::ShiftRight<4>(pA), hi3);

  const V top = hn::Set(d, 0x8000u);
  const V f8 = hn::ShiftRight<7>(hn::And(p8, top));
  const V f9 = hn::ShiftRight<6>(hn::And(p9, top));
  const V fA = hn::ShiftRight<5>(hn::And(pA, top));
  const V rF = hn::Or(dF, hn::Xor3(f8, f9, fA));

  hn::StoreU(r0, d, raw + 0 * n);   hn::StoreU(r1, d, raw + 1 * n);
  hn::StoreU(r2, d, raw + 2 * n);   hn::StoreU(r3, d, raw + 3 * n);
  hn::StoreU(r4, d, raw + 4 * n);   hn::StoreU(r5, d, raw + 5 * n);
  hn::StoreU(r6, d, raw + 6 * n);   hn::StoreU(r7, d, raw + 7 * n);
  hn::StoreU(r8, d, raw + 8 * n);   hn::StoreU(r9, d, raw + 9 * n);
  hn::StoreU(rA, d, raw + 10 * n);  hn::StoreU(rB, d, raw + 11 * n);
  hn::StoreU(rC, d, raw + 12 * n);  hn::StoreU(rD, d, raw + 13 * n);
  hn::StoreU(rE, d, raw + 14 * n);  hn::StoreU(rF, d, raw + 15 * n);
}

template <class D>
HWY_INLINE void bitpack_hyle_unpack_u16_13(D d, const std::uint16_t* packed_in,
                                           std::uint16_t* raw) {
  using V = hn::Vec<decltype(d)>;
  const std::size_t n = hn::Lanes(d);
  const V p0 = hn::LoadU(d, packed_in + 0 * n);
  const V p1 = hn::LoadU(d, packed_in + 1 * n);
  const V p2 = hn::LoadU(d, packed_in + 2 * n);
  const V p3 = hn::LoadU(d, packed_in + 3 * n);
  const V p4 = hn::LoadU(d, packed_in + 4 * n);
  const V p5 = hn::LoadU(d, packed_in + 5 * n);
  const V p6 = hn::LoadU(d, packed_in + 6 * n);
  const V p7 = hn::LoadU(d, packed_in + 7 * n);
  const V p8 = hn::LoadU(d, packed_in + 8 * n);
  const V p9 = hn::LoadU(d, packed_in + 9 * n);
  const V pA = hn::LoadU(d, packed_in + 10 * n);
  const V pB = hn::LoadU(d, packed_in + 11 * n);
  const V pC = hn::LoadU(d, packed_in + 12 * n);

  const V lo8 = hn::Set(d, 0x00FFu);
  const V d0 = hn::And(p0, lo8);
  const V d1 = hn::ShiftRight<8>(p0);
  const V d2 = hn::And(p1, lo8);
  const V d3 = hn::ShiftRight<8>(p1);
  const V d4 = hn::And(p2, lo8);
  const V d5 = hn::ShiftRight<8>(p2);
  const V d6 = hn::And(p3, lo8);
  const V d7 = hn::ShiftRight<8>(p3);
  const V d8 = hn::And(p4, lo8);
  const V d9 = hn::ShiftRight<8>(p4);
  const V dA = hn::And(p5, lo8);
  const V dB = hn::ShiftRight<8>(p5);
  const V dC = hn::And(p6, lo8);
  const V dD = hn::ShiftRight<8>(p6);
  const V dE = hn::And(p7, lo8);
  const V dF = hn::ShiftRight<8>(p7);

  const V hi5 = hn::Set(d, 0x1F00u);
  const V r0 = hn::OrAnd(d0, hn::ShiftLeft<8>(p8), hi5);
  const V r1 = hn::OrAnd(d1, hn::ShiftLeft<8>(p9), hi5);
  const V r2 = hn::OrAnd(d2, hn::ShiftLeft<8>(pA), hi5);
  const V r3 = hn::OrAnd(d3, hn::ShiftLeft<8>(pB), hi5);
  const V r4 = hn::OrAnd(d4, hn::ShiftLeft<8>(pC), hi5);
  const V r5 = hn::OrAnd(d5, hn::ShiftLeft<3>(p8), hi5);
  const V r6 = hn::OrAnd(d6, hn::ShiftLeft<3>(p9), hi5);
  const V r7 = hn::OrAnd(d7, hn::ShiftLeft<3>(pA), hi5);
  const V r8 = hn::OrAnd(d8, hn::ShiftLeft<3>(pB), hi5);
  const V r9 = hn::OrAnd(d9, hn::ShiftLeft<3>(pC), hi5);
  const V rA = hn::OrAnd(dA, hn::ShiftRight<2>(p8), hi5);
  const V rB = hn::OrAnd(dB, hn::ShiftRight<2>(p9), hi5);
  const V rC = hn::OrAnd(dC, hn::ShiftRight<2>(pA), hi5);
  const V rD = hn::OrAnd(dD, hn::ShiftRight<2>(pB), hi5);
  const V rE = hn::OrAnd(dE, hn::ShiftRight<2>(pC), hi5);

  const V top = hn::Set(d, 0x8000u);
  const V f8 = hn::ShiftRight<7>(hn::And(p8, top));
  const V f9 = hn::ShiftRight<6>(hn::And(p9, top));
  const V fA = hn::ShiftRight<5>(hn::And(pA, top));
  const V fB = hn::ShiftRight<4>(hn::And(pB, top));
  const V fC = hn::ShiftRight<3>(hn::And(pC, top));
  const V rF = hn::Or(dF, hn::Or(hn::Xor3(f8, f9, fA), hn::Xor(fB, fC)));

  hn::StoreU(r0, d, raw + 0 * n);   hn::StoreU(r1, d, raw + 1 * n);
  hn::StoreU(r2, d, raw + 2 * n);   hn::StoreU(r3, d, raw + 3 * n);
  hn::StoreU(r4, d, raw + 4 * n);   hn::StoreU(r5, d, raw + 5 * n);
  hn::StoreU(r6, d, raw + 6 * n);   hn::StoreU(r7, d, raw + 7 * n);
  hn::StoreU(r8, d, raw + 8 * n);   hn::StoreU(r9, d, raw + 9 * n);
  hn::StoreU(rA, d, raw + 10 * n);  hn::StoreU(rB, d, raw + 11 * n);
  hn::StoreU(rC, d, raw + 12 * n);  hn::StoreU(rD, d, raw + 13 * n);
  hn::StoreU(rE, d, raw + 14 * n);  hn::StoreU(rF, d, raw + 15 * n);
}

template <unsigned kBits, class D>
HWY_INLINE void bitpack_hyle_unpack_u16(D d, const std::uint16_t* packed,
                                        std::uint16_t* raw) {
  static_assert(kBits == 11 || kBits == 13);
  if constexpr (kBits == 11) bitpack_hyle_unpack_u16_11(d, packed, raw);
  else bitpack_hyle_unpack_u16_13(d, packed, raw);
}

}  // namespace hyle::simd::detail
