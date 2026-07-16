#pragma once

#include <cstddef>
#include <cstdint>

namespace hyle::simd::detail {

// The corrected 11/13-bit layouts retain the compact vector block form. The 13-bit packer uses
// the intended 0x8000 rawF scatter mask, and both inverses select their source vectors before
// shifting and masking.
template <class D>
HYLE_FORCE_INLINE void bitpack_hyle_pack_u16_13(D d, const std::uint16_t* raw,
                                         std::uint16_t* packed) {
  using V = VFromD<decltype(d)>;
  const std::size_t n = Lanes(d);
  const V r0 = LoadU(d, raw + 0 * n);
  const V r1 = LoadU(d, raw + 1 * n);
  const V r2 = LoadU(d, raw + 2 * n);
  const V r3 = LoadU(d, raw + 3 * n);
  const V r4 = LoadU(d, raw + 4 * n);
  const V r5 = LoadU(d, raw + 5 * n);
  const V r6 = LoadU(d, raw + 6 * n);
  const V r7 = LoadU(d, raw + 7 * n);
  const V r8 = LoadU(d, raw + 8 * n);
  const V r9 = LoadU(d, raw + 9 * n);
  const V rA = LoadU(d, raw + 10 * n);
  const V rB = LoadU(d, raw + 11 * n);
  const V rC = LoadU(d, raw + 12 * n);
  const V rD = LoadU(d, raw + 13 * n);
  const V rE = LoadU(d, raw + 14 * n);
  const V rF = LoadU(d, raw + 15 * n);

  const V lo8 = Set(d, 0x00FFu);
  const V p0 = OrAnd(ShiftLeft<8>(r1), r0, lo8);
  const V p1 = OrAnd(ShiftLeft<8>(r3), r2, lo8);
  const V p2 = OrAnd(ShiftLeft<8>(r5), r4, lo8);
  const V p3 = OrAnd(ShiftLeft<8>(r7), r6, lo8);
  const V p4 = OrAnd(ShiftLeft<8>(r9), r8, lo8);
  const V p5 = OrAnd(ShiftLeft<8>(rB), rA, lo8);
  const V p6 = OrAnd(ShiftLeft<8>(rD), rC, lo8);
  const V p7 = OrAnd(ShiftLeft<8>(rF), rE, lo8);
  StoreU(p0, d, packed + 0 * n); StoreU(p1, d, packed + 1 * n);
  StoreU(p2, d, packed + 2 * n); StoreU(p3, d, packed + 3 * n);
  StoreU(p4, d, packed + 4 * n); StoreU(p5, d, packed + 5 * n);
  StoreU(p6, d, packed + 6 * n); StoreU(p7, d, packed + 7 * n);

  V mask = Set(d, 0x03E0u);
  V p8 = OrAnd(ShiftRight<8>(r0), ShiftRight<3>(r5), mask);
  V p9 = OrAnd(ShiftRight<8>(r1), ShiftRight<3>(r6), mask);
  V pA = OrAnd(ShiftRight<8>(r2), ShiftRight<3>(r7), mask);
  V pB = OrAnd(ShiftRight<8>(r3), ShiftRight<3>(r8), mask);
  V pC = OrAnd(ShiftRight<8>(r4), ShiftRight<3>(r9), mask);
  mask = Set(d, 0x7C00u);
  p8 = OrAnd(p8, ShiftLeft<2>(rA), mask);
  p9 = OrAnd(p9, ShiftLeft<2>(rB), mask);
  pA = OrAnd(pA, ShiftLeft<2>(rC), mask);
  pB = OrAnd(pB, ShiftLeft<2>(rD), mask);
  pC = OrAnd(pC, ShiftLeft<2>(rE), mask);
  mask = Set(d, 0x8000u);
  p8 = OrAnd(p8, ShiftLeft<7>(rF), mask);
  p9 = OrAnd(p9, ShiftLeft<6>(rF), mask);
  pA = OrAnd(pA, ShiftLeft<5>(rF), mask);
  pB = OrAnd(pB, ShiftLeft<4>(rF), mask);
  pC = OrAnd(pC, ShiftLeft<3>(rF), mask);
  StoreU(p8, d, packed + 8 * n);  StoreU(p9, d, packed + 9 * n);
  StoreU(pA, d, packed + 10 * n); StoreU(pB, d, packed + 11 * n);
  StoreU(pC, d, packed + 12 * n);
}

template <unsigned kBits, class D>
HYLE_FORCE_INLINE void bitpack_hyle_pack_u16(D d, const std::uint16_t* raw,
                                      std::uint16_t* packed) {
  static_assert(kBits == 11 || kBits == 13);
  if constexpr (kBits == 11) Pack16<11>().Pack(d, raw, packed);
  else bitpack_hyle_pack_u16_13(d, raw, packed);
}

template <class D>
HYLE_FORCE_INLINE void bitpack_hyle_unpack_u16_11(D d, const std::uint16_t* packed_in,
                                           std::uint16_t* raw) {
  using V = VFromD<decltype(d)>;
  const std::size_t n = Lanes(d);
  const V p0 = LoadU(d, packed_in + 0 * n);
  const V p1 = LoadU(d, packed_in + 1 * n);
  const V p2 = LoadU(d, packed_in + 2 * n);
  const V p3 = LoadU(d, packed_in + 3 * n);
  const V p4 = LoadU(d, packed_in + 4 * n);
  const V p5 = LoadU(d, packed_in + 5 * n);
  const V p6 = LoadU(d, packed_in + 6 * n);
  const V p7 = LoadU(d, packed_in + 7 * n);
  const V p8 = LoadU(d, packed_in + 8 * n);
  const V p9 = LoadU(d, packed_in + 9 * n);
  const V pA = LoadU(d, packed_in + 10 * n);

  const V lo8 = Set(d, 0x00FFu);
  const V d0 = And(p0, lo8);
  const V d1 = ShiftRight<8>(p0);
  const V d2 = And(p1, lo8);
  const V d3 = ShiftRight<8>(p1);
  const V d4 = And(p2, lo8);
  const V d5 = ShiftRight<8>(p2);
  const V d6 = And(p3, lo8);
  const V d7 = ShiftRight<8>(p3);
  const V d8 = And(p4, lo8);
  const V d9 = ShiftRight<8>(p4);
  const V dA = And(p5, lo8);
  const V dB = ShiftRight<8>(p5);
  const V dC = And(p6, lo8);
  const V dD = ShiftRight<8>(p6);
  const V dE = And(p7, lo8);
  const V dF = ShiftRight<8>(p7);

  const V hi3 = Set(d, 0x0700u);
  const V r0 = OrAnd(d0, ShiftLeft<8>(p8), hi3);
  const V r1 = OrAnd(d1, ShiftLeft<8>(p9), hi3);
  const V r2 = OrAnd(d2, ShiftLeft<8>(pA), hi3);
  const V r3 = OrAnd(d3, ShiftLeft<5>(p8), hi3);
  const V r4 = OrAnd(d4, ShiftLeft<5>(p9), hi3);
  const V r5 = OrAnd(d5, ShiftLeft<5>(pA), hi3);
  const V r6 = OrAnd(d6, ShiftLeft<2>(p8), hi3);
  const V r7 = OrAnd(d7, ShiftLeft<2>(p9), hi3);
  const V r8 = OrAnd(d8, ShiftLeft<2>(pA), hi3);
  const V r9 = OrAnd(d9, ShiftRight<1>(p8), hi3);
  const V rA = OrAnd(dA, ShiftRight<1>(p9), hi3);
  const V rB = OrAnd(dB, ShiftRight<1>(pA), hi3);
  const V rC = OrAnd(dC, ShiftRight<4>(p8), hi3);
  const V rD = OrAnd(dD, ShiftRight<4>(p9), hi3);
  const V rE = OrAnd(dE, ShiftRight<4>(pA), hi3);

  const V top = Set(d, 0x8000u);
  const V f8 = ShiftRight<7>(And(p8, top));
  const V f9 = ShiftRight<6>(And(p9, top));
  const V fA = ShiftRight<5>(And(pA, top));
  const V rF = Or(dF, Xor3(f8, f9, fA));

  StoreU(r0, d, raw + 0 * n);   StoreU(r1, d, raw + 1 * n);
  StoreU(r2, d, raw + 2 * n);   StoreU(r3, d, raw + 3 * n);
  StoreU(r4, d, raw + 4 * n);   StoreU(r5, d, raw + 5 * n);
  StoreU(r6, d, raw + 6 * n);   StoreU(r7, d, raw + 7 * n);
  StoreU(r8, d, raw + 8 * n);   StoreU(r9, d, raw + 9 * n);
  StoreU(rA, d, raw + 10 * n);  StoreU(rB, d, raw + 11 * n);
  StoreU(rC, d, raw + 12 * n);  StoreU(rD, d, raw + 13 * n);
  StoreU(rE, d, raw + 14 * n);  StoreU(rF, d, raw + 15 * n);
}

template <class D>
HYLE_FORCE_INLINE void bitpack_hyle_unpack_u16_13(D d, const std::uint16_t* packed_in,
                                           std::uint16_t* raw) {
  using V = VFromD<decltype(d)>;
  const std::size_t n = Lanes(d);
  const V p0 = LoadU(d, packed_in + 0 * n);
  const V p1 = LoadU(d, packed_in + 1 * n);
  const V p2 = LoadU(d, packed_in + 2 * n);
  const V p3 = LoadU(d, packed_in + 3 * n);
  const V p4 = LoadU(d, packed_in + 4 * n);
  const V p5 = LoadU(d, packed_in + 5 * n);
  const V p6 = LoadU(d, packed_in + 6 * n);
  const V p7 = LoadU(d, packed_in + 7 * n);
  const V p8 = LoadU(d, packed_in + 8 * n);
  const V p9 = LoadU(d, packed_in + 9 * n);
  const V pA = LoadU(d, packed_in + 10 * n);
  const V pB = LoadU(d, packed_in + 11 * n);
  const V pC = LoadU(d, packed_in + 12 * n);

  const V lo8 = Set(d, 0x00FFu);
  const V d0 = And(p0, lo8);
  const V d1 = ShiftRight<8>(p0);
  const V d2 = And(p1, lo8);
  const V d3 = ShiftRight<8>(p1);
  const V d4 = And(p2, lo8);
  const V d5 = ShiftRight<8>(p2);
  const V d6 = And(p3, lo8);
  const V d7 = ShiftRight<8>(p3);
  const V d8 = And(p4, lo8);
  const V d9 = ShiftRight<8>(p4);
  const V dA = And(p5, lo8);
  const V dB = ShiftRight<8>(p5);
  const V dC = And(p6, lo8);
  const V dD = ShiftRight<8>(p6);
  const V dE = And(p7, lo8);
  const V dF = ShiftRight<8>(p7);

  const V hi5 = Set(d, 0x1F00u);
  const V r0 = OrAnd(d0, ShiftLeft<8>(p8), hi5);
  const V r1 = OrAnd(d1, ShiftLeft<8>(p9), hi5);
  const V r2 = OrAnd(d2, ShiftLeft<8>(pA), hi5);
  const V r3 = OrAnd(d3, ShiftLeft<8>(pB), hi5);
  const V r4 = OrAnd(d4, ShiftLeft<8>(pC), hi5);
  const V r5 = OrAnd(d5, ShiftLeft<3>(p8), hi5);
  const V r6 = OrAnd(d6, ShiftLeft<3>(p9), hi5);
  const V r7 = OrAnd(d7, ShiftLeft<3>(pA), hi5);
  const V r8 = OrAnd(d8, ShiftLeft<3>(pB), hi5);
  const V r9 = OrAnd(d9, ShiftLeft<3>(pC), hi5);
  const V rA = OrAnd(dA, ShiftRight<2>(p8), hi5);
  const V rB = OrAnd(dB, ShiftRight<2>(p9), hi5);
  const V rC = OrAnd(dC, ShiftRight<2>(pA), hi5);
  const V rD = OrAnd(dD, ShiftRight<2>(pB), hi5);
  const V rE = OrAnd(dE, ShiftRight<2>(pC), hi5);

  const V top = Set(d, 0x8000u);
  const V f8 = ShiftRight<7>(And(p8, top));
  const V f9 = ShiftRight<6>(And(p9, top));
  const V fA = ShiftRight<5>(And(pA, top));
  const V fB = ShiftRight<4>(And(pB, top));
  const V fC = ShiftRight<3>(And(pC, top));
  const V rF = Or(dF, Or(Xor3(f8, f9, fA), Xor(fB, fC)));

  StoreU(r0, d, raw + 0 * n);   StoreU(r1, d, raw + 1 * n);
  StoreU(r2, d, raw + 2 * n);   StoreU(r3, d, raw + 3 * n);
  StoreU(r4, d, raw + 4 * n);   StoreU(r5, d, raw + 5 * n);
  StoreU(r6, d, raw + 6 * n);   StoreU(r7, d, raw + 7 * n);
  StoreU(r8, d, raw + 8 * n);   StoreU(r9, d, raw + 9 * n);
  StoreU(rA, d, raw + 10 * n);  StoreU(rB, d, raw + 11 * n);
  StoreU(rC, d, raw + 12 * n);  StoreU(rD, d, raw + 13 * n);
  StoreU(rE, d, raw + 14 * n);  StoreU(rF, d, raw + 15 * n);
}

template <unsigned kBits, class D>
HYLE_FORCE_INLINE void bitpack_hyle_unpack_u16(D d, const std::uint16_t* packed,
                                        std::uint16_t* raw) {
  static_assert(kBits == 11 || kBits == 13);
  if constexpr (kBits == 11) bitpack_hyle_unpack_u16_11(d, packed, raw);
  else bitpack_hyle_unpack_u16_13(d, packed, raw);
}

}  // namespace hyle::simd::detail
