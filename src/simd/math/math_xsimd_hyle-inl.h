// Copyright (c) 2016, Johan Mabille, Sylvain Corlay, Wolf Vollprecht and Martin Renou
// Copyright (c) 2016, QuantStack
// Copyright (c) 2018, Serge Guelton
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// HYLE D-tag port of xsimd common math product-winning kernels.
// Origin: xsimd/arch/common/xsimd_common_math.hpp (+ trigo tanh_kernel)
// upstream SHA: 5d2490f515cf773f984da9ec0a1aec103d00d238
// Algorithm/constants preserved from xsimd; ops lowered through hyle::simd D-tag primitives.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "hyle/macros.h"
#include "hyle/simd/vec.h"
#include "src/simd/math/math_hyle-inl.h"

namespace hyle::simd::detail::xsimd_hyle {
using namespace hyle::simd;

// ---- bit constants / Horner -------------------------------------------------

template <class D>
HYLE_FORCE_INLINE VFromD<D> BitsF32(D d, std::uint32_t bits) noexcept {
  static_assert(std::is_same_v<TFromD<D>, float>, "BitsF32: float tag");
  float v;
  std::memcpy(&v, &bits, sizeof(v));
  return Set(d, v);
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> BitsF64(D d, std::uint64_t bits) noexcept {
  static_assert(std::is_same_v<TFromD<D>, double>, "BitsF64: double tag");
  double v;
  std::memcpy(&v, &bits, sizeof(v));
  return Set(d, v);
}

// xsimd horner: c0 + x*(c1 + x*(c2 + ...)) with hex IEEE bits (lowest degree first in
// template pack order; same order as xsimd's most-significant-first hex coeff lists).
template <class D, std::uint32_t c0>
HYLE_FORCE_INLINE VFromD<D> HornerF32(D d, VFromD<D>) noexcept {
  return BitsF32(d, c0);
}
template <class D, std::uint32_t c0, std::uint32_t c1, std::uint32_t... rest>
HYLE_FORCE_INLINE VFromD<D> HornerF32(D d, VFromD<D> x) noexcept {
  return MulAdd(x, HornerF32<D, c1, rest...>(d, x), BitsF32(d, c0));
}

template <class D, std::uint64_t c0>
HYLE_FORCE_INLINE VFromD<D> HornerF64(D d, VFromD<D>) noexcept {
  return BitsF64(d, c0);
}
template <class D, std::uint64_t c0, std::uint64_t c1, std::uint64_t... rest>
HYLE_FORCE_INLINE VFromD<D> HornerF64(D d, VFromD<D> x) noexcept {
  return MulAdd(x, HornerF64<D, c1, rest...>(d, x), BitsF64(d, c0));
}

// horner1 ends with an implicit leading 1.
template <class D>
HYLE_FORCE_INLINE VFromD<D> Horner1F32(D d, VFromD<D>) noexcept {
  return Set(d, 1.0f);
}
template <class D, std::uint32_t c0>
HYLE_FORCE_INLINE VFromD<D> Horner1F32(D d, VFromD<D> x) noexcept {
  return Add(x, BitsF32(d, c0));
}
template <class D, std::uint32_t c0, std::uint32_t c1, std::uint32_t... rest>
HYLE_FORCE_INLINE VFromD<D> Horner1F32(D d, VFromD<D> x) noexcept {
  return MulAdd(x, Horner1F32<D, c1, rest...>(d, x), BitsF32(d, c0));
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Horner1F64(D d, VFromD<D>) noexcept {
  return Set(d, 1.0);
}
template <class D, std::uint64_t c0>
HYLE_FORCE_INLINE VFromD<D> Horner1F64(D d, VFromD<D> x) noexcept {
  return Add(x, BitsF64(d, c0));
}
template <class D, std::uint64_t c0, std::uint64_t c1, std::uint64_t... rest>
HYLE_FORCE_INLINE VFromD<D> Horner1F64(D d, VFromD<D> x) noexcept {
  return MulAdd(x, Horner1F64<D, c1, rest...>(d, x), BitsF64(d, c0));
}

// self * 2^k via (k + maxexp) << nmb then BitCast multiply (xsimd ldexp).
template <class D, class VI>
HYLE_FORCE_INLINE VFromD<D> Ldexp(D d, VFromD<D> self, VI k_int) noexcept {
  using T = TFromD<D>;
  using TI = TFromV<VI>;
  static_assert(std::is_integral_v<TI> && std::is_signed_v<TI>, "Ldexp: signed int exponents");
  constexpr TI kMaxExp = std::is_same_v<T, float> ? TI{127} : TI{1023};
  constexpr int kNmb = std::is_same_v<T, float> ? 23 : 52;
  const RebindToSigned<D> di;
  const auto ik = ShiftLeft<kNmb>(Add(k_int, Set(di, kMaxExp)));
  return Mul(self, BitCast(d, ik));
}

// ---- helpers ----------------------------------------------------------------

template <class D, class V>
HYLE_FORCE_INLINE V BitOfSign(D d, V x) noexcept {
  return And(x, SignBit(d));
}

template <class D, class V>
HYLE_FORCE_INLINE auto IsFlint(D d, V x) noexcept {
  const V diff = Sub(x, x);
  const V frac = IfThenElse(IsNaN(diff), NaN(d), Sub(x, Trunc(x)));
  return Eq(frac, Zero(d));
}

template <class D, class V>
HYLE_FORCE_INLINE auto IsEven(D d, V x) noexcept {
  return IsFlint(d, Mul(x, Set(d, TFromD<D>{0.5})));
}

template <class D, class M>
HYLE_FORCE_INLINE bool AnyTrue(D d, M m) noexcept {
  return !AllFalse(d, m);
}

// nearbyint as signed int lanes. NearestInt/RoundToInt is f32->i32 only on some backends;
// ConvertTo(di, Round(k)) covers both f32 and f64.
template <class D, class V>
HYLE_FORCE_INLINE VFromD<RebindToSigned<D>> ToNearbyInt(D d, V k) noexcept {
  const RebindToSigned<D> di;
  return ConvertTo(di, Round(k));
}

// ---- exp family -------------------------------------------------------------

template <class D>
HYLE_FORCE_INLINE VFromD<D> ExpF32(D d, VFromD<D> self) noexcept {
  static_assert(std::is_same_v<TFromD<D>, float>);
  using V = VFromD<D>;
  const V invlog2 = Set(d, 1.4426950408889634f);
  const V log2hi = BitsF32(d, 0x3f318000u);
  const V log2lo = BitsF32(d, 0xb95e8083u);
  V k = Round(Mul(invlog2, self));
  V x = NegMulAdd(k, log2hi, self);
  x = NegMulAdd(k, log2lo, x);
  V y = HornerF32<D, 0x3f000000u, 0x3e2aa9a5u, 0x3d2aa957u, 0x3c098d8bu, 0x3ab778cfu>(d, x);
  x = Add(Set(d, 1.0f), MulAdd(y, Mul(x, x), x));
  x = IfThenElse(Le(self, Set(d, -88.3762626647949f)), Zero(d), Ldexp(d, x, ToNearbyInt(d, k)));
  x = IfThenElse(Ge(self, Set(d, 88.3762626647949f)), Inf(d), x);
  return x;
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Exp2F32(D d, VFromD<D> self) noexcept {
  static_assert(std::is_same_v<TFromD<D>, float>);
  using V = VFromD<D>;
  V k = Round(self);
  V x = Sub(self, k);
  V y = HornerF32<D, 0x3e75fdf1u, 0x3d6356ebu, 0x3c1d9422u, 0x3ab01218u, 0x3922c8c4u>(d, x);
  x = Add(Set(d, 1.0f),
          MulAdd(y, Mul(x, x), Mul(x, Set(d, 0.6931471805599453f))));
  x = IfThenElse(Le(self, Set(d, -127.0f)), Zero(d), Ldexp(d, x, ToNearbyInt(d, k)));
  x = IfThenElse(Ge(self, Set(d, 127.0f)), Inf(d), x);
  return x;
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Exp10F32(D d, VFromD<D> self) noexcept {
  static_assert(std::is_same_v<TFromD<D>, float>);
  using V = VFromD<D>;
  const V invlog10_2 = Set(d, 3.3219280948873623f);
  const V log10_2hi = BitsF32(d, 0x3e9a0000u);
  const V log10_2lo = BitsF32(d, 0x39826a14u);
  V k = Round(Mul(invlog10_2, self));
  V x = NegMulAdd(k, log10_2hi, self);
  x = Sub(x, Mul(k, log10_2lo));
  V y = HornerF32<D, 0x40135d8eu, 0x4029a926u, 0x400237dau, 0x3f95eb4cu, 0x3f0aacefu,
                  0x3e54dff1u>(d, x);
  x = Add(Set(d, 1.0f), Mul(y, x));
  x = IfThenElse(Le(self, Set(d, -37.89999771118164f)), Zero(d), Ldexp(d, x, ToNearbyInt(d, k)));
  x = IfThenElse(Ge(self, Set(d, 38.23080825805664f)), Inf(d), x);
  return x;
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> ExpF64(D d, VFromD<D> self) noexcept {
  static_assert(std::is_same_v<TFromD<D>, double>);
  using V = VFromD<D>;
  const V invlog2 = Set(d, 1.442695040888963407359924681001892137426645954152986);
  const V log2hi = BitsF64(d, 0x3fe62e42fee00000ull);
  const V log2lo = BitsF64(d, 0x3dea39ef35793c76ull);
  V k = Round(Mul(invlog2, self));
  V hi = NegMulAdd(k, log2hi, self);
  V lo = Mul(k, log2lo);
  V x = Sub(hi, lo);
  V t = Mul(x, x);
  V c = NegMulAdd(
      t,
      HornerF64<D, 0x3fc555555555553eull, 0xbf66c16c16bebd93ull, 0x3f11566aaf25de2cull,
                0xbebbbd41c5d26bf1ull, 0x3e66376972bea4d0ull>(d, t),
      x);
  const V two = Set(d, 2.0);
  const V one = Set(d, 1.0);
  c = Sub(one, Sub(Sub(lo, Div(Mul(x, c), Sub(two, c))), hi));
  c = IfThenElse(Le(self, Set(d, -708.3964185322641)), Zero(d), Ldexp(d, c, ToNearbyInt(d, k)));
  c = IfThenElse(Ge(self, Set(d, 709.78271289338400)), Inf(d), c);
  return c;
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Exp2F64(D d, VFromD<D> self) noexcept {
  static_assert(std::is_same_v<TFromD<D>, double>);
  using V = VFromD<D>;
  V k = Round(self);
  V x = Mul(Sub(self, k), Set(d, 0.6931471805599453094172321214581765680755001343602553));
  V t = Mul(x, x);
  V c = NegMulAdd(
      t,
      HornerF64<D, 0x3fc555555555553eull, 0xbf66c16c16bebd93ull, 0x3f11566aaf25de2cull,
                0xbebbbd41c5d26bf1ull, 0x3e66376972bea4d0ull>(d, t),
      x);
  const V two = Set(d, 2.0);
  const V one = Set(d, 1.0);
  c = Add(Add(one, x), Div(Mul(x, c), Sub(two, c)));
  c = IfThenElse(Le(self, Set(d, -1023.0)), Zero(d), Ldexp(d, c, ToNearbyInt(d, k)));
  c = IfThenElse(Ge(self, Set(d, 1023.0)), Inf(d), c);
  return c;
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Exp10F64(D d, VFromD<D> self) noexcept {
  static_assert(std::is_same_v<TFromD<D>, double>);
  using V = VFromD<D>;
  const V invlog10_2 = Set(d, 3.32192809488736234787031942949);
  const V log10_2hi = BitsF64(d, 0x3fd3440000000000ull);
  const V log10_2lo = BitsF64(d, 0x3ed3509f79fef312ull);
  V k = Round(Mul(invlog10_2, self));
  V x = NegMulAdd(k, log10_2hi, self);
  x = NegMulAdd(k, log10_2lo, x);
  V xx = Mul(x, x);
  V px = Mul(x, HornerF64<D, 0x40a2b4798e134a01ull, 0x40796b7a050349e4ull, 0x40277d9474c55934ull,
                          0x3fa4fd75f3062dd4ull>(d, xx));
  V x2 = Div(px, Sub(Horner1F64<D, 0x40a03f37650df6e2ull, 0x4093e05eefd67782ull,
                                0x405545fdce51ca08ull>(d, xx),
                     px));
  V c = Add(Set(d, 1.0), Add(x2, x2));
  c = IfThenElse(Le(self, Set(d, -308.2547155599167)), Zero(d), Ldexp(d, c, ToNearbyInt(d, k)));
  c = IfThenElse(Ge(self, Set(d, 308.2547155599167)), Inf(d), c);
  return c;
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Exp(D d, VFromD<D> x) noexcept {
  if constexpr (std::is_same_v<TFromD<D>, float>) return ExpF32(d, x);
  else return ExpF64(d, x);
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> Exp2(D d, VFromD<D> x) noexcept {
  if constexpr (std::is_same_v<TFromD<D>, float>) return Exp2F32(d, x);
  else return Exp2F64(d, x);
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> Exp10(D d, VFromD<D> x) noexcept {
  if constexpr (std::is_same_v<TFromD<D>, float>) return Exp10F32(d, x);
  else return Exp10F64(d, x);
}

// ---- tanh -------------------------------------------------------------------

template <class D>
HYLE_FORCE_INLINE VFromD<D> TanhF32(D d, VFromD<D> self) noexcept {
  static_assert(std::is_same_v<TFromD<D>, float>);
  using V = VFromD<D>;
  const V one = Set(d, 1.0f);
  const V x = Abs(self);
  const auto test = Lt(x, Set(d, 0.625f));
  const V bts = BitOfSign(d, self);
  V z = one;
  if (AnyTrue(d, test)) {
    const V sqrx = Mul(x, x);
    z = MulAdd(Mul(HornerF32<D, 0xbeaaaa99u, 0x3e088393u, 0xbd5c1e2du, 0x3ca9134eu, 0xbbbaf0eau>(d, sqrx),
                   sqrx),
               x, x);
    if (AllTrue(d, test)) return Xor(z, bts);
  }
  const V r = MulAdd(Set(d, -2.0f), Div(one, Add(one, ExpF32(d, Add(x, x)))), one);
  return Xor(IfThenElse(test, z, r), bts);
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> TanhF64(D d, VFromD<D> self) noexcept {
  static_assert(std::is_same_v<TFromD<D>, double>);
  using V = VFromD<D>;
  const V one = Set(d, 1.0);
  const V x = Abs(self);
  const auto test = Lt(x, Set(d, 0.625));
  const V bts = BitOfSign(d, self);
  V z = one;
  if (AnyTrue(d, test)) {
    const V sqrx = Mul(x, x);
    const V p = HornerF64<D, 0xc0993ac030580563ull, 0xc058d26a0e26682dull, 0xbfeedc5baafd6f4bull>(d, sqrx);
    const V q = Horner1F64<D, 0x40b2ec102442040cull, 0x40a176fa0e5535faull, 0x405c33f28a581B86ull>(d, sqrx);
    z = MulAdd(Div(Mul(sqrx, p), q), x, x);
    if (AllTrue(d, test)) return Xor(z, bts);
  }
  const V r = MulAdd(Set(d, -2.0), Div(one, Add(one, ExpF64(d, Add(x, x)))), one);
  return Xor(IfThenElse(test, z, r), bts);
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Tanh(D d, VFromD<D> x) noexcept {
  if constexpr (std::is_same_v<TFromD<D>, float>) return TanhF32(d, x);
  else return TanhF64(d, x);
}

// ---- erf / erfc -------------------------------------------------------------

template <class D>
HYLE_FORCE_INLINE VFromD<D> ErfF32(D d, VFromD<D> self) noexcept {
  static_assert(std::is_same_v<TFromD<D>, float>);
  using V = VFromD<D>;
  const V x = Abs(self);
  V r1 = Zero(d);
  const auto test1 = Lt(x, Set(d, 2.0f / 3.0f));
  if (AnyTrue(d, test1)) {
    r1 = Mul(self, HornerF32<D, 0x3f906ebau, 0xbec0937eu, 0x3de70f22u, 0xbcdb61f4u, 0x3ba4468du,
                             0xba1fc83bu>(d, Mul(x, x)));
    if (AllTrue(d, test1)) return r1;
  }
  V z = Div(x, Add(Set(d, 1.0f), x));
  z = Sub(z, Set(d, 0.4f));
  V r2 = Sub(Set(d, 1.0f),
             Mul(ExpF32(d, Neg(Mul(x, x))),
                 HornerF32<D, 0x3f0a0e8bu, 0xbf918a62u, 0x3e243828u, 0x3ec4ca6eu, 0x3e1175c7u,
                           0x3e2006f0u, 0xbfaea865u, 0x4050b063u, 0xc0cd1a85u, 0x40d67e3bu,
                           0xc0283611u>(d, z)));
  r2 = IfThenElse(Lt(self, Zero(d)), Neg(r2), r2);
  r1 = IfThenElse(test1, r1, r2);
  r1 = IfThenElse(IsInf(self), CopySign(Set(d, 1.0f), self), r1);
  return r1;
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> ErfF64(D d, VFromD<D> self) noexcept {
  static_assert(std::is_same_v<TFromD<D>, double>);
  using V = VFromD<D>;
  const V x = Abs(self);
  const V xx = Mul(x, x);
  const V lim1 = Set(d, 0.65);
  const V lim2 = Set(d, 2.2);
  const auto test1 = Lt(x, lim1);
  V r1 = Zero(d);
  if (AnyTrue(d, test1)) {
    const V num = HornerF64<D, 0x3ff20dd750429b61ull, 0x3fc16500f106c0a5ull, 0x3fa4a59a4f02579cull,
                            0x3f53b7664358865aull, 0x3f110512d5b20332ull>(d, xx);
    const V den = HornerF64<D, 0x3ff0000000000000ull, 0x3fdd0a84eb1ca867ull, 0x3fb64536ca92ea2full,
                            0x3f8166f75999dbd1ull, 0x3f37ea4332348252ull>(d, xx);
    r1 = Mul(self, Div(num, den));
    if (AllTrue(d, test1)) return r1;
  }
  const auto test2 = Lt(x, lim2);
  const auto test3 = And(test2, Not(test1));
  const V ex = ExpF64(d, Neg(xx));
  if (AnyTrue(d, test3)) {
    const V erfc2 =
        Div(HornerF64<D, 0x3feffffffbbb552bull, 0x3ff54dfe9b258a60ull, 0x3fec1986509e687bull,
                      0x3fd53dd7a67c7e9full, 0x3fb2488a6b5cb5e5ull, 0x3f7cf4cfe0aacbb4ull, 0x0ull>(d, x),
            HornerF64<D, 0x3ff0000000000000ull, 0x4003adeae79b9708ull, 0x40053b1052dca8bdull,
                      0x3ff9e677c2777c3cull, 0x3fe307622fcff772ull, 0x3fc033c113a7deeeull,
                      0x3f89a996639b0d00ull>(d, x));
    V z = Sub(Set(d, 1.0), Mul(ex, erfc2));
    V r2 = IfThenElse(Lt(self, Zero(d)), Neg(z), z);
    r1 = IfThenElse(test1, r1, r2);
    if (AllTrue(d, Or(test1, test3))) return r1;
  }
  const V erfc3 =
      Div(HornerF64<D, 0x3fefff5a9e697ae2ull, 0x3ff9fa202deb88e5ull, 0x3ff44744306832aeull,
                    0x3fe29be1cff90d94ull, 0x3fc42210f88b9d43ull, 0x3f971d0907ea7a92ull, 0x0ull>(d, x),
          HornerF64<D, 0x3ff0000000000000ull, 0x400602f24bf3fdb6ull, 0x400afd487397568full,
                    0x400315ffdfd5ce91ull, 0x3ff0cfd4cb6cde9full, 0x3fd1d7ab774bb837ull,
                    0x3fa47bd61bbb3843ull>(d, x));
  V z = Sub(Set(d, 1.0), Mul(ex, erfc3));
  z = IfThenElse(Lt(self, Zero(d)), Neg(z), z);
  z = IfThenElse(IsInf(self), CopySign(Set(d, 1.0), self), z);
  return IfThenElse(test2, r1, z);
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Erf(D d, VFromD<D> x) noexcept {
  if constexpr (std::is_same_v<TFromD<D>, float>) return ErfF32(d, x);
  else return ErfF64(d, x);
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> ErfcF32(D d, VFromD<D> self) noexcept {
  static_assert(std::is_same_v<TFromD<D>, float>);
  using V = VFromD<D>;
  const V x = Abs(self);
  const auto test0 = Lt(self, Zero(d));
  V r1 = Zero(d);
  const auto test1 = Lt(Mul(Set(d, 3.0f), x), Set(d, 2.0f));
  V z = Div(x, Add(Set(d, 1.0f), x));
  if (AnyTrue(d, test1)) {
    r1 = Mul(Sub(Set(d, 1.0f), z),
             HornerF32<D, 0x3f7ffffeu, 0xbe036d7eu, 0xbfa11698u, 0xbffc9284u, 0xc016c985u,
                       0x3f2cff3bu, 0xc010d956u, 0x401b5680u, 0x41aa8e55u>(d, z));
    if (AllTrue(d, test1)) {
      return IfThenElse(test0, Sub(Set(d, 2.0f), r1), r1);
    }
  }
  z = Sub(z, Set(d, 0.4f));
  V r2 = Mul(ExpF32(d, Neg(Mul(x, x))),
             HornerF32<D, 0x3f0a0e8bu, 0xbf918a62u, 0x3e243828u, 0x3ec4ca6eu, 0x3e1175c7u, 0x3e2006f0u,
                       0xbfaea865u, 0x4050b063u, 0xc0cd1a85u, 0x40d67e3bu, 0xc0283611u>(d, z));
  r1 = IfThenElse(test1, r1, r2);
  r1 = IfThenElse(Eq(x, Inf(d)), Zero(d), r1);
  return IfThenElse(test0, Sub(Set(d, 2.0f), r1), r1);
}

// Double erfc: 1 - erf (sufficient for product symbol; full multi-branch erfc not needed).
template <class D>
HYLE_FORCE_INLINE VFromD<D> ErfcF64(D d, VFromD<D> self) noexcept {
  static_assert(std::is_same_v<TFromD<D>, double>);
  return Sub(Set(d, 1.0), ErfF64(d, self));
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Erfc(D d, VFromD<D> x) noexcept {
  if constexpr (std::is_same_v<TFromD<D>, float>) return ErfcF32(d, x);
  else return ErfcF64(d, x);
}

// ---- simple binary / unary --------------------------------------------------

template <class D>
HYLE_FORCE_INLINE VFromD<D> HypotInRange(D d, VFromD<D> a, VFromD<D> b) noexcept {
  return Sqrt(MulAdd(a, a, Mul(b, b)));
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Fdim(D d, VFromD<D> a, VFromD<D> b) noexcept {
  return Max(Zero(d), Sub(a, b));
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Fmod(D d, VFromD<D> a, VFromD<D> b) noexcept {
  // fnma(trunc(a/b), b, a)
  return NegMulAdd(Trunc(Div(a, b)), b, a);
}

// ---- tgamma / stirling ------------------------------------------------------

template <class D>
HYLE_FORCE_INLINE VFromD<D> StirlingKernel(D d, VFromD<D> w) noexcept {
  if constexpr (std::is_same_v<TFromD<D>, float>) {
    return HornerF32<D, 0x3daaaaabu, 0x3b638e39u, 0xbb2fb930u, 0xb970b359u>(d, w);
  } else {
    return HornerF64<D, 0x3fb5555555555986ull, 0x3f6c71c71b98c5fdull, 0xbf65f72607d44fd7ull,
                     0xbf2e166b27e61d7cull, 0x3f49cc72592d7293ull>(d, w);
  }
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> StirlingSplitLimit(D d) noexcept {
  if constexpr (std::is_same_v<TFromD<D>, float>) return BitsF32(d, 0x41d628f6u);
  else return BitsF64(d, 0x4061e083ba3443d4ull);
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> StirlingLargeLimit(D d) noexcept {
  if constexpr (std::is_same_v<TFromD<D>, float>) return BitsF32(d, 0x420c28f3u);
  else return BitsF64(d, 0x4065800000000000ull);
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Stirling(D d, VFromD<D> a) noexcept {
  using V = VFromD<D>;
  using T = TFromD<D>;
  const V split = StirlingSplitLimit(d);
  const V large = StirlingLargeLimit(d);
  V x = IfThenElse(Ge(a, Zero(d)), a, NaN(d));
  V w = Div(Set(d, T{1}), x);
  w = MulAdd(w, StirlingKernel(d, w), Set(d, T{1}));
  V y = Exp(d, Neg(x));
  const auto test = Lt(x, split);
  V z = Sub(x, Set(d, T{0.5}));
  z = IfThenElse(test, z, Mul(Set(d, T{0.5}), z));
  // v = exp(z * log(|x|)) -- product Log (Highway) + product Exp (xsimd)
  V v = Exp(d, Mul(z, hyle::simd::detail::Log(d, Abs(x))));
  y = Mul(y, v);
  y = IfThenElse(test, y, Mul(y, v));
  V sqrt_2pi;
  if constexpr (std::is_same_v<T, float>) sqrt_2pi = BitsF32(d, 0x40206c99u);
  else sqrt_2pi = BitsF64(d, 0x40040d931ff62704ull);
  y = Mul(y, Mul(sqrt_2pi, w));
  y = IfThenElse(IsInf(x), x, y);
  return IfThenElse(Gt(x, large), Inf(d), y);
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> TgammaKernel(D d, VFromD<D> x) noexcept {
  if constexpr (std::is_same_v<TFromD<D>, float>) {
    return HornerF32<D, 0x3f800000u, 0x3ed87799u, 0x3ed2d411u, 0x3da82a34u, 0x3d93ae7cu, 0x3b91db14u,
                     0x3ba90c99u, 0x3ad28b22u>(d, x);
  } else {
    const auto num =
        HornerF64<D, 0x3ff0000000000000ull, 0x3fdfa1373993e312ull, 0x3fca8da9dcae7d31ull,
                  0x3fa863d918c423d3ull, 0x3f8557cde9db14b0ull, 0x3f5384e3e686bfabull,
                  0x3f24fcb839982153ull>(d, x);
    const auto den =
        HornerF64<D, 0x3ff0000000000000ull, 0x3fb24944c9cd3c51ull, 0xbfce071a9d4287c2ull,
                  0x3fa25779e33fde67ull, 0x3f8831ed5b1bb117ull, 0xbf7240e4e750b44aull,
                  0x3f41ae8a29152573ull, 0xbef8487a8400d3aFULL>(d, x);
    return Div(num, den);
  }
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> TgammaLargeNegative(D d, VFromD<D> a) noexcept {
  using V = VFromD<D>;
  using T = TFromD<D>;
  V st = Stirling(d, a);
  V p = Floor(a);
  V sgngam = IfThenElse(IsEven(d, p), Set(d, T{-1}), Set(d, T{1}));
  V z = Sub(a, p);
  const auto test2 = Lt(z, Set(d, T{0.5}));
  z = IfThenElse(test2, Sub(z, Set(d, T{1})), z);
  // sin(pi * z)
  V pi;
  if constexpr (std::is_same_v<T, float>) pi = BitsF32(d, 0x40490fdbu);
  else pi = BitsF64(d, 0x400921fb54442d18ull);
  z = Abs(Mul(a, hyle::simd::detail::Sin(d, Mul(z, pi))));
  return Div(Mul(sgngam, pi), Mul(z, st));
}

template <class D, class M>
HYLE_FORCE_INLINE VFromD<D> TgammaOther(D d, VFromD<D> a, M large_neg) noexcept {
  using V = VFromD<D>;
  using T = TFromD<D>;
  const V one = Set(d, T{1});
  const V two = Set(d, T{2});
  const V three = Set(d, T{3});
  V x = IfThenElse(large_neg, two, a);
  const auto inf_result = Eq(a, Inf(d));
  x = IfThenElse(inf_result, two, x);
  V z = one;

  // Fixed-iteration masked reduction (covers |a| < 33 domain of this path).
  for (int i = 0; i < 64; ++i) {
    const auto test1 = Ge(x, three);
    x = IfThenElse(test1, Sub(x, one), x);
    z = IfThenElse(test1, Mul(z, x), z);
  }
  for (int i = 0; i < 64; ++i) {
    const auto test1 = Lt(x, Zero(d));
    z = IfThenElse(test1, Div(z, x), z);
    x = IfThenElse(test1, Add(x, one), x);
  }
  for (int i = 0; i < 8; ++i) {
    const auto test2 = Lt(x, two);
    z = IfThenElse(test2, Div(z, x), z);
    x = IfThenElse(test2, Add(x, one), x);
  }
  x = Mul(z, TgammaKernel(d, Sub(x, two)));
  return IfThenElse(inf_result, a, x);
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Tgamma(D d, VFromD<D> self) noexcept {
  using V = VFromD<D>;
  using T = TFromD<D>;
  auto nan_result = And(Lt(self, Zero(d)), IsFlint(d, self));
  nan_result = Or(IsNaN(self), nan_result);
  V q = Abs(self);
  const auto test = Lt(self, Set(d, T{-33}));
  V r = NaN(d);
  if (AnyTrue(d, test)) {
    r = TgammaLargeNegative(d, q);
    if (AllTrue(d, test)) return IfThenElse(nan_result, NaN(d), r);
  }
  V r1 = TgammaOther(d, self, test);
  V r2 = IfThenElse(test, r, r1);
  const V signed_inf = CopySign(Inf(d), self);
  return IfThenElse(Eq(self, Zero(d)), signed_inf, IfThenElse(nan_result, NaN(d), r2));
}

}  // namespace hyle::simd::detail::xsimd_hyle
