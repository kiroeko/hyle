// Length-agnostic SVE/RVV conversions for the four common f16/bf16 <-> f32 batch APIs.
// Storage lanes remain the public POD bit carriers; scalable registers never cross this TU.

#include "hyle/platform.h"

#if (defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)) || \
    (defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT))

#include "src/simd/dtype/kernels.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(HYLE_ARCH_ARM64) && !defined(HYLE_SIMD_TARGET_SVE)
#define HYLE_SIMD_TARGET_SVE
#elif defined(HYLE_ARCH_RISCV64) && !defined(HYLE_SIMD_TARGET_RVV)
#define HYLE_SIMD_TARGET_RVV
#endif
#include "hyle/simd/vec.h"

namespace hyle::simd::detail {
namespace {

#if defined(HYLE_ARCH_ARM64)
#define HYLE_SCALABLE_DTYPE_SUFFIX sve
#else
#define HYLE_SCALABLE_DTYPE_SUFFIX rvv
#endif
#define HYLE_SCALABLE_DTYPE_NAME_I(name, suffix) name##_##suffix
#define HYLE_SCALABLE_DTYPE_NAME_II(name, suffix) HYLE_SCALABLE_DTYPE_NAME_I(name, suffix)
#define HYLE_SCALABLE_DTYPE_NAME(name) \
  HYLE_SCALABLE_DTYPE_NAME_II(name, HYLE_SCALABLE_DTYPE_SUFFIX)

static_assert(sizeof(float16_t) == sizeof(std::uint16_t));
static_assert(sizeof(bfloat16_t) == sizeof(std::uint16_t));
static_assert(std::is_trivially_copyable_v<float16_t>);
static_assert(std::is_trivially_copyable_v<bfloat16_t>);

}  // namespace

void HYLE_SCALABLE_DTYPE_NAME(f16_to_f32)(const float16_t* src, float* dst,
                                           std::size_t n) noexcept {
  const ScalableTag<float> df;
  const Rebind<std::uint32_t, decltype(df)> du32;
  const Rebind<std::uint16_t, decltype(df)> du16;
  const std::size_t w = Lanes(df);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto h = PromoteTo(du32, LoadU(du16, reinterpret_cast<const std::uint16_t*>(src + i)));
    const auto sign = ShiftLeft<16>(And(h, Set(du32, 0x8000u)));
    const auto be = And(ShiftRight<10>(h), Set(du32, 0x1Fu));
    const auto mant = And(h, Set(du32, 0x3FFu));

    const auto be32 = IfThenElse(Eq(be, Set(du32, 31u)), Set(du32, 255u),
                                 Add(be, Set(du32, 112u)));
    const auto normal_bits = Or(sign, Or(ShiftLeft<23>(be32), ShiftLeft<13>(mant)));

    // Half subnormals are mantissa * 2^-24. This is exact in f32; applying the sign in the
    // integer domain also preserves negative zero.
    const auto sub_abs = Mul(ConvertTo(df, mant), Set(df, 0x1p-24f));
    const auto sub_bits = Or(BitCast(du32, sub_abs), sign);
    StoreU(BitCast(df, IfThenElse(Eq(be, Zero(du32)), sub_bits, normal_bits)), df, dst + i);
  }
  for (; i < n; ++i) dst[i] = hyle::simd::f16_to_f32(src[i]);
}

void HYLE_SCALABLE_DTYPE_NAME(f32_to_f16)(const float* src, float16_t* dst,
                                           std::size_t n) noexcept {
  const ScalableTag<float> df;
  const Rebind<std::uint32_t, decltype(df)> du32;
  const Rebind<std::uint16_t, decltype(df)> du16;
  const std::size_t w = Lanes(df);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto u = BitCast(du32, LoadU(df, src + i));
    const auto absv = And(u, Set(du32, 0x7FFFFFFFu));
    const auto be = And(ShiftRight<23>(absv), Set(du32, 0xFFu));
    const auto exceptional = Or(Lt(be, Set(du32, 113u)), Gt(be, Set(du32, 142u)));
    if (!AllFalse(du32, exceptional)) {
      for (std::size_t lane = 0; lane < w; ++lane)
        dst[i + lane] = hyle::simd::f32_to_f16(src[i + lane]);
      continue;
    }

    // Normal finite half range: RNE the discarded 13 bits, then rebias 127 -> 15.
    const auto lsb = And(ShiftRight<13>(absv), Set(du32, 1u));
    const auto rounded = Add(absv, Add(Set(du32, 0xFFFu), lsb));
    const auto payload = Sub(ShiftRight<13>(rounded), Set(du32, 0x1C000u));
    const auto sign = ShiftRight<16>(And(u, Set(du32, 0x80000000u)));
    StoreU(DemoteTo(du16, Or(sign, payload)), du16,
           reinterpret_cast<std::uint16_t*>(dst + i));
  }
  for (; i < n; ++i) dst[i] = hyle::simd::f32_to_f16(src[i]);
}

void HYLE_SCALABLE_DTYPE_NAME(bf16_to_f32)(const bfloat16_t* src, float* dst,
                                            std::size_t n) noexcept {
  const ScalableTag<float> df;
  const Rebind<std::uint32_t, decltype(df)> du32;
  const Rebind<std::uint16_t, decltype(df)> du16;
  const std::size_t w = Lanes(df);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto bits = ShiftLeft<16>(
        PromoteTo(du32, LoadU(du16, reinterpret_cast<const std::uint16_t*>(src + i))));
    StoreU(BitCast(df, bits), df, dst + i);
  }
  for (; i < n; ++i) dst[i] = hyle::simd::bf16_to_f32(src[i]);
}

void HYLE_SCALABLE_DTYPE_NAME(f32_to_bf16)(const float* src, bfloat16_t* dst,
                                            std::size_t n) noexcept {
  const ScalableTag<float> df;
  const Rebind<std::uint32_t, decltype(df)> du32;
  const Rebind<std::uint16_t, decltype(df)> du16;
  const std::size_t w = Lanes(df);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto u = BitCast(du32, LoadU(df, src + i));
    const auto absv = And(u, Set(du32, 0x7FFFFFFFu));
    const auto finite = Lt(absv, Set(du32, 0x7F800000u));
    const auto is_nan = Gt(absv, Set(du32, 0x7F800000u));
    const auto lsb = And(ShiftRight<16>(u), Set(du32, 1u));
    const auto incr = IfThenElseZero(finite, Add(Set(du32, 0x7FFFu), lsb));
    const auto sticky = IfThenElseZero(is_nan, Set(du32, 1u << 6));
    const auto out = Or(sticky, ShiftRight<16>(Add(u, incr)));
    StoreU(DemoteTo(du16, out), du16, reinterpret_cast<std::uint16_t*>(dst + i));
  }
  for (; i < n; ++i) dst[i] = hyle::simd::f32_to_bf16(src[i]);
}

#undef HYLE_SCALABLE_DTYPE_NAME
#undef HYLE_SCALABLE_DTYPE_NAME_II
#undef HYLE_SCALABLE_DTYPE_NAME_I
#undef HYLE_SCALABLE_DTYPE_SUFFIX

}  // namespace hyle::simd::detail

#endif
