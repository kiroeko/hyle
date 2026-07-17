// Exhaustive scalar oracle for the half/combine family. Full-register and sub-full
// shapes are permanent default-on regressions for LowerHalf, UpperHalf, Combine,
// ZeroExtendVector, and InsertIntoUpper. Expected values are computed only with
// hand-written scalar array operations, never with the SIMD operations under test.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "hyle/simd/vec.h"

namespace hs = hyle::simd;

namespace {

template <typename T>
T LanePattern(std::size_t i) {
  if constexpr (std::is_same_v<T, float>) return static_cast<float>(i) + 0.25f;
  else if constexpr (std::is_same_v<T, double>) return static_cast<double>(i) + 0.5;
  else if constexpr (std::is_signed_v<T>)
    return static_cast<T>(static_cast<std::int64_t>(i) * 3 - 17);
  else
    return static_cast<T>(0xA0u + static_cast<unsigned>(i) * 3u);
}

template <typename T>
void ExpectLaneEq(T got, T want, const char* what, std::size_t i, std::size_t N) {
  EXPECT_EQ(got, want) << what << " lane " << i << " N=" << N
                       << " sizeof(T)=" << sizeof(T);
}

// Scalar oracles (array only — never call ops under test).
template <typename T, std::size_t N>
void OracleLower(const T* in, T* out) {
  for (std::size_t i = 0; i < N / 2; ++i) out[i] = in[i];
}
template <typename T, std::size_t N>
void OracleUpper(const T* in, T* out) {
  constexpr std::size_t NH = N / 2;
  for (std::size_t i = 0; i < NH; ++i) out[i] = in[NH + i];
}
template <typename T, std::size_t N>
void OracleCombine(const T* hi, const T* lo, T* out) {
  constexpr std::size_t NH = N / 2;
  for (std::size_t i = 0; i < NH; ++i) out[i] = lo[i];
  for (std::size_t i = 0; i < NH; ++i) out[NH + i] = hi[i];
}
template <typename T, std::size_t N>
void OracleZeroExtend(const T* lo, T* out) {
  constexpr std::size_t NH = N / 2;
  for (std::size_t i = 0; i < NH; ++i) out[i] = lo[i];
  for (std::size_t i = NH; i < N; ++i) out[i] = static_cast<T>(0);
}
template <typename T, std::size_t N>
void OracleInsertUpper(const T* hi, const T* v, T* out) {
  constexpr std::size_t NH = N / 2;
  for (std::size_t i = 0; i < NH; ++i) out[i] = v[i];
  for (std::size_t i = 0; i < NH; ++i) out[NH + i] = hi[i];
}

// Any legal half-able fixed-width shape: N power-of-two >=2 and no wider than
// this target's native register. This includes sub-XMM logical vectors on SSE4,
// XMM partials on AVX2, and XMM/YMM partials on AVX-512.
template <typename T, std::size_t N>
void CheckHalfCombineFamily() {
  static_assert(N >= 2 && (N & (N - 1)) == 0, "N power of two >=2");
  static_assert(N * sizeof(T) <= hs::kNativeVectorBytes, "fits native register");
  constexpr std::size_t NH = N / 2;

  alignas(64) T in[N];
  alignas(64) T lo_ref[NH], hi_ref[NH];
  alignas(64) T comb_ref[N], zext_ref[N], ins_ref[N];
  alignas(64) T base[N];
  alignas(64) T got[N];
  alignas(64) T half_got[NH];

  for (std::size_t i = 0; i < N; ++i) {
    in[i] = LanePattern<T>(i);
    base[i] = LanePattern<T>(100 + i);
  }
  OracleLower<T, N>(in, lo_ref);
  OracleUpper<T, N>(in, hi_ref);
  OracleCombine<T, N>(hi_ref, lo_ref, comb_ref);
  OracleZeroExtend<T, N>(lo_ref, zext_ref);
  OracleInsertUpper<T, N>(lo_ref, base, ins_ref);

  const hs::FixedTag<T, N> d{};
  // Half tag with kPow2=0 so Store matches (Half tags are kPow2=-1).
  const hs::FixedTag<T, NH> dh0{};

  const auto v = hs::Load(d, in);
  const auto lo = hs::LowerHalf(dh0, v);
  const auto hi = hs::UpperHalf(dh0, v);

  std::memset(half_got, 0xCD, sizeof(half_got));
  hs::Store(lo, dh0, half_got);
  for (std::size_t i = 0; i < NH; ++i)
    ExpectLaneEq(half_got[i], lo_ref[i], "LowerHalf", i, N);

  std::memset(half_got, 0xCD, sizeof(half_got));
  hs::Store(hs::LowerHalf(v), dh0, half_got);
  for (std::size_t i = 0; i < NH; ++i)
    ExpectLaneEq(half_got[i], lo_ref[i], "LowerHalf(no-tag)", i, N);

  std::memset(half_got, 0xCD, sizeof(half_got));
  hs::Store(hi, dh0, half_got);
  for (std::size_t i = 0; i < NH; ++i)
    ExpectLaneEq(half_got[i], hi_ref[i], "UpperHalf", i, N);

  std::memset(half_got, 0xCD, sizeof(half_got));
  hs::Store(hs::UpperHalf(v), dh0, half_got);
  for (std::size_t i = 0; i < NH; ++i)
    ExpectLaneEq(half_got[i], hi_ref[i], "UpperHalf(no-tag)", i, N);

  // Combine(d, hi, lo): Highway arg order (hi, lo); memory layout lo|hi.
  std::memset(got, 0xCD, sizeof(got));
  hs::Store(hs::Combine(d, hi, lo), d, got);
  for (std::size_t i = 0; i < N; ++i)
    ExpectLaneEq(got[i], comb_ref[i], "Combine", i, N);

  std::memset(got, 0xCD, sizeof(got));
  hs::Store(hs::ZeroExtendVector(d, lo), d, got);
  for (std::size_t i = 0; i < N; ++i)
    ExpectLaneEq(got[i], zext_ref[i], "ZeroExtendVector", i, N);

  const auto bv = hs::Load(d, base);
  std::memset(got, 0xCD, sizeof(got));
  hs::Store(hs::InsertIntoUpper(d, lo, bv), d, got);
  for (std::size_t i = 0; i < N; ++i)
    ExpectLaneEq(got[i], ins_ref[i], "InsertIntoUpper", i, N);

  // ZeroExtend of zero half → all zero.
  const auto zlo = hs::LowerHalf(dh0, hs::Zero(d));
  std::memset(got, 0xCD, sizeof(got));
  hs::Store(hs::ZeroExtendVector(d, zlo), d, got);
  for (std::size_t i = 0; i < N; ++i)
    ExpectLaneEq(got[i], static_cast<T>(0), "ZeroExtend(zero)", i, N);
}

}  // namespace

// ---- 128-bit and sub-128-bit shapes -------------------------------------------------

#define HYLE_SIMD_TEST_INSTANTIATE_FULL_HALF_COMBINE(T, N, Name) \
  TEST(HalfCombineOracle, Name) { CheckHalfCombineFamily<T, N>(); }

HYLE_SIMD_TEST_INSTANTIATE_FULL_HALF_COMBINE(std::int8_t, 16, i8x16)
HYLE_SIMD_TEST_INSTANTIATE_FULL_HALF_COMBINE(std::uint8_t, 16, u8x16)
HYLE_SIMD_TEST_INSTANTIATE_FULL_HALF_COMBINE(std::int16_t, 8, i16x8)
HYLE_SIMD_TEST_INSTANTIATE_FULL_HALF_COMBINE(std::uint16_t, 8, u16x8)
HYLE_SIMD_TEST_INSTANTIATE_FULL_HALF_COMBINE(std::int32_t, 4, i32x4)
HYLE_SIMD_TEST_INSTANTIATE_FULL_HALF_COMBINE(std::uint32_t, 4, u32x4)
HYLE_SIMD_TEST_INSTANTIATE_FULL_HALF_COMBINE(float, 4, f32x4)
HYLE_SIMD_TEST_INSTANTIATE_FULL_HALF_COMBINE(std::int64_t, 2, i64x2)
HYLE_SIMD_TEST_INSTANTIATE_FULL_HALF_COMBINE(std::uint64_t, 2, u64x2)
HYLE_SIMD_TEST_INSTANTIATE_FULL_HALF_COMBINE(double, 2, f64x2)

#undef HYLE_SIMD_TEST_INSTANTIATE_FULL_HALF_COMBINE

// ---- Sub-full primary vectors ------------------------------------------------------
// These cases originally exposed N-blind half/pad bugs. They are permanent
// default-on regressions now that UpperHalf/ZeroExtend/InsertIntoUpper are fixed.

#define HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(T, N, Name) \
  TEST(HalfCombineOracle, Name) { CheckHalfCombineFamily<T, N>(); }

HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(std::int8_t, 2, i8x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(std::int8_t, 4, i8x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(std::int8_t, 8, i8x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(std::uint8_t, 2, u8x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(std::uint8_t, 4, u8x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(std::uint8_t, 8, u8x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(std::int16_t, 2, i16x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(std::int16_t, 4, i16x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(std::uint16_t, 2, u16x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(std::uint16_t, 4, u16x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(std::int32_t, 2, i32x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(std::uint32_t, 2, u32x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE(float, 2, f32x2)

#undef HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_HALF_COMBINE

#if defined(HYLE_SIMD_TARGET_AVX2) || defined(HYLE_SIMD_TARGET_AVX512)

// ---- 256-bit shapes (native on AVX2, partial on AVX-512) ----------------------------

#define HYLE_SIMD_TEST_INSTANTIATE_256_HALF_COMBINE(T, N, Name) \
  TEST(HalfCombineOracle, Name) { CheckHalfCombineFamily<T, N>(); }

HYLE_SIMD_TEST_INSTANTIATE_256_HALF_COMBINE(std::int8_t, 32, i8x32)
HYLE_SIMD_TEST_INSTANTIATE_256_HALF_COMBINE(std::uint8_t, 32, u8x32)
HYLE_SIMD_TEST_INSTANTIATE_256_HALF_COMBINE(std::int16_t, 16, i16x16)
HYLE_SIMD_TEST_INSTANTIATE_256_HALF_COMBINE(std::uint16_t, 16, u16x16)
HYLE_SIMD_TEST_INSTANTIATE_256_HALF_COMBINE(std::int32_t, 8, i32x8)
HYLE_SIMD_TEST_INSTANTIATE_256_HALF_COMBINE(std::uint32_t, 8, u32x8)
HYLE_SIMD_TEST_INSTANTIATE_256_HALF_COMBINE(float, 8, f32x8)
HYLE_SIMD_TEST_INSTANTIATE_256_HALF_COMBINE(std::int64_t, 4, i64x4)
HYLE_SIMD_TEST_INSTANTIATE_256_HALF_COMBINE(std::uint64_t, 4, u64x4)
HYLE_SIMD_TEST_INSTANTIATE_256_HALF_COMBINE(double, 4, f64x4)

#undef HYLE_SIMD_TEST_INSTANTIATE_256_HALF_COMBINE

#endif

#if defined(HYLE_SIMD_TARGET_AVX512)

// ---- 512-bit native shapes ----------------------------------------------------------

#define HYLE_SIMD_TEST_INSTANTIATE_512_HALF_COMBINE(T, N, Name) \
  TEST(HalfCombineOracle, Name) { CheckHalfCombineFamily<T, N>(); }

HYLE_SIMD_TEST_INSTANTIATE_512_HALF_COMBINE(std::int8_t, 64, i8x64)
HYLE_SIMD_TEST_INSTANTIATE_512_HALF_COMBINE(std::uint8_t, 64, u8x64)
HYLE_SIMD_TEST_INSTANTIATE_512_HALF_COMBINE(std::int16_t, 32, i16x32)
HYLE_SIMD_TEST_INSTANTIATE_512_HALF_COMBINE(std::uint16_t, 32, u16x32)
HYLE_SIMD_TEST_INSTANTIATE_512_HALF_COMBINE(std::int32_t, 16, i32x16)
HYLE_SIMD_TEST_INSTANTIATE_512_HALF_COMBINE(std::uint32_t, 16, u32x16)
HYLE_SIMD_TEST_INSTANTIATE_512_HALF_COMBINE(float, 16, f32x16)
HYLE_SIMD_TEST_INSTANTIATE_512_HALF_COMBINE(std::int64_t, 8, i64x8)
HYLE_SIMD_TEST_INSTANTIATE_512_HALF_COMBINE(std::uint64_t, 8, u64x8)
HYLE_SIMD_TEST_INSTANTIATE_512_HALF_COMBINE(double, 8, f64x8)

#undef HYLE_SIMD_TEST_INSTANTIATE_512_HALF_COMBINE

#endif
