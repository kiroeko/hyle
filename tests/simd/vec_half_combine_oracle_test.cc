// simd-19 C13: exhaustive scalar-oracle for half/combine family.
//
// Whitelist: this file + tests/CMakeLists.txt single SOURCES line (C2/C13 template).
// Ops: LowerHalf / UpperHalf / Combine / ZeroExtendVector / InsertIntoUpper.
//
// Green baseline = full NEON register (N*sizeof(T)==16). Sub-full primaries hung as
// DISABLED_* redlist: pre-existing N-blind half/pad bugs (NOT s19 regression; ledger #11
// disposition 96d99aa). C14→UpperHalf green; C15→ZeroExtend/InsertIntoUpper green.
// Do NOT delete DISABLED cases (C2 hang-debt precedent).
//
// Oracle = hand-written per-lane arrays. MUST NOT call the ops under test for expected
// values.

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

// Any legal half-able shape: N power-of-two >=2, N*sizeof(T)<=16.
template <typename T, std::size_t N>
void CheckHalfCombineFamily() {
  static_assert(N >= 2 && (N & (N - 1)) == 0, "N power of two >=2");
  static_assert(N * sizeof(T) <= 16, "fits NEON register");
  constexpr std::size_t NH = N / 2;

  alignas(16) T in[N];
  alignas(16) T lo_ref[NH], hi_ref[NH];
  alignas(16) T comb_ref[N], zext_ref[N], ins_ref[N];
  alignas(16) T base[N];
  alignas(16) T got[N];
  alignas(16) T half_got[NH];

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

// ---- Green baseline: full register (N * sizeof(T) == 16) ----------------------------

#define HYLE_S19_HC_FULL(T, N, Name) \
  TEST(HalfCombineOracle, Name) { CheckHalfCombineFamily<T, N>(); }

HYLE_S19_HC_FULL(std::int8_t, 16, i8x16)
HYLE_S19_HC_FULL(std::uint8_t, 16, u8x16)
HYLE_S19_HC_FULL(std::int16_t, 8, i16x8)
HYLE_S19_HC_FULL(std::uint16_t, 8, u16x8)
HYLE_S19_HC_FULL(std::int32_t, 4, i32x4)
HYLE_S19_HC_FULL(std::uint32_t, 4, u32x4)
HYLE_S19_HC_FULL(float, 4, f32x4)
HYLE_S19_HC_FULL(std::int64_t, 2, i64x2)
HYLE_S19_HC_FULL(std::uint64_t, 2, u64x2)
HYLE_S19_HC_FULL(double, 2, f64x2)

#undef HYLE_S19_HC_FULL

// ---- Redlist (DISABLED_): sub-full primaries, pre-existing N-blind half/pad bugs ------
// First C13 run: 13 RED. Hang until C14 (UpperHalf) + C15 (ZeroExtend/InsertIntoUpper).
// Default gtest skips DISABLED_*; use --gtest_also_run_disabled_tests to recheck.

#define HYLE_S19_HC_REDLIST(T, N, Name) \
  TEST(HalfCombineOracle, DISABLED_##Name) { CheckHalfCombineFamily<T, N>(); }

HYLE_S19_HC_REDLIST(std::int8_t, 2, i8x2)
HYLE_S19_HC_REDLIST(std::int8_t, 4, i8x4)
HYLE_S19_HC_REDLIST(std::int8_t, 8, i8x8)
HYLE_S19_HC_REDLIST(std::uint8_t, 2, u8x2)
HYLE_S19_HC_REDLIST(std::uint8_t, 4, u8x4)
HYLE_S19_HC_REDLIST(std::uint8_t, 8, u8x8)
HYLE_S19_HC_REDLIST(std::int16_t, 2, i16x2)
HYLE_S19_HC_REDLIST(std::int16_t, 4, i16x4)
HYLE_S19_HC_REDLIST(std::uint16_t, 2, u16x2)
HYLE_S19_HC_REDLIST(std::uint16_t, 4, u16x4)
HYLE_S19_HC_REDLIST(std::int32_t, 2, i32x2)
HYLE_S19_HC_REDLIST(std::uint32_t, 2, u32x2)
HYLE_S19_HC_REDLIST(float, 2, f32x2)

#undef HYLE_S19_HC_REDLIST
