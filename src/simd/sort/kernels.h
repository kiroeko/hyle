#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/types.h"

// Internal per-target whole-buffer ordering entry points. Generated target TUs
// define one suffix each; no vendor vector or order tag crosses this boundary.
namespace hyle::simd::detail {

#define HYLE_SORT_KEY_TYPES(X) \
  X(std::uint16_t, u16)        \
  X(std::int16_t, i16)         \
  X(std::uint32_t, u32)        \
  X(std::int32_t, i32)         \
  X(std::uint64_t, u64)        \
  X(std::int64_t, i64)         \
  X(float, f32)                \
  X(double, f64)

#define HYLE_SORT_COMPOSITE_TYPES(X) \
  X(UInt128, u128)                    \
  X(KeyValue32, kv32)                 \
  X(KeyValue64, kv64)

#define HYLE_SORT_TYPES(X) \
  HYLE_SORT_KEY_TYPES(X)   \
  HYLE_SORT_COMPOSITE_TYPES(X)

// Compatibility names consumed by the bounded Highway adaptation. They are
// aliases, not representation-copy structs, so public object lifetime remains
// unchanged across the kernel boundary.
using U128Bits = UInt128;
using KV32Bits = KeyValue32;
using KV64Bits = KeyValue64;
#define HYLE_SORT_U128_TYPES(X, tier) X(U128Bits, u128, tier)
#define HYLE_SORT_KV_TYPES(X, tier) \
  X(KV32Bits, kv32, tier)                 \
  X(KV64Bits, kv64, tier)

#define HYLE_SORT_DECLARE_ONE(T, tag, suffix)                                  \
  void sort_##tag##_##suffix(T*, std::size_t, bool) noexcept;                  \
  void psort_##tag##_##suffix(T*, std::size_t, std::size_t, bool) noexcept;    \
  void select_##tag##_##suffix(T*, std::size_t, std::size_t, bool) noexcept;

#define HYLE_SORT_DECLARE_scalar(T, tag) HYLE_SORT_DECLARE_ONE(T, tag, scalar)
#define HYLE_SORT_DECLARE_sse4(T, tag) HYLE_SORT_DECLARE_ONE(T, tag, sse4)
#define HYLE_SORT_DECLARE_avx2(T, tag) HYLE_SORT_DECLARE_ONE(T, tag, avx2)
#define HYLE_SORT_DECLARE_avx512(T, tag) HYLE_SORT_DECLARE_ONE(T, tag, avx512)
#define HYLE_SORT_DECLARE_neon(T, tag) HYLE_SORT_DECLARE_ONE(T, tag, neon)
#define HYLE_SORT_DECLARE_sve(T, tag) HYLE_SORT_DECLARE_ONE(T, tag, sve)
#define HYLE_SORT_DECLARE_rvv(T, tag) HYLE_SORT_DECLARE_ONE(T, tag, rvv)

HYLE_SORT_TYPES(HYLE_SORT_DECLARE_scalar)
HYLE_SORT_TYPES(HYLE_SORT_DECLARE_sse4)
HYLE_SORT_TYPES(HYLE_SORT_DECLARE_avx2)
HYLE_SORT_TYPES(HYLE_SORT_DECLARE_avx512)
HYLE_SORT_TYPES(HYLE_SORT_DECLARE_neon)
HYLE_SORT_TYPES(HYLE_SORT_DECLARE_sve)
HYLE_SORT_TYPES(HYLE_SORT_DECLARE_rvv)

#undef HYLE_SORT_DECLARE_rvv
#undef HYLE_SORT_DECLARE_sve
#undef HYLE_SORT_DECLARE_neon
#undef HYLE_SORT_DECLARE_avx512
#undef HYLE_SORT_DECLARE_avx2
#undef HYLE_SORT_DECLARE_sse4
#undef HYLE_SORT_DECLARE_scalar
#undef HYLE_SORT_DECLARE_ONE

}  // namespace hyle::simd::detail
