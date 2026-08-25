#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/platform.h"  // HYLE_ARCH_X64 / HYLE_ARCH_ARM64 (borrow cores are arch-specific)

// kernels.h (internal) -- per-tier vqsort symbols the sort dispatcher takes the address of.
// Same shape as math/kernels.h: the scalar TU (std::sort/partial_sort/nth_element) is ALWAYS
// defined + is the non-x86 / fallback floor (NOT a stub -- it is the correctness oracle). The
// wide cores are a HYLE port of Highway vqsort (src/simd/sort/vqsort_hyle/), compiled per ISA
// tier on the matching arch (x86 avx2/avx512, arm64 neon); sort_dispatch enters them only when
// the CPU supports the tier (cpu_features()), so the wide codegen never runs on an older CPU.
// Not a public header.
//
// The compile-time SortAscending/SortDescending order tag is chosen INSIDE each TU from the
// runtime `ascending` bool -- so one symbol per (type,op,tier) covers both directions (the tag
// is a zero-size type; the branch folds to two static instantiations, no dispatch cost).

namespace hyle::simd::detail {

// X-macro over the scalar key types: <C++ type>, <short tag>.
#define HYLE_SORT_KEY_TYPES(X)   \
  X(std::uint16_t, u16)          \
  X(std::int16_t, i16)           \
  X(std::uint32_t, u32)          \
  X(std::int32_t, i32)           \
  X(std::uint64_t, u64)          \
  X(std::int64_t, i64)           \
  X(float, f32)                  \
  X(double, f64)

// Declares the sort/partial_sort/select trio for one (type, tier). `ascending` picks the order
// tag inside the TU; `k` is in units of keys (partial_sort: [0,k) sorted, k<=n; select: k-th
// placed, k<n). n==0 is a no-op; other size-contract violations abort in the public wrapper.
#define HYLE_SORT_DECL_TRIO(T, tag, tier)                                                   \
  void sort_##tag##_##tier(T* keys, std::size_t n, bool ascending) noexcept;                \
  void psort_##tag##_##tier(T* keys, std::size_t n, std::size_t k, bool ascending) noexcept;\
  void select_##tag##_##tier(T* keys, std::size_t n, std::size_t k, bool ascending) noexcept;

// scalar floor (std::) -- always available, every arch.
#define HYLE_SORT_DECL_SCALAR(T, tag) HYLE_SORT_DECL_TRIO(T, tag, scalar)
HYLE_SORT_KEY_TYPES(HYLE_SORT_DECL_SCALAR)
#undef HYLE_SORT_DECL_SCALAR

#if defined(HYLE_ARCH_X64)
// AVX2 (256-bit) + AVX-512 (512-bit) HYLE vqsort port symbols. Product CMake omits the AVX-512
// instantiation for the classic MSVC frontend and sort_dispatch binds that product tier to AVX2;
// clang-cl and GNU-like frontends compile and dispatch the 512-bit core normally.
#define HYLE_SORT_DECL_AVX2(T, tag) HYLE_SORT_DECL_TRIO(T, tag, avx2)
HYLE_SORT_KEY_TYPES(HYLE_SORT_DECL_AVX2)
#undef HYLE_SORT_DECL_AVX2
#define HYLE_SORT_DECL_AVX512(T, tag) HYLE_SORT_DECL_TRIO(T, tag, avx512)
HYLE_SORT_KEY_TYPES(HYLE_SORT_DECL_AVX512)
#undef HYLE_SORT_DECL_AVX512
#endif  // HYLE_ARCH_X64

#if defined(HYLE_ARCH_ARM64)
// aarch64 NEON HYLE vqsort port core (ARMv8 baseline; sort_dispatch enters it unconditionally,
// no runtime gate -- NEON is guaranteed present).
#define HYLE_SORT_DECL_NEON(T, tag) HYLE_SORT_DECL_TRIO(T, tag, neon)
HYLE_SORT_KEY_TYPES(HYLE_SORT_DECL_NEON)
#undef HYLE_SORT_DECL_NEON
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
#define HYLE_SORT_DECL_SVE(T, tag) HYLE_SORT_DECL_TRIO(T, tag, sve)
HYLE_SORT_KEY_TYPES(HYLE_SORT_DECL_SVE)
#undef HYLE_SORT_DECL_SVE
#endif
#endif  // HYLE_ARCH_ARM64

#if defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
#define HYLE_SORT_DECL_RVV(T, tag) HYLE_SORT_DECL_TRIO(T, tag, rvv)
HYLE_SORT_KEY_TYPES(HYLE_SORT_DECL_RVV)
#undef HYLE_SORT_DECL_RVV
#endif

#undef HYLE_SORT_DECL_TRIO

// Unsigned 128-bit numeric key. The public UInt128 and the port's uint128_t are byte-identical
// to this internal POD; keeping the public type out of internal headers avoids an API dependency
// in per-target kernels.
struct alignas(16) U128Bits { std::uint64_t lo; std::uint64_t hi; };
#define HYLE_SORT_DECL_U128_TRIO(T, tag, tier)                                               \
  void sort_##tag##_##tier(T* keys, std::size_t n, bool ascending) noexcept;                 \
  void psort_##tag##_##tier(T* keys, std::size_t n, std::size_t k, bool ascending) noexcept; \
  void select_##tag##_##tier(T* keys, std::size_t n, std::size_t k, bool ascending) noexcept;
#define HYLE_SORT_U128_TYPES(X, tier) X(U128Bits, u128, tier)

HYLE_SORT_U128_TYPES(HYLE_SORT_DECL_U128_TRIO, scalar)
#if defined(HYLE_ARCH_X64)
HYLE_SORT_U128_TYPES(HYLE_SORT_DECL_U128_TRIO, avx2)
HYLE_SORT_U128_TYPES(HYLE_SORT_DECL_U128_TRIO, avx512)
#endif
#if defined(HYLE_ARCH_ARM64)
HYLE_SORT_U128_TYPES(HYLE_SORT_DECL_U128_TRIO, neon)
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
HYLE_SORT_U128_TYPES(HYLE_SORT_DECL_U128_TRIO, sve)
#endif
#endif
#if defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
HYLE_SORT_U128_TYPES(HYLE_SORT_DECL_U128_TRIO, rvv)
#endif
#undef HYLE_SORT_DECL_U128_TRIO

// --- key-value (top-k) symbols -----------------------------------------------------------
// KV sorts by key (value rides along). Public KeyValue32/64 are byte-identical to the port's
// K32V32/K64V64 (same alignas + field order), so dispatch reinterpret_casts public->KVBits and
// the port TU casts KVBits->K32V32/K64V64 zero-copy.
struct alignas(8) KV32Bits { std::uint32_t value; std::uint32_t key; };
struct alignas(16) KV64Bits { std::uint64_t value; std::uint64_t key; };
#define HYLE_SORT_DECL_KV_TRIO(KVT, kvtag, tier)                                              \
  void sort_##kvtag##_##tier(KVT* kv, std::size_t n, bool ascending) noexcept;                \
  void psort_##kvtag##_##tier(KVT* kv, std::size_t n, std::size_t k, bool ascending) noexcept;\
  void select_##kvtag##_##tier(KVT* kv, std::size_t n, std::size_t k, bool ascending) noexcept;
#define HYLE_SORT_KV_TYPES(X, tier) \
  X(KV32Bits, kv32, tier)          \
  X(KV64Bits, kv64, tier)

HYLE_SORT_KV_TYPES(HYLE_SORT_DECL_KV_TRIO, scalar)
#if defined(HYLE_ARCH_X64)
HYLE_SORT_KV_TYPES(HYLE_SORT_DECL_KV_TRIO, avx2)
HYLE_SORT_KV_TYPES(HYLE_SORT_DECL_KV_TRIO, avx512)
#endif
#if defined(HYLE_ARCH_ARM64)
HYLE_SORT_KV_TYPES(HYLE_SORT_DECL_KV_TRIO, neon)
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
HYLE_SORT_KV_TYPES(HYLE_SORT_DECL_KV_TRIO, sve)
#endif
#endif
#if defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
HYLE_SORT_KV_TYPES(HYLE_SORT_DECL_KV_TRIO, rvv)
#endif
#undef HYLE_SORT_DECL_KV_TRIO

// P1-3 streaming partial_sort (k ≪ n). Returns true if handled.
bool stream_partial_u32(std::uint32_t* keys, std::size_t n, std::size_t k, bool asc) noexcept;
bool stream_partial_i32(std::int32_t* keys, std::size_t n, std::size_t k, bool asc) noexcept;
bool stream_partial_u16(std::uint16_t* keys, std::size_t n, std::size_t k, bool asc) noexcept;
bool stream_partial_i16(std::int16_t* keys, std::size_t n, std::size_t k, bool asc) noexcept;
bool stream_partial_u64(std::uint64_t* keys, std::size_t n, std::size_t k, bool asc) noexcept;
bool stream_partial_i64(std::int64_t* keys, std::size_t n, std::size_t k, bool asc) noexcept;
bool stream_partial_f32(float* keys, std::size_t n, std::size_t k, bool asc) noexcept;
bool stream_partial_f64(double* keys, std::size_t n, std::size_t k, bool asc) noexcept;
bool stream_partial_kv32(KV32Bits* kv, std::size_t n, std::size_t k, bool asc) noexcept;
bool stream_partial_kv64(KV64Bits* kv, std::size_t n, std::size_t k, bool asc) noexcept;

}  // namespace hyle::simd::detail
