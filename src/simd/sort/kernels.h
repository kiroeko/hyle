#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/platform.h"  // HYLE_ARCH_X64 / HYLE_ARCH_ARM64 (borrow cores are arch-specific)

// kernels.h (internal) -- per-tier vqsort borrow symbols the sort dispatcher takes the address
// of. Same shape as math/kernels.h: the scalar TU (std::sort/partial_sort/nth_element) is ALWAYS
// defined + is the non-x86 / fallback floor (NOT a stub -- it is the correctness oracle). The
// Highway vqsort borrow cores are vendored header-only and compiled per ISA tier on the matching
// arch (x86 avx2/avx512, arm64 neon); sort_dispatch enters them only when the CPU supports the
// tier (cpu_features()), so the wide codegen never runs on an older CPU. Not a public header.
//
// The compile-time SortAscending/SortDescending order tag is chosen INSIDE each TU from the
// runtime `ascending` bool -- so one symbol per (type,op,tier) covers both directions (the tag
// is a zero-size type; the branch folds to two static instantiations, no dispatch cost).

namespace hyle::simd::detail {

// X-macro over the six supported key types this milestone: <C++ type>, <short tag>.
#define HYLE_SORT_KEY_TYPES(X)   \
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
// AVX2 (256-bit) + AVX-512 (512-bit) Highway vqsort borrow cores. Entered by sort_dispatch when
// cpu_features().avx2 / the avx512 group. Unlike Highway's math (contrib/math) which is AVX3-
// broken on MSVC, vqsort's AVX3 target is NOT broken there (it is plain integer/compare SIMD,
// not the FP transcendental codegen), so the 512 core exists on MSVC too -- no _MSC_VER exclusion.
#define HYLE_SORT_DECL_AVX2(T, tag) HYLE_SORT_DECL_TRIO(T, tag, avx2)
HYLE_SORT_KEY_TYPES(HYLE_SORT_DECL_AVX2)
#undef HYLE_SORT_DECL_AVX2
#define HYLE_SORT_DECL_AVX512(T, tag) HYLE_SORT_DECL_TRIO(T, tag, avx512)
HYLE_SORT_KEY_TYPES(HYLE_SORT_DECL_AVX512)
#undef HYLE_SORT_DECL_AVX512
#endif  // HYLE_ARCH_X64

#if defined(HYLE_ARCH_ARM64)
// aarch64 NEON Highway vqsort core (ARMv8 baseline; sort_dispatch enters it unconditionally,
// no runtime gate -- NEON is guaranteed present, like the math NEON borrow).
#define HYLE_SORT_DECL_NEON(T, tag) HYLE_SORT_DECL_TRIO(T, tag, neon)
HYLE_SORT_KEY_TYPES(HYLE_SORT_DECL_NEON)
#undef HYLE_SORT_DECL_NEON
#endif  // HYLE_ARCH_ARM64

#undef HYLE_SORT_DECL_TRIO

}  // namespace hyle::simd::detail
