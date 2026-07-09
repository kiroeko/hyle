// vqsort_body.inc -- the SHARED body of the Highway vqsort borrow cores. Included by each ISA
// tier TU (vqsort_highway_avx2.cc / _avx512.cc / _neon.cc) AFTER it defines HYLE_SORT_TIER (the
// symbol suffix: avx2 / avx512 / neon) and includes the Highway headers. This is NOT a standalone
// TU (no include guard on purpose -- it is textually pasted once per tier with a different suffix,
// exactly like the math borrow TUs share the HYLE_HWY_MATH_* macro pattern).
//
// Each tier compiles at its own -march (via hyle_set_highway_isa / hyle_set_avx512_sort_isa in
// CMake), so hn:: / VQSortStatic lower to that tier's width. VQSORT_ONLY_STATIC=1 (set by the TU
// before including vqsort-inl.h) makes it header-only: no link against vqsort.cc, entropy comes
// from the stack/clock fallback (Fill16BytesStatic) instead of Fill16BytesSecure.
//
// The runtime `ascending` bool selects the compile-time order tag: both hwy::SortAscending{} and
// hwy::SortDescending{} instantiations are emitted, the branch picks one (zero dispatch cost --
// the tag is an empty type). Key types (u32/i32/u64/i64/f32/f64) ARE Highway lane types, so they
// pass straight through as the Key template arg.

#include "src/simd/sort/kernels.h"

#include <cstddef>
#include <cstdint>

namespace hn = hwy::HWY_NAMESPACE;

namespace hyle::simd::detail {

#define HYLE_SORT_PASTE2(a, b) a##b
#define HYLE_SORT_PASTE(a, b) HYLE_SORT_PASTE2(a, b)
// Builds e.g. sort_u32_avx2 -- HYLE_SORT_TIER is a macro so it needs the two-level paste to
// expand before concatenation.
#define HYLE_SORT_SYM(stem, tag) HYLE_SORT_PASTE(stem##_##tag##_, HYLE_SORT_TIER)

#define HYLE_SORT_DEFINE(T, tag)                                                              \
  void HYLE_SORT_SYM(sort, tag)(T* keys, std::size_t n, bool ascending) noexcept {            \
    if (ascending)                                                                            \
      hn::VQSortStatic(keys, n, hwy::SortAscending{});                                        \
    else                                                                                      \
      hn::VQSortStatic(keys, n, hwy::SortDescending{});                                       \
  }                                                                                           \
  void HYLE_SORT_SYM(psort, tag)(T* keys, std::size_t n, std::size_t k, bool ascending)       \
      noexcept {                                                                              \
    if (ascending)                                                                            \
      hn::VQPartialSortStatic(keys, n, k, hwy::SortAscending{});                              \
    else                                                                                      \
      hn::VQPartialSortStatic(keys, n, k, hwy::SortDescending{});                             \
  }                                                                                           \
  void HYLE_SORT_SYM(select, tag)(T* keys, std::size_t n, std::size_t k, bool ascending)      \
      noexcept {                                                                              \
    if (ascending)                                                                            \
      hn::VQSelectStatic(keys, n, k, hwy::SortAscending{});                                   \
    else                                                                                      \
      hn::VQSelectStatic(keys, n, k, hwy::SortDescending{});                                  \
  }

HYLE_SORT_KEY_TYPES(HYLE_SORT_DEFINE)

#undef HYLE_SORT_DEFINE
#undef HYLE_SORT_SYM
#undef HYLE_SORT_PASTE
#undef HYLE_SORT_PASTE2

}  // namespace hyle::simd::detail
