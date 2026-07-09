// Scalar sort floor -- std::sort / std::partial_sort / std::nth_element. ALWAYS compiled (every
// arch, the portable TU in the module base SOURCES); it is BOTH the correctness oracle the tests
// pin the vectorized cores against AND the runtime fallback when no SIMD tier is available (old
// x86 CPU without AVX2, or a non-x86/non-arm64 target). NOT a stub.
//
// Order: the runtime `ascending` bool picks std::less / std::greater as the comparator. This
// matches the vqsort semantics -- ascending = smallest first, descending = largest first. Like
// vqsort, std::sort / nth_element are NOT stable (equal keys' order unspecified) -- same contract.
//
// NaN NOTE (f32/f64): std comparators give a strict-weak-order only for non-NaN input; NaN breaks
// it (UB in std::sort). Highway's vqsort instead partitions NaNs to the back deterministically.
// The sort tests feed finite data (the common case + a fair oracle comparison); NaN-bearing input
// is a documented caller concern, out of scope for the scalar oracle.
#include "src/simd/sort/kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace hyle::simd::detail {

#define HYLE_SORT_SCALAR_DEFINE(T, tag)                                                     \
  void sort_##tag##_scalar(T* keys, std::size_t n, bool ascending) noexcept {               \
    if (ascending)                                                                          \
      std::sort(keys, keys + n, std::less<T>{});                                            \
    else                                                                                    \
      std::sort(keys, keys + n, std::greater<T>{});                                         \
  }                                                                                         \
  void psort_##tag##_scalar(T* keys, std::size_t n, std::size_t k, bool ascending)          \
      noexcept {                                                                            \
    if (ascending)                                                                          \
      std::partial_sort(keys, keys + k, keys + n, std::less<T>{});                          \
    else                                                                                    \
      std::partial_sort(keys, keys + k, keys + n, std::greater<T>{});                       \
  }                                                                                         \
  void select_##tag##_scalar(T* keys, std::size_t n, std::size_t k, bool ascending)         \
      noexcept {                                                                            \
    if (ascending)                                                                          \
      std::nth_element(keys, keys + k, keys + n, std::less<T>{});                           \
    else                                                                                    \
      std::nth_element(keys, keys + k, keys + n, std::greater<T>{});                        \
  }

HYLE_SORT_KEY_TYPES(HYLE_SORT_SCALAR_DEFINE)
#undef HYLE_SORT_SCALAR_DEFINE

}  // namespace hyle::simd::detail
