// Scalar sort floor -- std::sort / std::partial_sort / std::nth_element. ALWAYS compiled (every
// arch, the portable TU in the module base SOURCES); it is BOTH the correctness oracle the tests
// pin the vectorized cores against AND the runtime fallback when no SIMD tier is available (old
// x86 CPU without AVX2, or a non-x86/non-arm64 target). NOT a stub.
//
// Order: the runtime `ascending` bool picks std::less / std::greater as the comparator. This
// matches the vqsort semantics -- ascending = smallest first, descending = largest first. Like
// vqsort, std::sort / nth_element are NOT stable (equal keys' order unspecified) -- same contract.
//
// Floating order matches VQSort: finite values follow the requested direction and every NaN is
// ordered after every finite value for both directions. NaNs compare equivalent to each other.
// This keeps the scalar fallback a valid strict-weak-order instead of feeding std with plain < on NaN.
#include "src/simd/sort/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace hyle::simd::detail {

template <typename T, bool kAscending>
struct ScalarOrder {
  bool operator()(T a, T b) const noexcept {
    if constexpr (std::is_floating_point_v<T>) {
      const bool a_nan = std::isnan(a);
      const bool b_nan = std::isnan(b);
      if (a_nan || b_nan) return !a_nan && b_nan;
    }
    if constexpr (kAscending) return a < b;
    else return a > b;
  }
};

#define HYLE_SORT_SCALAR_DEFINE(T, tag)                                                     \
  void sort_##tag##_scalar(T* keys, std::size_t n, bool ascending) noexcept {               \
    if (ascending)                                                                          \
      std::sort(keys, keys + n, ScalarOrder<T, true>{});                                            \
    else                                                                                    \
      std::sort(keys, keys + n, ScalarOrder<T, false>{});                                         \
  }                                                                                         \
  void psort_##tag##_scalar(T* keys, std::size_t n, std::size_t k, bool ascending)          \
      noexcept {                                                                            \
    if (ascending)                                                                          \
      std::partial_sort(keys, keys + k, keys + n, ScalarOrder<T, true>{});                          \
    else                                                                                    \
      std::partial_sort(keys, keys + k, keys + n, ScalarOrder<T, false>{});                       \
  }                                                                                         \
  void select_##tag##_scalar(T* keys, std::size_t n, std::size_t k, bool ascending)         \
      noexcept {                                                                            \
    if (ascending)                                                                          \
      std::nth_element(keys, keys + k, keys + n, ScalarOrder<T, true>{});                           \
    else                                                                                    \
      std::nth_element(keys, keys + k, keys + n, ScalarOrder<T, false>{});                        \
  }

HYLE_SORT_KEY_TYPES(HYLE_SORT_SCALAR_DEFINE)
#undef HYLE_SORT_SCALAR_DEFINE

struct ScalarOrder128 {
  bool ascending;
  bool operator()(const U128Bits& a, const U128Bits& b) const noexcept {
    const bool less = a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
    const bool greater = a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
    return ascending ? less : greater;
  }
};

void sort_u128_scalar(U128Bits* keys, std::size_t n, bool ascending) noexcept {
  std::sort(keys, keys + n, ScalarOrder128{ascending});
}
void psort_u128_scalar(U128Bits* keys, std::size_t n, std::size_t k, bool ascending) noexcept {
  std::partial_sort(keys, keys + k, keys + n, ScalarOrder128{ascending});
}
void select_u128_scalar(U128Bits* keys, std::size_t n, std::size_t k, bool ascending) noexcept {
  std::nth_element(keys, keys + k, keys + n, ScalarOrder128{ascending});
}

// KV scalar floor: sort by .key (value rides along). Comparators compare only the key field.
#define HYLE_SORT_KV_SCALAR_DEFINE(KVT, kvtag, tier)                                          \
  void sort_##kvtag##_##tier(KVT* kv, std::size_t n, bool ascending) noexcept {               \
    auto lt = [](const KVT& a, const KVT& b) { return a.key < b.key; };                       \
    auto gt = [](const KVT& a, const KVT& b) { return a.key > b.key; };                       \
    if (ascending) std::sort(kv, kv + n, lt); else std::sort(kv, kv + n, gt);                 \
  }                                                                                           \
  void psort_##kvtag##_##tier(KVT* kv, std::size_t n, std::size_t k, bool ascending) noexcept { \
    auto lt = [](const KVT& a, const KVT& b) { return a.key < b.key; };                       \
    auto gt = [](const KVT& a, const KVT& b) { return a.key > b.key; };                       \
    if (ascending) std::partial_sort(kv, kv + k, kv + n, lt);                                 \
    else std::partial_sort(kv, kv + k, kv + n, gt);                                           \
  }                                                                                           \
  void select_##kvtag##_##tier(KVT* kv, std::size_t n, std::size_t k, bool ascending) noexcept { \
    auto lt = [](const KVT& a, const KVT& b) { return a.key < b.key; };                       \
    auto gt = [](const KVT& a, const KVT& b) { return a.key > b.key; };                       \
    if (ascending) std::nth_element(kv, kv + k, kv + n, lt);                                  \
    else std::nth_element(kv, kv + k, kv + n, gt);                                            \
  }
HYLE_SORT_KV_TYPES(HYLE_SORT_KV_SCALAR_DEFINE, scalar)
#undef HYLE_SORT_KV_SCALAR_DEFINE

}  // namespace hyle::simd::detail
