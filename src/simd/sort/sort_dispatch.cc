#include "hyle/simd/sort.h"

#include <cstddef>

#include "hyle/core/compiler.h"
#include "src/simd/dispatch_internal.h"
#include "src/simd/runtime_internal.h"
#include "src/simd/sort/kernels.h"

// Baseline whole-buffer dispatcher. The private Runtime table is the sole
// ordinary authority for both function pointers and selected-target metadata;
// exact builds bind their sole generated suffix at compile time.
namespace hyle::simd {

#define HYLE_ORDERING_SYMBOL_I(name, suffix) name##_##suffix
#define HYLE_ORDERING_SYMBOL(name, suffix) HYLE_ORDERING_SYMBOL_I(name, suffix)

#if defined(HYLE_SIMD_EXACT_TARGET_SCALAR) || defined(HYLE_SIMD_EXACT_TARGET_SSE4)
#define HYLE_ORDERING_EXACT_SYMBOL(name) detail::HYLE_ORDERING_SYMBOL(name, scalar)
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX512) && defined(HYLE_COMPILER_MSVC)
#define HYLE_ORDERING_EXACT_SYMBOL(name) detail::HYLE_ORDERING_SYMBOL(name, avx2)
#elif defined(HYLE_SIMD_EXACT)
#define HYLE_ORDERING_EXACT_SYMBOL(name) HYLE_SIMD_EXACT_SYMBOL(name)
#endif

#if defined(HYLE_SIMD_EXACT)
#define HYLE_SORT_DO_SORT(tag, ...) HYLE_ORDERING_EXACT_SYMBOL(sort_##tag)(__VA_ARGS__)
#define HYLE_SORT_DO_PARTIAL(tag, ...) HYLE_ORDERING_EXACT_SYMBOL(psort_##tag)(__VA_ARGS__)
#define HYLE_SORT_DO_SELECT(tag, ...) HYLE_ORDERING_EXACT_SYMBOL(select_##tag)(__VA_ARGS__)
#else
#define HYLE_SORT_DO_SORT(tag, ...) runtime().sort_ops.sort_##tag(__VA_ARGS__)
#define HYLE_SORT_DO_PARTIAL(tag, ...) runtime().sort_ops.partial_sort_##tag(__VA_ARGS__)
#define HYLE_SORT_DO_SELECT(tag, ...) runtime().sort_ops.select_##tag(__VA_ARGS__)
#endif

#define HYLE_SORT_PUBLIC(T, tag)                                                \
  void sort(T* keys, std::size_t n, SortOrder order) noexcept {                 \
    if (n == 0) return;                                                         \
    HYLE_FATAL_ASSERT(keys != nullptr);                                         \
    HYLE_SORT_DO_SORT(tag, keys, n, order == SortOrder::kAscending);           \
  }                                                                             \
  void partial_sort(T* keys, std::size_t n, std::size_t k,                     \
                    SortOrder order) noexcept {                                 \
    HYLE_FATAL_ASSERT(k <= n);                                                  \
    if (n == 0) return;                                                         \
    HYLE_FATAL_ASSERT(keys != nullptr);                                         \
    if (k == 0) return;                                                         \
    HYLE_SORT_DO_PARTIAL(tag, keys, n, k, order == SortOrder::kAscending);     \
  }                                                                             \
  void select(T* keys, std::size_t n, std::size_t k,                           \
              SortOrder order) noexcept {                                       \
    if (n == 0) return;                                                         \
    HYLE_FATAL_ASSERT(keys != nullptr);                                         \
    HYLE_FATAL_ASSERT(k < n);                                                   \
    HYLE_SORT_DO_SELECT(tag, keys, n, k, order == SortOrder::kAscending);      \
  }

HYLE_SORT_TYPES(HYLE_SORT_PUBLIC)

#undef HYLE_SORT_PUBLIC
#undef HYLE_SORT_DO_SELECT
#undef HYLE_SORT_DO_PARTIAL
#undef HYLE_SORT_DO_SORT
#if defined(HYLE_ORDERING_EXACT_SYMBOL)
#undef HYLE_ORDERING_EXACT_SYMBOL
#endif
#undef HYLE_ORDERING_SYMBOL
#undef HYLE_ORDERING_SYMBOL_I

}  // namespace hyle::simd
