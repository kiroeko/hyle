#include "ordering_kernels.h"

#define VQSORT_ONLY_STATIC 1
#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"
#undef VQSORT_ONLY_STATIC

#include "hyle/core/compiler.h"

namespace hyle_bench::ordering {
namespace hn = hwy::HWY_NAMESPACE;

HYLE_NOINLINE void highway_sort_f32(float* values,
                                    std::size_t count) noexcept {
  hn::VQSortStatic(values, count, hwy::SortAscending{});
}

HYLE_NOINLINE void highway_partial_sort_f32(float* values,
                                            std::size_t count,
                                            std::size_t k) noexcept {
  hn::VQPartialSortStatic(values, count, k, hwy::SortAscending{});
}

}  // namespace hyle_bench::ordering
