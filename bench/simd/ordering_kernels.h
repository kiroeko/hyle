#pragma once

#include <cstddef>

namespace hyle_bench::ordering {

void highway_sort_f32(float* values, std::size_t count) noexcept;
void highway_partial_sort_f32(float* values, std::size_t count,
                              std::size_t k) noexcept;

}  // namespace hyle_bench::ordering
