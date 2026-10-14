#pragma once

#include <cstdint>
#include "hyle/compute/vulkan.h"

namespace hyle::bench::resident {
// Independent recipe using the pinned original full-F32 flash shader. The
// existing public Program carries admission, completion and output ownership.
compute::vulkan::Program reference_program(const compute::vulkan::Context &,
                                           const compute::TensorDesc &query,
                                           const compute::TensorDesc &cache,
                                           bool causal);
// Benchmark-only independent dense input recipe. Q8 cache carriers are U8
// [batch, KV, heads, (logical_depth / 32) * 34]; no HYLE attention lowering.
compute::vulkan::Program encoded_reference_program(
    const compute::vulkan::Context &, const compute::TensorDesc &query,
    const compute::TensorDesc &key, const compute::TensorDesc &value,
    const compute::TensorDesc &output, bool causal, std::int32_t window_left,
    float scale);
} // namespace hyle::bench::resident
