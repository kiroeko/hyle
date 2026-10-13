#pragma once

#include "hyle/compute/vulkan.h"

namespace hyle::bench::resident {
// Independent recipe using the pinned original full-F32 flash shader. The
// existing public Program carries admission, completion and output ownership.
compute::vulkan::Program reference_program(const compute::vulkan::Context &,
                                           const compute::TensorDesc &query,
                                           const compute::TensorDesc &cache,
                                           bool causal);
} // namespace hyle::bench::resident
