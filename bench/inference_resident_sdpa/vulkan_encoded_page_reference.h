#pragma once

#include "hyle/compute/batch_paged_native.h"
#include "hyle/compute/vulkan.h"

namespace hyle::bench::resident {
// Benchmark-only original scalar Q8 pipeline, with charged physical-page gather.
Result<compute::vulkan::Program> encoded_page_reference_program(
    const compute::vulkan::Context &, const compute::NativePagedBatchProgram &,
    compute::BatchLayoutVersion, const compute::WorkloadContext &);
}
