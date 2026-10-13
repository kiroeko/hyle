#pragma once
#include "hyle/compute/vulkan_batch.h"
namespace hyle::bench::resident {
compute::vulkan::Program ragged_reference_program(
    const compute::vulkan::Context&,const compute::BatchLayoutVersion&,
    const compute::RaggedAttention&);
}
