#pragma once
#include "hyle/compute/batch_paged_native.h"
#if defined(HYLE_PAGE_APPLE_BENCH)
#include "hyle/compute/apple_batch.h"
namespace hyle::bench::resident {
Result<compute::apple::Program> page_reference_program(const compute::apple::Context&,
    const compute::NativePagedBatchProgram&,compute::BatchLayoutVersion,const compute::WorkloadContext&);
}
#else
#include "hyle/compute/vulkan_batch.h"
namespace hyle::bench::resident {
Result<compute::vulkan::Program> page_reference_program(const compute::vulkan::Context&,
    const compute::NativePagedBatchProgram&,compute::BatchLayoutVersion,const compute::WorkloadContext&);
}
#endif
