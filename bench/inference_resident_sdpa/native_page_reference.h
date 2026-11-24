#pragma once
#include "hyle/compute/batch_paged_native.h"
#include "hyle/compute/apple_batch.h"
namespace hyle::bench::resident {
Result<compute::apple::Program> page_reference_program(const compute::apple::Context&,
    const compute::NativePagedBatchProgram&,compute::BatchLayoutVersion,const compute::WorkloadContext&);
}
