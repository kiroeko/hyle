#pragma once
#include "hyle/compute/apple_batch.h"
namespace hyle::bench::resident {
compute::apple::Program ragged_reference_program(
    const compute::apple::Context&,const compute::BatchLayoutVersion&,
    const compute::RaggedAttention&);
}
