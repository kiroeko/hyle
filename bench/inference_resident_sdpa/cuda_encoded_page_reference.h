#pragma once
#include "hyle/compute/batch_execution.h"
#include "hyle/compute/paged_arena.h"
#include "hyle/mem/buffer.h"
#include <array>

namespace hyle::bench::resident {
struct CudaEncodedReferenceData final {
  compute::BatchLayoutVersion layout;
  compute::PagedArenaRead keys,values;
  std::array<mem::Buffer,2> expanded;
  bool ready=false;
};
// The Context, memory resource and this data outlive every private provider slot.
compute::Provider cuda_encoded_reference_provider(const CudaEncodedReferenceData *) noexcept;
Status prepare_cuda_encoded_reference(CudaEncodedReferenceData &,compute::Queue &,
    compute::BatchLayoutVersion,compute::PagedArenaRead,compute::PagedArenaRead,
    mem::MemoryResourceRef) noexcept;
}
