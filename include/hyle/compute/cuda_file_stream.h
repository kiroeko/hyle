#pragma once

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/file_stream.h"
#include "hyle/visibility.h"

namespace hyle::compute::cuda {

#if defined(HYLE_COMPUTE_CUDA)
// Uses the original CUDA Queue/batch program. Staging must be the Context's
// original pinned-host resource; no CPU math provider is selected by streaming.
HYLE_COMPUTE_STREAM_CUDA_API Result<FileResourcePipeline> create_file_pipeline(
    io::IoContext &, const Context &, Queue &, mem::MemoryResourceRef staging,
    std::span<const CopyResourceSpec> targets, FileStreamOptions options = {}) noexcept;
#endif

} // namespace hyle::compute::cuda
