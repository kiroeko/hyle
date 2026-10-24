#pragma once

#include "hyle/compute/apple.h"
#include "hyle/compute/file_stream.h"
#include "hyle/visibility.h"

#if defined(HYLE_COMPUTE_APPLE)
namespace hyle::compute::apple {

// Original MLX/Metal stream and shared native resources; no common Queue.
HYLE_COMPUTE_STREAM_APPLE_API Result<FileResourcePipeline> create_file_pipeline(
    io::IoContext &, const Context &, mem::MemoryResourceRef staging,
    std::span<const CopyResourceSpec> targets, FileStreamOptions options = {}) noexcept;

} // namespace hyle::compute::apple
#endif
