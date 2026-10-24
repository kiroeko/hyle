#pragma once

#include "hyle/compute/vulkan.h"
#include "hyle/compute/file_stream.h"
#include "hyle/visibility.h"

#if defined(HYLE_COMPUTE_VULKAN)
namespace hyle::compute::vulkan {

// Original Runtime/transfer families and fences, without a common Queue.
HYLE_COMPUTE_STREAM_VULKAN_API Result<FileResourcePipeline> create_file_pipeline(
    io::IoContext &, const Context &, mem::MemoryResourceRef staging,
    std::span<const CopyResourceSpec> targets, FileStreamOptions options = {}) noexcept;

} // namespace hyle::compute::vulkan
#endif
