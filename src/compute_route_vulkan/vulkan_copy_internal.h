#pragma once

#include "vulkan_memory_internal.h"

#include "hyle/compute/detail/copy_backend.h"

namespace hyle::compute::vulkan::detail {
struct CopyRecording;

// One cold native phase. The public copy controller supplies complete range,
// producer and F1 permission checks separately. No new queue/runtime exists.
HYLE_COMPUTE_ROUTE_VULKAN_API Result<std::shared_ptr<CopyRecording>>
prepare_copy_recording(const Context &context, std::uint32_t capacity,
                       mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
HYLE_COMPUTE_ROUTE_VULKAN_API Status bind_copy_recording(CopyRecording &recording,
    std::span<const compute::detail::CopyBackendRange> ranges,
    bool host_upload_ready = false) noexcept;
HYLE_COMPUTE_ROUTE_VULKAN_API Result<Completion> submit_copy_recording(CopyRecording &recording) noexcept;
HYLE_COMPUTE_ROUTE_VULKAN_API CopyBatchReport copy_recording_report(const CopyRecording &recording) noexcept;
} // namespace hyle::compute::vulkan::detail
