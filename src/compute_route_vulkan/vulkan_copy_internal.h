#pragma once

#include "vulkan_memory_internal.h"

#include "hyle/compute/detail/copy_backend.h"

namespace hyle::compute::vulkan::detail {
struct CopyRecording;
enum class CopyRecordingMode : std::uint8_t { kOnce, kReplayable };
// The optional halfword path is an exact storage-buffer copy on the original
// compute queue. Ordinary transfer recordings retain their four-byte contract.
enum class CopyGranularity : std::uint8_t { kWord, kHalfword };
using ResultCopySourceFn=Result<CopySpan>(*)(void*,std::size_t)noexcept;
Result<CopySpan> result_copy_span(const std::shared_ptr<Runtime>&,const Results&,
    std::size_t,const TensorDesc&)noexcept;

// One cold native phase. The public copy controller supplies complete range,
// producer and F1 permission checks separately. No new queue/runtime exists.
HYLE_COMPUTE_ROUTE_VULKAN_API Result<std::shared_ptr<CopyRecording>>
prepare_copy_recording(const Context &context, std::uint32_t capacity,
                       mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
HYLE_COMPUTE_ROUTE_VULKAN_API Result<std::shared_ptr<CopyRecording>>
prepare_copy_recording_runtime(const std::shared_ptr<Runtime> &runtime,
    std::uint32_t capacity, mem::AllocatorRef allocator, std::size_t max_bytes = 64u * 1024u * 1024u,
    CopyRecordingMode mode = CopyRecordingMode::kOnce) noexcept;
HYLE_COMPUTE_ROUTE_VULKAN_API Result<std::shared_ptr<CopyRecording>>
prepare_copy_recording_runtime(const std::shared_ptr<Runtime> &runtime,
    std::uint32_t capacity, mem::AllocatorRef allocator, std::size_t max_bytes,
    CopyRecordingMode mode, CopyGranularity granularity) noexcept;
HYLE_COMPUTE_ROUTE_VULKAN_API Status bind_copy_recording(CopyRecording &recording,
    std::span<const compute::detail::CopyBackendRange> ranges,
    bool host_upload_ready = false, bool prefix_marker = false,
    ResultCopySourceFn source=nullptr,void*source_owner=nullptr) noexcept;
HYLE_COMPUTE_ROUTE_VULKAN_API Result<Completion> submit_copy_recording(CopyRecording &recording) noexcept;
HYLE_COMPUTE_ROUTE_VULKAN_API CopyBatchReport copy_recording_report(const CopyRecording &recording) noexcept;
// A retired fence alone is insufficient: the original Completion must also
// release its actual Run before a recorded phase can be reused.
HYLE_COMPUTE_ROUTE_VULKAN_API Status copy_recording_reuse_gate(const CopyRecording &recording) noexcept;
HYLE_COMPUTE_ROUTE_VULKAN_API Status clear_copy_recording(CopyRecording &recording) noexcept;
// Pure known-outcome query for an already submitted same-Graph writer. No poll
// or wait; pending work stays ordered on the original Runtime compute queue.
HYLE_COMPUTE_ROUTE_VULKAN_API Status copy_recording_write_status(const CopyRecording &) noexcept;
// Cleanup gate only: retired closed/failed recordings may release imports.
HYLE_COMPUTE_ROUTE_VULKAN_API Status copy_recording_retired_gate(const CopyRecording &) noexcept;
Status copy_recording_terminal_hint(const CopyRecording &recording) noexcept;
Result<CompletionPoll> poll_copy_recording(const CopyRecording &recording) noexcept;
} // namespace hyle::compute::vulkan::detail
