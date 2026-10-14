#pragma once

#include "apple_runtime_internal.h"
#include "hyle/compute/detail/copy_backend.h"

namespace hyle::compute::apple::detail {
struct CopyRecording;
struct CopySpan final {
  void *buffer = nullptr;
  compute::detail::PhysicalCopyResource physical;
};
// Cold type/device/capacity proof. Buffer handles stay original opaque Metal
// objects; ordinary common Compute headers never acquire vendor declarations.
Result<CopySpan> copy_span(mem::ConstBufferView view) noexcept;
bool is_copy_resource(mem::MemoryResourceRef resource) noexcept;
// A readonly MLX export owns a distinct resource token for correct freeing.
// Only that private carrier may match a nominated native source family.
bool copy_source_resource_matches(mem::MemoryResourceRef nominated,
                                  mem::MemoryResourceRef actual) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Result<std::shared_ptr<CopyRecording>> prepare_copy_recording(
    const Context &, std::uint32_t capacity, mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
Result<std::shared_ptr<CopyRecording>> prepare_copy_recording_runtime(const std::shared_ptr<Runtime> &,
    std::uint32_t capacity, mem::AllocatorRef allocator, std::size_t max_bytes) noexcept;
// Borrowed Buffer/View/resource lifetimes are supplied by the public F2 lease
// controller; private/native callers must retain them through actual use.
HYLE_COMPUTE_ROUTE_APPLE_API Status bind_copy_recording(CopyRecording &,
    std::span<const compute::detail::CopyBackendRange>, bool prefix_marker = false) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Result<Completion> submit_copy_recording(CopyRecording &) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API CopyBatchReport copy_recording_report(const CopyRecording &) noexcept;
Status clear_copy_recording(CopyRecording &) noexcept;
// Permission/dependency predicate only, run on the original owner before blit.
// The supplied metadata must be retained through this Job's actual retirement.
using CopyPreflight = Status (*)(void *) noexcept;
Status set_copy_preflight(CopyRecording &, CopyPreflight, void *) noexcept;
Result<CompletionPoll> poll_copy_recording(const CopyRecording &) noexcept;
Status wait_copy_recording(const CopyRecording &) noexcept;
Status qualify_recording_dependency(const std::shared_ptr<Runtime> &, compute::detail::CompletionDependency) noexcept;
} // namespace hyle::compute::apple::detail
