#pragma once

#include "apple_runtime_internal.h"
#include "apple_carrier.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/compute/detail/copy_backend.h"

namespace hyle::compute::apple::detail {
struct CopyRecording;
enum class CopyRecordingMode : std::uint8_t { kOnce, kReplayable };
struct CopySpan final {
  void *buffer = nullptr;
  compute::detail::PhysicalCopyResource physical;
};
// Cold type/device/capacity/access proof. Buffer handles stay original opaque Metal
// objects; ordinary common Compute headers never acquire vendor declarations.
HYLE_COMPUTE_ROUTE_APPLE_API Result<CopySpan> copy_span(mem::ConstBufferView view, mem::HostAccess access) noexcept;
Result<CopySpan> result_copy_span(const Results&,std::size_t,const TensorDesc&)noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API bool is_copy_resource(mem::MemoryResourceRef resource) noexcept;
struct ConstantCopy final {
  TensorView destination;
  ConstTensorView source;
};
struct MemoryAccess final {
  // Cold copies into exclusively Graph-owned shared Metal allocations. Ready
  // readonly native imports are source carriers only, never writable targets.
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<bool> can_copy_constant(
      const std::shared_ptr<Runtime> &, const ConstantCopy &) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status copy_constants(
      const std::shared_ptr<Runtime> &, std::span<const ConstantCopy>,
      mem::AllocatorRef) noexcept;
  // Metadata view of actual caller-owned shared Metal bytes. This does not
  // evaluate a producer or acquire ownership/permission; the original Graph
  // binding/lease must outlive the native use and every borrowed alias.
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<mlx::core::array> borrow_tensor(ConstTensorView) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<mlx::core::array> owned_tensor(ReadLease, const TensorDesc &,
      LeaseImportOptions = {}) noexcept;
  // Only successfully ready native outputs enter this owner export.
  HYLE_COMPUTE_ROUTE_APPLE_API static Status match_readonly(const mlx::core::array &,
      const TensorDesc &) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<ReadLease> export_readonly(const mlx::core::array &,
      const TensorDesc &, LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<NativeStorage> storage_info(const mlx::core::array &) noexcept;
};
// A readonly MLX export owns a distinct resource token for correct freeing.
// Only that private carrier may match a nominated native source family.
HYLE_COMPUTE_ROUTE_APPLE_API bool copy_source_resource_matches(mem::MemoryResourceRef nominated,
                                  mem::MemoryResourceRef actual) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Result<std::shared_ptr<CopyRecording>> prepare_copy_recording(
    const Context &, std::uint32_t capacity, mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Result<std::shared_ptr<CopyRecording>>
prepare_copy_recording_runtime(const std::shared_ptr<Runtime> &,
    std::uint32_t capacity, mem::AllocatorRef allocator, std::size_t max_bytes,
    CopyRecordingMode = CopyRecordingMode::kOnce) noexcept;
// A single fixed CPU destination mapping is acquired cold. Deferred native
// sources retain their real producer and resolve only in this Runtime's FIFO
// Job after pinning a new submitted epoch. No readonly import is made writable.
HYLE_COMPUTE_ROUTE_APPLE_API Status bind_host_copy_recording(CopyRecording &,
    InputSource, const TensorDesc &, mem::BufferView destination) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Status bind_host_copy_recording(CopyRecording &,
    ConstTensorView source, mem::BufferView destination, ReadLease source_owner = {}) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Status copy_recording_reuse_gate(const CopyRecording &) noexcept;
// Cleanup only: actual owners must be retired, but failed terminal work may
// release imports without reopening its execution slot.
HYLE_COMPUTE_ROUTE_APPLE_API Status copy_recording_retired_gate(const CopyRecording &) noexcept;
// The recording retains the actual producer state, preventing recycled wrapper
// addresses from qualifying a different native output after a cold rebind.
HYLE_COMPUTE_ROUTE_APPLE_API bool copy_recording_matches_source(
    const CopyRecording &, const InputSource &) noexcept;
// Borrowed Buffer/View/resource lifetimes are supplied by the public F2 lease
// controller; private/native callers must retain them through actual use.
HYLE_COMPUTE_ROUTE_APPLE_API Status bind_copy_recording(CopyRecording &,
    std::span<const compute::detail::CopyBackendRange>, bool prefix_marker = false) noexcept;
// Deferred descriptors contain destinations and logical source references,
// not invented native handles. The original FIFO Job's preflight resolves and
// validates all real source coordinates before touching its existing commands.
using ResultCopySourceFn=Result<CopySpan>(*)(void*,std::size_t)noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Status bind_result_copy_recording(CopyRecording&,
    std::span<const compute::detail::CopyBackendRange>)noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Status resolve_result_copy_recording(CopyRecording&,
    std::span<const compute::detail::CopyBackendRange>,ResultCopySourceFn,void*)noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Result<Completion> submit_copy_recording(CopyRecording &) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API CopyBatchReport copy_recording_report(const CopyRecording &) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Status clear_copy_recording(CopyRecording &) noexcept;
// A native producer is pinned at the original accepted copy epoch and released
// by CopyRecording::finish, including abandonment. Resolve only on that FIFO.
HYLE_COMPUTE_ROUTE_APPLE_API Status bind_native_copy_source(CopyRecording &,
    InputSource, const TensorDesc &) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Result<CopySpan> native_copy_source(CopyRecording &) noexcept;
// Permission/dependency predicate only, run on the original owner before blit.
// The supplied metadata must be retained through this Job's actual retirement.
using CopyPreflight = Status (*)(void *) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Status set_copy_preflight(CopyRecording &, CopyPreflight, void *) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Result<CompletionPoll> poll_copy_recording(const CopyRecording &) noexcept;
Status wait_copy_recording(const CopyRecording &) noexcept;
Status qualify_recording_dependency(const std::shared_ptr<Runtime> &, compute::detail::CompletionDependency) noexcept;
} // namespace hyle::compute::apple::detail
