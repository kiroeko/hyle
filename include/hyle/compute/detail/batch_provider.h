#pragma once
#include <array>
#include <cstddef>
#include <span>
#include "hyle/compute/batch_execution.h"
#include "hyle/compute/detail/provider_extension.h"

namespace hyle::compute::detail {
// Active immutable payload at the schema's fixed device-wire offsets. Spans
// borrow the version; inactive bytes are zero in an explicit full export.
struct BatchMetadataSegment final {
  std::size_t offset=0;
  std::span<const std::byte> bytes;
};
// Only the limits used by immutable page/layout ownership. No math domain is
// synthesized when a row cache shares the original layout implementation.
struct BatchLayoutGeometry final {
  BatchStorage storage = BatchStorage::kPaged;
  std::uint32_t max_requests=0,max_query_rows=0,max_kv_rows=0;
  std::uint32_t max_pages=0,max_page_refs=0,page_size=0;
};
struct BatchLayoutAccess final {
  HYLE_COMPUTE_BATCH_API static bool is_row_cache(const BatchSchema&) noexcept;
  HYLE_COMPUTE_BATCH_API static BatchLayoutGeometry geometry(const BatchSchema&) noexcept;
  static const BatchSchema& owner(const RowCacheBatchSchema&s) noexcept{return s.owner_;}
  static const BatchLayoutVersion& owner(const RowCacheLayoutVersion&v) noexcept{return v.owner_;}
  static RowCacheBatchSchema row_schema(BatchSchema s) noexcept{return RowCacheBatchSchema(std::move(s));}
  static RowCacheLayoutVersion row_layout(BatchLayoutVersion v) noexcept{return RowCacheLayoutVersion(std::move(v));}
  HYLE_COMPUTE_BATCH_API static std::array<BatchMetadataSegment,5> segments(const BatchLayoutVersion&) noexcept;
  // Lazy sorted unique physical page IDs. The immutable span borrows the
  // version; it carries no page permissions and never changes after success.
  HYLE_COMPUTE_BATCH_API static Result<std::span<const std::int32_t>> page_footprint(const BatchLayoutVersion&) noexcept;
  static std::array<BatchMetadataSegment,5> row_segments(const RowCacheLayoutVersion&v) noexcept{return segments(v.owner_);}
  static Result<std::span<const std::int32_t>> row_page_footprint(const RowCacheLayoutVersion&v) noexcept{return page_footprint(v.owner_);}
};
inline constexpr ProviderExtensionId kBatchPlanningExtensionId{0x48594c4542415443ULL,0x504c414e4e494e47ULL,7};
// Same carrier ABI, distinct two-payload mathematical/physical contract.
inline constexpr ProviderExtensionId kMlaBatchPlanningExtensionId{0x48594c454d4c4142ULL,0x504c414e4e494e47ULL,5};
struct BatchWorkspaceViews final {
  mem::ConstBufferView metadata;
  mem::BufferView float_workspace,int_workspace,host_workspace,output;
};
// Opt-in named provider capability, not a replacement Queue/runtime. Native
// slot metadata is never an accepted GPU owner: actual F1 uses retain buffers,
// including pinned planning source, through the original Queue completion.
struct BatchProviderVTable final {
  Status (*qualify)(void *,BatchCapacity,const BatchProgramOptions &) noexcept=nullptr;
  Status (*create)(void *,BatchCapacity,const BatchProgramOptions &,mem::AllocatorRef,void **) noexcept=nullptr;
  void (*destroy)(void *,mem::AllocatorRef) noexcept=nullptr;
  // For a Graph-capture-compatible runtime, true preserves every recorded
  // native command resource and launch argument: only runtime metadata content
  // and host layout/generation bookkeeping may change. False or error must not
  // change/release the old command resources or launch arguments. Common
  // preparation calls its before-change gate before fresh planning. Providers
  // unable to promise this keep runtime.capture_compatible false.
  Result<bool> (*update)(void *,const BatchLayoutVersion &,BatchPlanReport *) noexcept=nullptr;
  EnqueueResult (*plan)(void *,QueueSequence &,const BatchLayoutVersion &,BatchWorkspaceViews,BatchPlanReport *) noexcept=nullptr;
  EnqueueResult (*run)(void *,QueueSequence &,std::span<const ConstTensorView>,BatchWorkspaceViews) noexcept=nullptr;
  // The prepared run callback can record into an existing capture sequence;
  // plan/upload/waits remain cold. Graph recording additionally requires the
  // actual shared/managed/nonblocking runtime's capture_compatible opt-in.
  bool supports_command_capture=false;
  // False means the native owner allocates its own scratch. Common output and
  // metadata storage, native budget checks and completion retention still apply.
  bool uses_caller_workspace=true;
  // Cold, host-only sizing for this exact layout. No command submission or
  // plan mutation. Required for caller-owned scratch; reports requested bytes,
  // not the program's hard caps. Host planning scratch uses int_workspace_bytes.
  // allow_update carries the caller's existing preparation intent. Only
  // when true may a read-only reuse proof size the retained original plan.
  // False must size a fresh native plan, even when the old plan could run.
  Status (*workspace)(void *,Queue &,const BatchLayoutVersion &,bool allow_update,BatchPlanReport *) noexcept=nullptr;
  // Actual prepared-slot proof, queried only at a cold binding boundary.
  // True means this plan will not dereference any cache payload. The caller
  // additionally checks its owned zero-query/zero-page layout and exact ports.
  // Absent proof never permits an empty page footprint.
  bool (*empty_cache_reads)(void *) noexcept=nullptr;
  // Cold facts from the actual provider context. Layout-controlled batch slots
  // are shared-Queue/managed; absence conservatively means host-blocking.
  ProviderRuntimeProperties (*runtime)(void *) noexcept=nullptr;
};
} // namespace hyle::compute::detail
