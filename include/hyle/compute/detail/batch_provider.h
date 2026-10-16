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
struct BatchLayoutAccess final {
  HYLE_COMPUTE_BATCH_API static std::array<BatchMetadataSegment,5> segments(const BatchLayoutVersion&) noexcept;
};
inline constexpr ProviderExtensionId kBatchPlanningExtensionId{0x48594c4542415443ULL,0x504c414e4e494e47ULL,1};
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
  Result<bool> (*update)(void *,const BatchLayoutVersion &,BatchPlanReport *) noexcept=nullptr;
  EnqueueResult (*plan)(void *,Queue &,const BatchLayoutVersion &,BatchWorkspaceViews,BatchPlanReport *) noexcept=nullptr;
  EnqueueResult (*run)(void *,Queue &,std::span<const ConstTensorView>,BatchWorkspaceViews) noexcept=nullptr;
};
} // namespace hyle::compute::detail
