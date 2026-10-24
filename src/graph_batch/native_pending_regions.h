#pragma once

#include "hyle/graph/batch.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/region_provider.h"
#include "hyle/compute/detail/dependent_completion.h"
#include "hyle/mem/std_allocator.h"

#include <utility>
#include <vector>

namespace hyle::graph::batch::detail {
// The original typed CopyOutcomeSource is owned by the shared resource payload.
// Slots borrow it through one episode; it owns no executor or result slot.
struct NativePendingSourceVTable final {
  void (*destroy)(void *) noexcept = nullptr;
  Status (*reusable)(void *) noexcept = nullptr;
};
struct NativePendingSourceOwner final {
  const NativePendingSourceVTable *table = nullptr;
  void *state = nullptr;
  const void *identity = nullptr;
  NativePendingSourceOwner() noexcept = default;
  NativePendingSourceOwner(const NativePendingSourceOwner &) = delete;
  NativePendingSourceOwner &operator=(const NativePendingSourceOwner &) = delete;
  NativePendingSourceOwner(NativePendingSourceOwner &&other) noexcept { *this = std::move(other); }
  NativePendingSourceOwner &operator=(NativePendingSourceOwner &&other) noexcept {
    if (this != &other) {
      reset(); table = std::exchange(other.table, nullptr);
      state = std::exchange(other.state, nullptr);
      identity = std::exchange(other.identity, nullptr);
    }
    return *this;
  }
  ~NativePendingSourceOwner() { reset(); }
  void reset() noexcept {
    if (state) table->destroy(state);
    state = nullptr; table = nullptr; identity = nullptr;
  }
};
struct NativePendingSourceFactory final {
  const NativePendingSourceVTable *table = nullptr;
  const void *identity = nullptr;
  Result<void *> (*create)(void *prepared, std::size_t capacity,
      mem::AllocatorRef, compute::ResourceBudget) noexcept = nullptr;
};
// Exact selected physical readers, independently of attention/codec math or
// layout schema. Only external projected banks are included, never a query.
inline constexpr compute::ProviderExtensionId kNativePendingRegionExtensionId{
    0x48594c454e415450ULL, 0x454e44494e475247ULL, 1};
struct NativePendingRegionFacts final {
  std::span<const ValueId> inputs;
  std::span<const std::uint32_t> input_ports;
  NativePendingSourceFactory source;
};
struct NativePendingRegionVTable final {
  Result<NativePendingRegionFacts> (*inspect)(void *prepared, void *invocation) noexcept = nullptr;
  Status (*validate_frame)(void *prepared, void *invocation, ValueId,
      const compute::PagedArenaPendingRead &, const NativePendingSourceOwner &) noexcept = nullptr;
  // Infallible only after all actual readers and the Graph binding accepted.
  void (*commit_frame)(void *prepared, void *invocation, ValueId,
      compute::PagedArenaPendingRead, const NativePendingSourceOwner &) noexcept = nullptr;
  // Original Graph/source gates precede this infallible borrowed-frame release.
  void (*reset)(void *prepared, void *invocation) noexcept = nullptr;
};
struct NativePendingRegionRef final {
  const NativePendingRegionVTable *table = nullptr;
  const ProviderRegionPreparedVTable *implementation = nullptr;
  void *prepared = nullptr;
  void *invocation = nullptr;
  std::uint32_t step = 0;
  ExecutionLaneId lane;
  std::span<const ValueId> inputs;
  std::span<const std::uint32_t> input_ports;
  NativePendingSourceFactory source;
};
using NativePendingRegions = std::vector<NativePendingRegionRef,
    mem::StdAllocator<NativePendingRegionRef>>;
struct PendingPageInput final {
  ValueId value;
  compute::PagedArenaPendingRead frame;
  std::vector<ExecutionLaneId, mem::StdAllocator<ExecutionLaneId>> lanes;
  bool native = false;
};
struct NativePendingAccess final {
  void *source = nullptr;
  std::span<const PendingPageInput> inputs;
};
Status collect_native_pending_regions(Invocation &, NativePendingRegions &) noexcept;
void reset_native_pending_regions(std::span<const NativePendingRegionRef>) noexcept;
Result<bool> native_pending_frame(std::span<const NativePendingRegionRef>, ValueId,
    const compute::PagedArenaPendingRead &, const NativePendingSourceOwner &,
    std::vector<graph::detail::ProjectedReader, mem::StdAllocator<graph::detail::ProjectedReader>> &,
    std::vector<ExecutionLaneId, mem::StdAllocator<ExecutionLaneId>> &) noexcept;
void commit_native_pending_frame(std::span<const NativePendingRegionRef>, ValueId,
    compute::PagedArenaPendingRead, const NativePendingSourceOwner &) noexcept;

// The concrete source module validates/takes/restores its original Update type.
HYLE_GRAPH_BATCH_API Result<NativePendingAccess> native_pending_access(
    Invocation &, const NativePendingSourceVTable *) noexcept;
// A returned Completion consumes sources; rejection leaves that reference with
// the typed caller for source restoration and release.
HYLE_GRAPH_BATCH_API Result<compute::Completion> submit_native_pending(
    Invocation &, std::span<const QueueBinding>, const NativePendingSourceVTable *,
    compute::detail::OutcomeSources) noexcept;
HYLE_GRAPH_BATCH_API Result<void *> native_pending_recovery(
    Invocation &, const NativePendingSourceVTable *) noexcept;
HYLE_GRAPH_BATCH_API void finish_native_pending_recovery(Invocation &) noexcept;
} // namespace hyle::graph::batch::detail
