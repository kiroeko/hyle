#pragma once

#include "hyle/graph/batch.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/region_provider.h"
#include "native_pending_regions.h"
#include "native_append_regions.h"
#include "hyle/mem/std_allocator.h"

#include <utility>
#include <vector>

namespace hyle::graph::batch::detail {
// Cold layout capability of one exact selected prepared implementation. The
// payload ports describe Q/K/V (or Q/compact-cache). An explicit metadata
// input is bound once by the typed preparation; it is never a second executor.
// The physical input indices are the original region boundary port indices.
inline constexpr compute::ProviderExtensionId kNativeBatchRegionExtensionId{
    0x48594c454e415450ULL, 0x414745445245474eULL, 6};
struct NativeBatchRegionFacts final {
  NodeId node;
  // Nonnull only for the operation's immutable owned-layout recipe. Explicit
  // metadata operations leave this null throughout their slot lifetime. Paged
  // access is qualified by capacity.storage and the exact selected callbacks,
  // never by manufacturing or replacing this recipe pointer during prepare.
  const compute::NativePagedBatchProgram *paged_program = nullptr;
  compute::BatchCapacity capacity{};
  compute::MlaCacheLayout mla = compute::MlaCacheLayout::kNone;
  bool causal = true;
  std::int32_t window_left = -1;
  std::span<const ValueId> inputs;
  std::span<const std::uint32_t> input_ports;
  ValueId metadata_input;
  // Exact selected Q/output port extent R <= schema maximum. An owned recipe
  // fixes its active rows to R. Capacity+metadata operations admit active rows
  // <= R; their native implementation zeros the inactive result suffix.
  std::uint32_t output_query_rows = 0;
  // Explicit metadata operations have no owned initial layout. This is empty
  // until their first prepare, then names the actual owned slot version.
  compute::BatchLayoutVersion layout;
};
struct NativeBatchRegionVTable final {
  Result<NativeBatchRegionFacts> (*inspect)(void *prepared,
                                           void *invocation) noexcept = nullptr;
  Status (*prepare)(void *prepared, void *invocation,
                    compute::BatchLayoutVersion) noexcept = nullptr;
  Status (*validate_frame)(void *prepared, void *invocation, ValueId,
                           const compute::PagedArenaRead &) noexcept = nullptr;
  // Only called after every real reader and the original Graph binding have
  // accepted the exact frame. It cannot fail or manufacture ready permission.
  void (*commit_frame)(void *prepared, void *invocation, ValueId,
                        compute::PagedArenaRead) noexcept = nullptr;
  Status (*reset)(void *prepared, void *invocation) noexcept = nullptr;
  compute::NativePagedBatchReport (*report)(void *prepared,
                                           void *invocation) noexcept = nullptr;
  // Same-Graph ordered writer, not a ready frame or external copy update.
  // validate may import cold carriers after preparation was revoked. commit
  // cannot fail and runs only after core accepted the exact writer/reader proof.
  Status (*validate_written)(void *prepared, void *invocation, ValueId,
      compute::BatchLayoutVersion, compute::ConstTensorView,
      const NativeAppendRegionRef &) noexcept = nullptr;
  void (*commit_written)(void *prepared, void *invocation, ValueId,
      compute::BatchLayoutVersion, compute::ConstTensorView,
      const NativeAppendRegionRef &) noexcept = nullptr;
};
// Borrowed original slot references, retained only by its existing resource
// payload for cold reset. Destruction never dereferences these pointers.
struct NativeBatchRegionRef final {
  const NativeBatchRegionVTable *table = nullptr;
  const ProviderRegionPreparedVTable *implementation = nullptr;
  void *prepared = nullptr;
  void *invocation = nullptr;
  std::uint32_t step = 0;
  ExecutionLaneId lane;
  NodeId node;
  const compute::NativePagedBatchProgram *paged_program = nullptr;
  std::span<const ValueId> inputs;
  std::span<const std::uint32_t> input_ports;
  ValueId metadata_input;
  // Copies of immutable admission facts. A capacity operation's first prepare
  // may establish its layout, but cannot change its geometry or port contract.
  compute::BatchCapacity capacity{};
  compute::MlaCacheLayout mla = compute::MlaCacheLayout::kNone;
  bool causal = true;
  std::int32_t window_left = -1;
  std::uint32_t output_query_rows = 0;
};
using NativeBatchRegions = std::vector<NativeBatchRegionRef,
    mem::StdAllocator<NativeBatchRegionRef>>;
using NodeLayouts = std::vector<NodeLayout, mem::StdAllocator<NodeLayout>>;
using ProjectedReaders = std::vector<graph::detail::ProjectedReader,
    mem::StdAllocator<graph::detail::ProjectedReader>>;
Status collect_native_batch_regions(Invocation &, NativeBatchRegions &) noexcept;
Result<NativeBatchRegionFacts> inspect_native_batch_region(
    const NativeBatchRegionRef &) noexcept;
Status split_native_batch_layouts(std::span<const NativeBatchRegionRef>,
    std::span<const NodeLayout>, NodeLayouts &shared) noexcept;
Status prepare_native_batch_regions(std::span<const NativeBatchRegionRef>,
    std::span<const NodeLayout>) noexcept;
Status reset_native_batch_regions(std::span<const NativeBatchRegionRef>) noexcept;
Result<std::uint64_t> native_paged_read_capacity(
    std::span<const NativeBatchRegionRef>) noexcept;
Status native_paged_frame(std::span<const NativeBatchRegionRef>, ValueId,
    const compute::PagedArenaRead &, ProjectedReaders &) noexcept;
void commit_native_paged_frame(std::span<const NativeBatchRegionRef>, ValueId,
    compute::PagedArenaRead) noexcept;
Result<compute::NativePagedBatchReport> native_paged_report(
    Invocation &, NodeId) noexcept;
} // namespace hyle::graph::batch::detail
