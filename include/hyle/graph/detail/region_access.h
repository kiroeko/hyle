#pragma once
#include "hyle/graph/leased_invocation.h"
#include "hyle/compute/detail/provider_extension.h"
#include "hyle/compute/detail/queue_sequence.h"
namespace hyle::graph::detail {
struct RegionAccessView final {
  std::uint32_t step=0;
  std::uint64_t graph_generation=0;
  std::span<const std::uint32_t> node_ordinals;
  const compute::detail::ProviderExtensionRef*owner=nullptr;
  void*prepared=nullptr;
  void*invocation=nullptr;
  ExecutionLaneId lane{};
  // Only visit_bound supplies these original prepared port arrays. Borrowed
  // for this cold callback; no tensor execution or ready permission is implied.
  std::span<const compute::ConstTensorView> inputs{};
  std::span<const compute::TensorView> outputs{};
};
using RegionAccessFn=Status(*)(void*,const RegionAccessView&) noexcept;
struct OrderedGraphSubmission final {
  // Runs in each original shared-lane sequence, before any consumer command.
  // Record source ordering before admitting exact pending reads to uses.
  Status (*admit)(void*,ExecutionLaneId,compute::detail::QueueSequence&,
                  compute::LeaseUseBatch&) noexcept = nullptr;
  // Infallible after cold capacity/lifetime qualification. Maps the actual
  // accepted suffix in the same original public observation slot.
  compute::Completion (*map)(void*,compute::Completion&&) noexcept = nullptr;
  void* owner = nullptr;
  // Only these exact inputs defer their missing ready permission to admit.
  // Every other binding still goes through the ordinary lease checks.
  bool (*pending_input)(void*,ValueId) noexcept = nullptr;
};
// Explicit cold/idle opt-in only. Views borrow the callback scope, never a
// different execution slot. Ordinary compile/submit do not call this visitor.
struct RegionAccess final {
  // Cold adapter check against the retained target Context, independent of
  // tensor bindings. Replay preparation still owns the complete topology.
  HYLE_GRAPH_API static Status validate_queue(const Executable&,ExecutionLaneId,
      const compute::Queue&) noexcept;
  HYLE_GRAPH_API static Status visit(const Executable&,RegionAccessFn,void*) noexcept;
  HYLE_GRAPH_API static Status visit(Invocation&,RegionAccessFn,void*) noexcept;
  HYLE_GRAPH_API static Status visit(LeasedInvocation&,RegionAccessFn,void*) noexcept;
  // Requires complete current bindings and the same original idle/result gate.
  // Uses the invocation's existing preallocated port arrays, without copying.
  HYLE_GRAPH_API static Status visit_bound(Invocation&,RegionAccessFn,void*) noexcept;
  HYLE_GRAPH_API static Status visit_bound(LeasedInvocation&,RegionAccessFn,void*) noexcept;
  // Cold opt-in capacity; ordinary leased invocations retain their old cost.
  HYLE_GRAPH_API static Status reserve_reads(LeasedInvocation&,std::uint32_t,
      LeasedInvocationOptions) noexcept;
  // Trusted adapters supply the exact footprint and qualify every reader.
  // Rejects copies, ordinary operations, aliases and published capacity views.
  HYLE_GRAPH_API static Status bind_projected(LeasedInvocation&,ValueId,
      compute::ConstTensorView,std::span<const compute::ReadLease>,
      compute::ProviderId,std::span<const std::uint32_t> reader_steps) noexcept;
  // No ready permission is manufactured. The trusted adapter owns the typed
  // pending frame and supplies all exact guards in OrderedGraphSubmission.
  HYLE_GRAPH_API static Status bind_pending_projected(LeasedInvocation&,ValueId,
      compute::ConstTensorView,compute::ProviderId,
      std::span<const std::uint32_t> reader_steps) noexcept;
  HYLE_GRAPH_API static Result<compute::Completion> submit(LeasedInvocation&,
      std::span<const QueueBinding>,OrderedGraphSubmission) noexcept;
  // Observation/result gate only; failed or closed episodes may still return
  // their original source controls after actual retirement.
  HYLE_GRAPH_API static bool outcome_available(const LeasedInvocation&) noexcept;
  // Same immutable Graph, uniquely qualified dense reshape -> checked Gather.
  HYLE_GRAPH_API static Status bind_gather(LeasedInvocation&,const Graph&,
      ValueId capacity,ValueId indices,compute::ConstTensorView,
      std::span<const compute::ReadLease>) noexcept;
  // Cold successful idle caller output only; empty after readonly publication.
  HYLE_GRAPH_API static Result<compute::WriteLease> take_write(
      LeasedInvocation&,ValueId caller_output) noexcept;
};
} // namespace hyle::graph::detail
