#pragma once
#include "hyle/graph/leased_invocation.h"
#include "hyle/compute/detail/provider_extension.h"
namespace hyle::graph::detail {
struct RegionAccessView final {
  std::uint32_t step=0;
  std::uint64_t graph_generation=0;
  std::span<const std::uint32_t> node_ordinals;
  const compute::detail::ProviderExtensionRef*owner=nullptr;
  void*prepared=nullptr;
  void*invocation=nullptr;
};
using RegionAccessFn=Status(*)(void*,const RegionAccessView&) noexcept;
// Explicit cold/idle opt-in only. Views borrow the callback scope, never a
// different execution slot. Ordinary compile/submit do not call this visitor.
struct RegionAccess final {
  HYLE_GRAPH_API static Status visit(const Executable&,RegionAccessFn,void*) noexcept;
  HYLE_GRAPH_API static Status visit(Invocation&,RegionAccessFn,void*) noexcept;
  HYLE_GRAPH_API static Status visit(LeasedInvocation&,RegionAccessFn,void*) noexcept;
  // Cold opt-in capacity; ordinary leased invocations retain their old cost.
  HYLE_GRAPH_API static Status reserve_reads(LeasedInvocation&,std::uint32_t,
      LeasedInvocationOptions) noexcept;
  // Trusted adapters supply the exact footprint and qualify every reader.
  // Rejects copies, ordinary operations, aliases and published capacity views.
  HYLE_GRAPH_API static Status bind_projected(LeasedInvocation&,ValueId,
      compute::ConstTensorView,std::span<const compute::ReadLease>,
      compute::ProviderId,std::span<const std::uint32_t> reader_steps) noexcept;
  // Same immutable Graph, uniquely qualified dense reshape -> checked Gather.
  HYLE_GRAPH_API static Status bind_gather(LeasedInvocation&,const Graph&,
      ValueId capacity,ValueId indices,compute::ConstTensorView,
      std::span<const compute::ReadLease>) noexcept;
  // Cold successful idle caller output only; empty after readonly publication.
  HYLE_GRAPH_API static Result<compute::WriteLease> take_write(
      LeasedInvocation&,ValueId caller_output) noexcept;
};
} // namespace hyle::graph::detail
