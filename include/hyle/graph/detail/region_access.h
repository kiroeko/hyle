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
};
} // namespace hyle::graph::detail
