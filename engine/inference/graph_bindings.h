#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "hyle/graph/executable.h"

namespace hyle::inference {

// Cold composition of the original Graph bindings and placement facts. Domain
// ordinals are positions in the caller's actual CompileEnvironment; they never
// imply a backend or device type. The domain span is borrowed through assembly
// and compile(), and options() borrows this object's own arrays for that call.
// Graph retains its original value, allocator, owner and lease validation.
class GraphBindings final {
 public:
  struct Location final {
    graph::ExecutionTargetId target{};
    graph::StorageDomainId storage{};
  };

  explicit GraphBindings(std::span<const graph::StorageDomain> domains) noexcept
      : domains_(domains) {}
  GraphBindings(const GraphBindings &) = delete;
  GraphBindings &operator=(const GraphBindings &) = delete;

  Result<graph::StorageDomainId> storage_for(mem::MemoryResourceRef resource) const noexcept {
    for (std::size_t index = 0; index < domains_.size(); ++index) {
      if (domains_[index].resource != resource) continue;
      if (index >= (std::numeric_limits<std::uint32_t>::max)())
        return std::unexpected(Status::OutOfRange("graph binding storage ordinal overflows"));
      return graph::StorageDomainId{static_cast<std::uint32_t>(index)};
    }
    return std::unexpected(Status::InvalidArgument("graph binding resource is outside configured storage domains"));
  }

  // Repeated agreement is harmless when several recipes use the same actual
  // external value. A conflicting domain is rejected before changing anything.
  Status external(graph::ValueId value, graph::StorageDomainId storage) noexcept {
    if (!storage.valid() || storage.ordinal() >= domains_.size())
      return Status::InvalidArgument("graph binding storage domain is not configured");
    for (const auto &entry : external_resources_) {
      if (entry.value != value) continue;
      if (entry.storage != storage)
        return Status::InvalidArgument("conflicting external tensor resource contracts");
      return Status::Ok();
    }
    try { external_resources_.push_back({value, storage}); }
    catch (...) { return Status::ResourceExhausted("graph external binding cold metadata"); }
    return Status::Ok();
  }

  // Preserve the exact ConstantBinding, including policy, original owner and
  // retained immutable lease. Keep an existing explicit domain contract. In
  // its absence, an exact configured resource supplies the resident domain;
  // otherwise Graph keeps its default and qualifies the original readonly
  // resource itself. Native readonly exports need not have the allocation
  // resource's token. Materialization's destination remains an explicit
  // external() contract or Graph's default, not its source allocation's domain.
  Status bind_constant(graph::ConstantBinding binding) noexcept {
    const auto previous_externals = external_resources_.size();
    if (binding.policy == graph::ConstantPolicy::kBorrow) {
      bool explicit_contract = false;
      for (const auto &entry : external_resources_)
        if (entry.value == binding.value) { explicit_contract = true; break; }
      if (!explicit_contract) {
        const auto storage = storage_for(binding.tensor.storage().resource());
        if (storage) { HYLE_RETURN_IF_ERROR(external(binding.value, *storage)); }
        else if (!storage.error().IsInvalidArgument()) return storage.error();
      }
    }
    try { constants_.push_back(std::move(binding)); }
    catch (...) {
      if (external_resources_.size() != previous_externals) external_resources_.pop_back();
      return Status::ResourceExhausted("graph constant binding cold metadata");
    }
    return Status::Ok();
  }

  // The caller chooses the location from its real topology. Graph interprets
  // this node's target and allocation/effect default, without selecting a queue
  // or provider here.
  Status node(graph::NodeId node, Location location) noexcept {
    try { node_placements_.push_back({node, location.target, graph::PlacementStrength::kRequired, location.storage}); }
    catch (...) { return Status::ResourceExhausted("graph node placement cold metadata"); }
    return Status::Ok();
  }

  std::span<const graph::ConstantBinding> constants() const noexcept { return constants_; }

  graph::CompileOptions options(graph::CompileOptions options = {}) const noexcept {
    options.constants = constants_;
    options.node_placements = node_placements_;
    options.external_resources = external_resources_;
    return options;
  }

 private:
  std::span<const graph::StorageDomain> domains_;
  std::vector<graph::ConstantBinding> constants_;
  std::vector<graph::NodePlacement> node_placements_;
  std::vector<graph::ExternalResourceContract> external_resources_;
};

} // namespace hyle::inference
