#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/detail/resource_binding.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/provider.h"
#include "hyle/graph/executable.h"

namespace hyle::graph {

// Versioned, provider-local Graph region contract. Compute only sees this as a
// named extension table and therefore never depends on Graph types.
inline constexpr compute::ProviderExtensionId kRegionProviderExtensionId{
    0x48594c4547524150ULL, 0x48524547494f4e31ULL, 5};

// Provider discovery helper. false means this otherwise valid node set would
// leave and re-enter through a data/control path; it should remain unfused.
// Empty, duplicate or foreign IDs are errors. Input order is unrestricted.
// Scratch uses the supplied allocator and is released before returning.
HYLE_GRAPH_API Result<bool> is_region_contractible(
    const Graph &graph, std::span<const NodeId> nodes,
    mem::AllocatorRef allocator = mem::system_allocator()) noexcept;

// Cold-path facts for the exact static graph specialization. Net savings is
// relative to the provider's natural single-op fallback sequence when the
// provider has a portable estimate; nullopt means unknown and an engaged zero
// means exact no saving. Unknown estimates remain useful under the explicit
// provider-region policy and are never converted into benchmark-derived
// constants. The remaining fields expose hidden costs to policy/reporting.
struct RegionCostFacts final {
  std::optional<std::uint64_t> estimated_net_savings_ns{};
  std::optional<std::uint64_t> estimated_compile_ns{};
  std::optional<std::size_t> persistent_bytes{};
  std::optional<std::size_t> workspace_bytes{};
  std::size_t boundary_copy_bytes = 0;
  std::uint32_t synchronization_count = 0;
  // Number of backend execution episodes: one synchronous CPU executor
  // fan-out or one device command launch counts as one. Workers, tasks,
  // vector chunks and leaf calls do not count separately.
  std::uint32_t dispatch_count = 1;
  bool blocks_host = false;
  bool uses_private_threads = false;
};

struct RegionProposal final {
  // Nonempty graph-local node set. add() copies and canonicalizes this span;
  // providers retain no ownership and may use stack storage. Graph rejects a
  // foreign identity, duplicate node, and any set that would form a cycle
  // when contracted to one execution island.
  std::span<const NodeId> nodes{};
  RegionCostFacts cost{};
  // Provider-defined cold-path discriminator. It may select a fusion/compile
  // route but is never interpreted or searched on the steady path.
  std::uint64_t route = 0;
};

// Fixed-capacity proposal sink owned by Graph compile. Providers cannot grow
// it or hide allocation. Exhaustion is an explicit compile failure so a
// proposal budget never changes the chosen plan silently.
class RegionProposalWriter final {
public:
  RegionProposalWriter(std::span<RegionProposal> proposal_storage,
                       std::span<NodeId> node_storage) noexcept
      : proposal_storage_(proposal_storage), node_storage_(node_storage) {}

  Status add(RegionProposal proposal) noexcept {
    if (proposal.nodes.empty()) {
      return Status::InvalidArgument("graph region proposal node set is empty");
    }
    for (NodeId node : proposal.nodes) {
      if (!node.valid()) {
        return Status::InvalidArgument(
            "graph region proposal contains an invalid node identity");
      }
    }
    if (size_ == proposal_storage_.size()) {
      return Status::ResourceExhausted(
          "graph region proposal budget is exhausted");
    }
    if (proposal.nodes.size() > node_storage_.size() - node_size_) {
      return Status::ResourceExhausted(
          "graph region proposal node budget is exhausted");
    }
    std::span<NodeId> destination =
        node_storage_.subspan(node_size_, proposal.nodes.size());
    std::copy(proposal.nodes.begin(), proposal.nodes.end(),
              destination.begin());
    std::sort(destination.begin(), destination.end(),
              [](NodeId left, NodeId right) noexcept {
                return left.ordinal() < right.ordinal();
              });
    for (std::size_t index = 1; index < destination.size(); ++index) {
      if (destination[index - 1].ordinal() == destination[index].ordinal()) {
        return Status::InvalidArgument(
            "graph region proposal contains duplicate node ordinals");
      }
    }
    proposal.nodes = destination;
    proposal_storage_[size_++] = proposal;
    node_size_ += destination.size();
    return Status::Ok();
  }

  std::size_t size() const noexcept { return size_; }
  std::size_t capacity() const noexcept { return proposal_storage_.size(); }
  std::size_t node_size() const noexcept { return node_size_; }
  std::size_t node_capacity() const noexcept { return node_storage_.size(); }
  std::span<const RegionProposal> proposals() const noexcept {
    return proposal_storage_.first(size_);
  }

private:
  std::span<RegionProposal> proposal_storage_{};
  std::span<NodeId> node_storage_{};
  std::size_t size_ = 0;
  std::size_t node_size_ = 0;
};

struct RegionCompileRequest final {
  const Graph *graph = nullptr;
  std::span<const NodeId> nodes{};
  std::span<const ValueId> inputs{};
  std::span<const ValueId> outputs{};
  std::span<const ConstantBinding> constants{};
  std::span<const mem::MemoryResourceRef> input_resources{};
  std::span<const mem::MemoryResourceRef> output_resources{};
  mem::MemoryResourceRef workspace_resource{};
  std::uint64_t route = 0;
};

// Cold resource and runtime proposal for one exact candidate boundary. The
// provider fills one requirement per input/output. Graph chooses concrete
// storage domains, then passes their exact resources to compile().
struct RegionResourceProposal final {
  compute::ProviderRuntimeProperties runtime{};
  compute::detail::ResourceBindingRequirement workspace_resource{};
};

struct RegionArtifactProperties final {
  compute::WorkspaceRequirement workspace{};
  // nullopt means that opaque provider-owned code/metadata bytes are not
  // reportable. An engaged zero means that the artifact has none.
  std::optional<std::size_t> persistent_bytes{};
  // Packing is an additional provider-owned representation. Graph keeps
  // every source ConstantBinding borrowed because aggregate counts do not
  // identify which source could be released.
  std::size_t packed_constant_count = 0;
  std::size_t packed_constant_bytes = 0;
  std::size_t copy_bytes = 0;
  std::uint32_t synchronization_count = 0;
  // Exact backend execution episodes per submit; see RegionCostFacts.
  std::uint32_t dispatch_count = 1;
  compute::ProviderRuntimeProperties runtime{};
};

struct ProviderRegionPreparedVTable final {
  void (*destroy_prepared)(void *state,
                           mem::AllocatorRef allocator) noexcept = nullptr;
  compute::AliasSupport (*alias_support)(
      void *prepared, std::size_t output_index,
      std::size_t input_index) noexcept = nullptr;
  Status (*create_invocation)(void *prepared, mem::AllocatorRef allocator,
                              mem::BufferView workspace,
                              void **invocation) noexcept = nullptr;
  void (*destroy_invocation)(void *invocation,
                             mem::AllocatorRef allocator) noexcept = nullptr;
  // Exactly one submit route is present, selected by properties.runtime.mode.
  compute::detail::EnqueueResult (*shared_submit)(
      void *prepared, void *invocation, compute::Queue &queue,
      std::span<const compute::ConstTensorView> inputs,
      std::span<const compute::TensorView> outputs) noexcept = nullptr;
  Result<compute::Completion> (*delegated_submit)(
      void *prepared, void *invocation,
      std::span<const compute::detail::CompletionDependency> dependencies,
      std::span<const compute::ConstTensorView> inputs,
      std::span<const compute::TensorView> outputs) noexcept = nullptr;
  compute::detail::CompletionEndpoint (*endpoint)(
      void *prepared, void *invocation) noexcept = nullptr;
  bool (*can_import_endpoint)(
      void *prepared, void *invocation,
      compute::detail::CompletionEndpoint source) noexcept = nullptr;
};

struct ProviderRegionPrepared final {
  const ProviderRegionPreparedVTable *vtable = nullptr;
  void *state = nullptr;
  RegionArtifactProperties properties{};
};

// A provider proposes candidate node sets, then compiles only sets chosen by
// Graph. Both callbacks are cold-path and run under the owning Context's
// admission gate. Returned artifacts must use the supplied allocator and must
// not retain request spans or Graph references unless they create an explicit
// owner of their own.
struct RegionProviderVTable final {
  Status (*propose)(void *provider_context, const Graph &graph,
                    RegionProposalWriter &writer) noexcept = nullptr;
  Status (*propose_resources)(
      void *provider_context, const RegionCompileRequest &request,
      std::span<compute::detail::ResourceBindingRequirement>
          input_requirements,
      std::span<compute::detail::ResourceBindingRequirement>
          output_requirements,
      RegionResourceProposal *proposal) noexcept = nullptr;
  Status (*compile)(void *provider_context, const RegionCompileRequest &request,
                    mem::AllocatorRef allocator,
                    ProviderRegionPrepared *prepared) noexcept = nullptr;
};

} // namespace hyle::graph
