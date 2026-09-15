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
    0x48594c4547524150ULL, 0x48524547494f4e31ULL, 16};

// Provider discovery helper. false means this otherwise valid node set would
// leave and re-enter through a data/control path; it should remain unfused.
// Empty, duplicate or foreign IDs are errors. Input order is unrestricted.
// Scratch uses the supplied allocator and is released before returning.
HYLE_GRAPH_API Result<bool> is_region_contractible(
    const Graph &graph, std::span<const NodeId> nodes,
    mem::AllocatorRef allocator = mem::system_allocator()) noexcept;

struct RegionProposal final {
  // Nonempty graph-local node set. try_add() copies and canonicalizes this
  // span; providers retain no ownership and may use stack storage. Graph
  // rejects a foreign identity, duplicate node, and any set that would form a
  // cycle when contracted to one execution island.
  std::span<const NodeId> nodes{};
  compute::ProviderRouteCostFacts cost{};
  // Provider-defined cold-path discriminator. It may select a fusion/compile
  // route but is never interpreted or searched on the steady path.
  std::uint64_t route = 0;
  // Available before placement/artifact creation. A known class must agree
  // with propose_resources() and the artifact. Unknown may be refined under
  // allow-any, but never qualifies for a prepared-only compile.
  compute::ProviderReplayClass replay_class =
      compute::ProviderReplayClass::kUnknown;
  // Stable implementation identity, not the temporary prepare token above.
  // Unknown remains an eligible route, but cannot identify a profile entry.
  compute::ProviderRouteKey profile_key{};
};

// Fixed-capacity proposal sink owned by Graph compile. Providers cannot grow
// it; discovery scratch uses the callback's allocator. A capacity refusal stops
// this provider's discovery;
// Graph reports limited search and still validates any resulting whole plan.
class RegionProposalWriter final {
public:
  RegionProposalWriter(std::span<RegionProposal> proposal_storage,
                       std::span<NodeId> node_storage) noexcept
      : proposal_storage_(proposal_storage), node_storage_(node_storage) {}

  // true: copied; false: capacity refused the proposal, stop discovery and
  // return Status::Ok() from the provider callback. An unexpected Status is a
  // real error and must be propagated. Never translate arbitrary provider OOM
  // into false. No proposal or node storage is committed on refusal/error.
  Result<bool> try_add(RegionProposal proposal) noexcept {
    if (proposal.nodes.empty()) {
      return std::unexpected(
          Status::InvalidArgument("graph region proposal node set is empty"));
    }
    for (NodeId node : proposal.nodes) {
      if (!node.valid()) {
        return std::unexpected(Status::InvalidArgument(
            "graph region proposal contains an invalid node identity"));
      }
    }
    if (size_ == proposal_storage_.size()) {
      proposal_limit_reached_ = true;
      return false;
    }
    if (proposal.nodes.size() > node_storage_.size() - node_size_) {
      node_limit_reached_ = true;
      return false;
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
        return std::unexpected(Status::InvalidArgument(
            "graph region proposal contains duplicate node ordinals"));
      }
    }
    proposal.nodes = destination;
    proposal_storage_[size_++] = proposal;
    node_size_ += destination.size();
    return true;
  }

  std::size_t size() const noexcept { return size_; }
  std::size_t capacity() const noexcept { return proposal_storage_.size(); }
  std::size_t node_size() const noexcept { return node_size_; }
  std::size_t node_capacity() const noexcept { return node_storage_.size(); }
  bool proposal_limit_reached() const noexcept {
    return proposal_limit_reached_;
  }
  bool node_limit_reached() const noexcept { return node_limit_reached_; }
  std::span<const RegionProposal> proposals() const noexcept {
    return proposal_storage_.first(size_);
  }

private:
  std::span<RegionProposal> proposal_storage_{};
  std::span<NodeId> node_storage_{};
  std::size_t size_ = 0;
  std::size_t node_size_ = 0;
  bool proposal_limit_reached_ = false;
  bool node_limit_reached_ = false;
};

struct RegionEffectAccess final {
  EffectResourceId resource{};
  compute::ResourceAccess access = compute::ResourceAccess::kRead;
};

struct RegionCompileRequest final {
  const Graph *graph = nullptr;
  std::span<const NodeId> nodes{};
  std::span<const ValueId> inputs{};
  std::span<const ValueId> outputs{};
  // Resource query inspects descriptors/policy only; it must not read payload
  // or cache source addresses. At compile, materialize entries refer to the
  // Graph-owned canonical copy, which can be borrowed through artifact death.
  std::span<const ConstantBinding> constants{};
  std::span<const mem::MemoryResourceRef> input_resources{};
  std::span<const mem::MemoryResourceRef> output_resources{};
  mem::MemoryResourceRef workspace_resource{};
  std::uint64_t route = 0;
  // Canonical union of node accesses. These are not extra execution inputs;
  // the implementation must preserve these effects and the graph's ordering.
  std::span<const RegionEffectAccess> effect_accesses{};
  std::span<const EffectResourceBinding> effect_bindings{};
  bool has_unknown_effects = false;
};

// Cold resource and runtime proposal for one exact candidate boundary. The
// provider fills one requirement per input/output. Graph chooses concrete
// storage domains, then passes their exact resources to compile().
struct RegionResourceProposal final {
  compute::ProviderRuntimeProperties runtime{};
  compute::detail::ResourceBindingRequirement workspace_resource{};
  // Required when the request contains named or unknown external effects.
  // false declines that semantic route before artifact creation.
  bool preserves_effects = false;
  compute::ProviderCompilationStorage compilation_storage{};
};

struct RegionArtifactProperties final {
  compute::WorkspaceRequirement workspace{};
  // nullopt means that opaque provider-owned code/metadata bytes are not
  // reportable. An engaged zero means that the artifact has none.
  std::optional<std::size_t> persistent_bytes{};
  // Packing is additional provider-owned storage. Counts never shorten a
  // borrow or discard a Graph-owned canonical materialization.
  std::size_t packed_constant_count = 0;
  std::size_t packed_constant_bytes = 0;
  std::size_t copy_bytes = 0;
  std::uint32_t synchronization_count = 0;
  // Exact backend execution episodes per submit; see ProviderRouteCostFacts.
  std::uint32_t dispatch_count = 1;
  compute::ProviderRuntimeProperties runtime{};
  bool preserves_effects = false;
  // Additional per-invocation storage upper bound, excluding the supplied
  // allocator and caller workspace. Includes create/replay/teardown. Unknown
  // is not zero; do not reuse persistent_bytes for this independent lifetime.
  std::optional<std::size_t> invocation_private_bytes{};
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
// Graph. All callbacks are cold-path and run under the owning Context's
// admission gate. Returned artifacts must use the supplied allocator and must
// not retain request spans or Graph references unless they create an explicit
// owner of their own. Discovery/resource queries may use only the caller's
// output storage and the supplied allocator for scratch, released before
// returning. They must not create private heap state/artifacts before Graph
// can admit the proposal's compilation storage bounds. Error Status diagnostic
// storage remains outside the compilation budget.
struct RegionProviderVTable final {
  Status (*propose)(void *provider_context, const Graph &graph,
                    mem::AllocatorRef allocator,
                    RegionProposalWriter &writer) noexcept = nullptr;
  // Queried for hard-eligible candidates before selection, even if compile()
  // never follows. Graph caches the result across selection retries; this
  // callback describes constraints, it does not prepare a heavyweight artifact.
  Status (*propose_resources)(
      void *provider_context, const RegionCompileRequest &request,
      mem::AllocatorRef allocator,
      std::span<compute::detail::ResourceBindingRequirement> input_requirements,
      std::span<compute::detail::ResourceBindingRequirement>
          output_requirements,
      RegionResourceProposal *proposal) noexcept = nullptr;
  // Declines explicitly promise transactional rejection before accepted work.
  // Any partial artifact must supply destroy_prepared, even on failure. Error
  // statuses are fatal regardless of their code; Graph never guesses rollback.
  compute::ProviderPrepareOutcome (*compile)(
      void *provider_context, const RegionCompileRequest &request,
      mem::AllocatorRef allocator,
      ProviderRegionPrepared *prepared) noexcept = nullptr;
};

} // namespace hyle::graph
