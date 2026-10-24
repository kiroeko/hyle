#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
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
    0x48594c4547524150ULL, 0x48524547494f4e31ULL, 30};

namespace detail {
struct ConstantArtifactScope;
}

// Provider-defined immutable representation, including every packing option
// that can change its contents. The provider instance/device, immutable
// boundary and root are supplied by Graph; source pointers are never identities.
struct ConstantPackingKey final {
  std::uint64_t format = 0;
  std::uint64_t configuration = 0;
  // Optional provider-defined exact identity, borrowed only through
  // get_or_create(). Graph copies these bytes before invoking a miss factory;
  // it never interprets them or uses their address as identity.
  std::span<const std::byte> identity{};
  constexpr bool valid() const noexcept { return format != 0; }
  friend constexpr bool operator==(ConstantPackingKey left,
                                   ConstantPackingKey right) noexcept {
    return left.format == right.format &&
           left.configuration == right.configuration &&
           std::ranges::equal(left.identity, right.identity);
  }
};

struct ConstantArtifactDeclaration final {
  ConstantPackingKey key{};
  // Must be a configured storage domain compatible with this provider target.
  mem::MemoryResourceRef storage{};
  std::size_t packed_bytes = 0;
  // Opaque code/metadata retained by this payload, excluding packed_bytes.
  // Like RegionArtifactProperties::persistent_bytes, unknown is not zero.
  // This classification is independent of which allocator owns the bytes.
  std::optional<std::size_t> persistent_bytes{};
  // Additional bounds outside the supplied allocator, including preparation,
  // failure and destruction. Excluded from the enclosing region's bounds.
  compute::ProviderCompilationStorage private_storage{};
  // Records preparation on the original source Context's admitted Queue.
  // Requires owned physical source, destination resource and metadata backing.
  bool queue_preparation = false;
};

struct ConstantArtifactPayload final {
  void *state = nullptr;
  void (*destroy)(void *, mem::AllocatorRef) noexcept = nullptr;
  // Optional temporary native program/workspace release, called exactly once
  // after actual retirement (or a synchronous/rejected factory). destroy must
  // also handle a partial payload whose preparation was never started.
  void (*retire_preparation)(void *) noexcept = nullptr;
};

struct ConstantArtifactView final {
  // Immutable, borrowed through destruction of the compiled region. The
  // payload must own all data needed for replay, not borrow caller scratch.
  const void *state = nullptr;
  std::size_t packed_bytes = 0;
};

struct ConstantArtifactPrepareRequest final {
  compute::ConstTensorView source;
  mem::MemoryResourceRef destination;
  mem::AllocatorRef allocator;
  // Nonnull only for queue_preparation. Record every accepted prefix, even
  // when returning an error. The callback never finishes or retains this guard.
  compute::detail::QueueSequence *sequence = nullptr;
};
using ConstantArtifactFactory = Status (*)(
    void *context, const ConstantArtifactPrepareRequest &request,
    ConstantArtifactPayload *result) noexcept;

// Only available inside region compile(), under its existing admission gate.
// Copies/spans cannot escape that callback. This is a family-local cold cache,
// not a global registry or a runtime lookup. A factory runs only on a miss,
// completes synchronously or records into request.sequence. Graph attaches the
// actual owners before observing its original Completion. An owned partial
// payload on error stays alive through actual retirement. No recursive access.
class ConstantArtifactAccess final {
public:
  ConstantArtifactAccess() noexcept = default;
  bool available() const noexcept { return scope_ != nullptr; }
  // Pure cold qualification, before any accepted work. False leaves the
  // original runtime operation legal; it never authorizes retry after failure.
  HYLE_GRAPH_API bool can_prepare_queued(
      ValueId constant, mem::MemoryResourceRef destination) const noexcept;
  // Only a resource_declined outcome permits resource-budget fallback; a
  // factory error, including OOM, is a real error. On failure result is empty.
  // constant selects the actual request.constants boundary, which may be an
  // immutable metadata View rather than a direct logical constant.
  HYLE_GRAPH_API compute::ProviderPrepareOutcome
  get_or_create(ValueId constant, ConstantArtifactDeclaration declaration,
                ConstantArtifactFactory factory, void *context,
                ConstantArtifactView *result) const noexcept;
  // Cold immutable region output, owned by an artifact acquired in this scope.
  // The binding must match an unpublished boundary output and its exact storage.
  // Registered outputs are omitted from the native submit's output span, in the
  // original port order. Logical node/data/control dependencies are unchanged.
  // Registration commits only if this compiled region is selected successfully.
  HYLE_GRAPH_API Status bind_prepared_output(
      ValueId output, ConstantArtifactView owner,
      compute::ConstTensorView binding) const noexcept;

private:
  friend struct detail::ConstantArtifactScope;
  explicit ConstantArtifactAccess(detail::ConstantArtifactScope *scope) noexcept
      : scope_(scope) {}
  detail::ConstantArtifactScope *scope_ = nullptr;
};

// Provider discovery helper. false means this otherwise valid node set would
// leave and re-enter through a data/control path; it should remain unfused.
// Empty, duplicate or foreign IDs are errors. Input order is unrestricted.
// Scratch uses the supplied allocator and is released before returning.
HYLE_GRAPH_API Result<bool> is_region_contractible(
    const Graph &graph, std::span<const NodeId> nodes,
    mem::AllocatorRef allocator = mem::system_allocator()) noexcept;

// Currently discovered, admitted leaf proposals for one execution target.
// provider belongs to this exact provider instance, not its family ID; best
// is the lowest declared rank among that node's discovered leaf proposals.
// These are policy ranks, not time estimates, prepare/budget guarantees or a
// claim about the fastest implementation at the eventual physical bindings.
// A missing proposal stays unknown, including region-only capabilities.
struct RegionLeafPreference final {
  std::optional<std::uint32_t> provider{};
  std::optional<std::uint32_t> best{};
};

// Borrowed only through discovery. An empty mask permits every node; otherwise
// its length equals graph.node_count(), indexed by NodeId::ordinal(), and zero
// excludes that node from this actual source/provider's proposed regions.
// Allowed membership is not proof of mathematical or resource support.
struct RegionDiscoveryRequest final {
  const Graph &graph;
  std::span<const std::uint8_t> allowed_nodes{};
  // Indexed by original node ordinal. Empty for native-only sources; otherwise
  // derived from the same cold leaf discovery, after policy/replay admission.
  // Neither this span nor its values may escape the discovery callback.
  std::span<const RegionLeafPreference> leaf_preferences{};
};

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
  compute::ProviderImplementationId implementation{};
  compute::RoutePolicyId policy{};
  // UINT32_MAX inherits the Context ProviderBinding preference. A
  // Provider-specific policy row may supply a more specific HYLE preference.
  std::uint32_t preference = (std::numeric_limits<std::uint32_t>::max)();
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

// Cold physical provenance supplied by Graph after metadata View placement.
// value is the actual region boundary; root is its immutable logical constant.
// byte_offset is relative to that root's tensor, never an allocation address.
struct RegionConstantOrigin final {
  ValueId value{};
  ValueId root{};
  std::size_t byte_offset = 0;
};

struct RegionCompileRequest final {
  const Graph *graph = nullptr;
  std::span<const NodeId> nodes{};
  std::span<const ValueId> inputs{};
  std::span<const ValueId> outputs{};
  // Resource query inspects descriptors/policy only; it must not read payload
  // or cache source addresses. At compile, entries cover this region's
  // constant inputs at their actual physical bindings, including immutable
  // metadata View boundaries. Their owner/lease come from the physical root;
  // tensor is the exact boundary view. Owned replicas can be borrowed through
  // artifact death.
  std::span<const ConstantBinding> constants{};
  std::span<const RegionConstantOrigin> constant_origins{};
  // Empty during resource discovery. Managed immutable packing is retained
  // independently of mutable region invocation/capture state.
  ConstantArtifactAccess constant_artifacts{};
  std::span<const mem::MemoryResourceRef> input_resources{};
  std::span<const mem::MemoryResourceRef> output_resources{};
  mem::MemoryResourceRef workspace_resource{};
  std::uint64_t route = 0;
  // Canonical union of node accesses. These are not extra execution inputs;
  // the implementation must preserve these effects and the graph's ordering.
  std::span<const RegionEffectAccess> effect_accesses{};
  std::span<const EffectResourceBinding> effect_bindings{};
  bool has_unknown_effects = false;
  // Identical during resource qualification and exact compile. A model-only
  // route declines here if its complete conditions do not match; discovery
  // remains a structural DAG operation. Never read these facts during replay.
  compute::WorkloadContext workload{};
  // Original shared scope for actual provider-owned allocations. Known
  // native backings must retain their reservations with the real backing,
  // including escaped results; opaque driver/private bytes remain separate.
  // These domains are borrowed only during this cold callback. An artifact
  // may retain selected resource refs under the ordinary resource lifetime
  // contract, never the span or an enclosing CompileEnvironment address.
  compute::ResourceBudget resource_budget{};
  std::span<const StorageDomain> available_storage_domains{};
};

enum class RegionOutputKind : std::uint8_t {
  kCaller,
  kStableGenerated, // Backing available at slot creation, content still unready.
  kBoundGenerated,  // Backing determined by cold input binding.
  kDeferredGenerated, // Actual handle appears after the producing episode.
};

// A guaranteed representation, independent of handle availability. Native-only
// ports may expose an additional real byte projection when an episode permits
// it, but consumers cannot assume that projection during cold route selection.
enum class RegionOutputProjection : std::uint8_t { kTensorView, kNativeOnly };
struct RegionOutputContract final {
  RegionOutputKind kind = RegionOutputKind::kCaller;
  RegionOutputProjection projection = RegionOutputProjection::kTensorView;
  constexpr bool valid() const noexcept {
    return kind <= RegionOutputKind::kDeferredGenerated &&
        projection <= RegionOutputProjection::kNativeOnly &&
        (kind != RegionOutputKind::kCaller || projection == RegionOutputProjection::kTensorView);
  }
  friend constexpr bool operator==(RegionOutputContract, RegionOutputContract) noexcept = default;
};
struct RegionOutputRequirement final {
  compute::detail::ResourceBindingRequirement resource{};
  RegionOutputContract value{};
};
struct RegionInputRequirement final {
  compute::detail::ResourceBindingRequirement resource{};
  // Accepts a real native value or original output reference from this same
  // source owner. The selected handler must qualify implementation, port,
  // description and runtime during cold binding; this flag grants no cast.
  bool accepts_reference = false;
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
  // Excludes private bounds requested through constant_artifacts.
  compute::ProviderCompilationStorage compilation_storage{};
};

struct RegionArtifactProperties final {
  compute::WorkspaceRequirement workspace{};
  // nullopt means that opaque provider-owned code/metadata bytes are not
  // reportable. An engaged zero means that the artifact has none.
  std::optional<std::size_t> persistent_bytes{};
  // Independent provider-owned packing, excluding constant_artifacts-managed
  // payloads (Graph counts their unique owners). Counts never shorten a borrow
  // or discard a Graph-owned canonical materialization. persistent_bytes above
  // must likewise exclude managed payload storage.
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

struct ProviderRegionPreparedVTable;
// Cold reference to an original selected output port, not a ready handle or
// Completion. The enclosing Invocation keeps both producer and consumer alive.
// Native adapters qualify the exact implementation/owner before storing it.
struct RegionOutputReference final {
  const ProviderRegionPreparedVTable *vtable = nullptr;
  void *prepared = nullptr;
  void *invocation = nullptr;
  std::size_t port = 0;
  bool valid() const noexcept { return vtable != nullptr; }
};

// Borrowed physical port of the selected artifact. Native handles belong to
// its original slot/result; these records neither create a tensor system nor
// grant ready access. A memory projection is present only when it is real.
struct RegionValue final {
  compute::TensorDesc description;
  compute::ConstTensorView tensor;
  const void *native_type = nullptr;
  const void *native_handle = nullptr;
  RegionOutputReference reference{};
  // Optional actual readonly permission, borrowed during cold binding. An
  // adapter may copy/subview it into its original native array owner. A raw
  // TensorView without this owner cannot grant independent backing ownership.
  const compute::ReadLease *lease = nullptr;
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
      void *prepared, void *invocation, compute::detail::QueueSequence &sequence,
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
  // Generated outputs do not receive a shadow Graph allocation. Ordinary
  // caller-buffer artifacts omit these hooks and retain their direct spans.
  RegionOutputContract (*output_contract)(void *prepared, std::size_t port) noexcept = nullptr;
  Result<RegionValue> (*output_value)(void *prepared, void *invocation,
                                    std::size_t port) noexcept = nullptr;
  Status (*prepare_bindings)(void *prepared, void *invocation,
      std::span<const RegionValue> inputs,
      // One entry per logical output; generated entries are empty.
      std::span<const compute::TensorView> caller_outputs) noexcept = nullptr;
  // Runs after internal references/bindings are installed, including repeated
  // cold preparation with unchanged ordinary TensorViews. It checks the
  // current slot's readiness and freezes only internal references, never
  // escaped user result owners. Failure revokes the Graph's prepared route.
  Status (*seal_bindings)(void *prepared, void *invocation) noexcept = nullptr;
  Result<compute::ReadLease> (*output_lease)(void *prepared, void *invocation,
                                          std::size_t port) noexcept = nullptr;
  // Optional cold capability of this exact selected artifact. The returned
  // table is borrowed through artifact destruction. It cannot change routing,
  // allocate a second execution slot or grant permission to unrelated ports.
  // Ordinary replay never queries this hook.
  const void *(*query_extension)(void *prepared,
      compute::ProviderExtensionId id) noexcept = nullptr;
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
  Status (*propose)(void *provider_context, const RegionDiscoveryRequest &request,
                    mem::AllocatorRef allocator,
                    RegionProposalWriter &writer) noexcept = nullptr;
  // Queried for hard-eligible candidates before selection, even if compile()
  // never follows. Graph caches the result across selection retries; this
  // callback describes constraints, it does not prepare a heavyweight artifact.
  Status (*propose_resources)(
      void *provider_context, const RegionCompileRequest &request,
      mem::AllocatorRef allocator,
      std::span<RegionInputRequirement> input_requirements,
      std::span<RegionOutputRequirement> output_requirements,
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
