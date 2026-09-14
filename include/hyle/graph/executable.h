#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/tensor.h"
#include "hyle/error/result.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/visibility.h"
#include "hyle/mem/resource.h"

namespace hyle::graph {

namespace detail {
struct PlanState;
struct InvocationState;
struct ReportState;
struct SignatureState;
struct CompileReportAccess;
struct ExecutableAccess;
struct InvocationAccess;
struct SpecializationSignatureAccess;
} // namespace detail

class Invocation;

class ExecutionTargetId final {
public:
  constexpr ExecutionTargetId() noexcept = default;
  constexpr explicit ExecutionTargetId(std::uint32_t ordinal) noexcept
      : ordinal_(ordinal) {}

  constexpr bool valid() const noexcept { return ordinal_ != UINT32_MAX; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  constexpr std::uint32_t ordinal() const noexcept { return ordinal_; }

  friend constexpr bool operator==(ExecutionTargetId,
                                   ExecutionTargetId) noexcept = default;

private:
  std::uint32_t ordinal_ = UINT32_MAX;
};

class StorageDomainId final {
public:
  constexpr StorageDomainId() noexcept = default;
  constexpr explicit StorageDomainId(std::uint32_t ordinal) noexcept
      : ordinal_(ordinal) {}

  constexpr bool valid() const noexcept { return ordinal_ != UINT32_MAX; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  constexpr std::uint32_t ordinal() const noexcept { return ordinal_; }

  friend constexpr bool operator==(StorageDomainId,
                                   StorageDomainId) noexcept = default;

private:
  std::uint32_t ordinal_ = UINT32_MAX;
};

class ExecutionLaneId final {
public:
  constexpr ExecutionLaneId() noexcept = default;
  constexpr explicit ExecutionLaneId(std::uint32_t ordinal) noexcept
      : ordinal_(ordinal) {}

  constexpr bool valid() const noexcept { return ordinal_ != UINT32_MAX; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  constexpr std::uint32_t ordinal() const noexcept { return ordinal_; }

  friend constexpr bool operator==(ExecutionLaneId,
                                   ExecutionLaneId) noexcept = default;

private:
  std::uint32_t ordinal_ = UINT32_MAX;
};

// Borrowed cold compile inputs. Contexts are retained by the emitted plan;
// MemoryResource providers remain caller-owned and must outlive the
// Executable, its Invocations, and every submitted Completion.
struct StorageDomain final {
  mem::MemoryResourceRef resource{};
  std::size_t workspace_budget_bytes = SIZE_MAX;
};

struct ExecutionTarget final {
  const compute::Context *context = nullptr;
  StorageDomainId default_storage{};
};

enum class ExecutionLaneRole : std::uint8_t {
  kCompute = 0,
  kTransfer,
  kCommunication,
};

struct ExecutionLane final {
  ExecutionTargetId target{};
  ExecutionLaneRole role = ExecutionLaneRole::kCompute;
};

struct CompileEnvironment final {
  std::span<const StorageDomain> storage_domains{};
  std::span<const ExecutionTarget> targets{};
  // Empty means one implicit compute lane per target.
  std::span<const ExecutionLane> lanes{};
};

enum class PlacementStrength : std::uint8_t {
  kPreferred = 0,
  kRequired,
};

struct NodePlacement final {
  NodeId node{};
  ExecutionTargetId target{};
  PlacementStrength strength = PlacementStrength::kPreferred;
};

struct ValuePlacement final {
  ValueId value{};
  StorageDomainId storage{};
  PlacementStrength strength = PlacementStrength::kPreferred;
};

// Exact compile-time resource locality for a caller-bound graph value. The
// allocation itself may change between replays, but it must use the selected
// StorageDomain's exact resource. Omitted source contracts use storage zero;
// an omitted produced output stays in its producer's selected storage.
struct ExternalResourceContract final {
  ValueId value{};
  StorageDomainId storage{};
};

struct QueueBinding final {
  ExecutionLaneId lane{};
  compute::Queue *queue = nullptr;
};

enum class ReplayMode : std::uint8_t {
  kCommands = 0,
  kPreferCapture,
  kRequireCapture,
};

enum class HostWaitPolicy : std::uint8_t {
  // Permit prepare_replay() to select an explicit host wait when the exact
  // destination Queue or delegated island cannot consume the source
  // endpoint's asynchronous prefix.
  kAllow = 0,
  // Reject such a Queue topology during cold preparation. This is useful for
  // latency-sensitive deployments that require fully device-ordered replay.
  kReject,
};

struct ReplayOptions final {
  ReplayMode mode = ReplayMode::kCommands;
  HostWaitPolicy host_wait = HostWaitPolicy::kAllow;
};

enum class ReplayRoute : std::uint8_t {
  kUnprepared = 0,
  kCommands,
  kCaptured,
  kHybrid,
};

enum class ReplayDependencyRoute : std::uint8_t {
  // Both logical domains share the same physical ordered Queue.
  kOrderedQueue = 0,
  // The source backend completes every Queue prefix before returning it.
  kSynchronousSource,
  // The destination Queue or delegated provider consumes the source
  // completion without blocking the submitting host thread.
  kDeviceWait,
  // Replay explicitly waits on the host before submitting the destination
  // segment. This route is visible and can be rejected through ReplayOptions.
  kHostWait,
};

struct ReplayReport final {
  ReplayRoute route = ReplayRoute::kUnprepared;
  std::size_t immediate_transfer_count = 0;
  std::size_t queued_transfer_count = 0;
  std::size_t peer_transfer_count = 0;
  std::size_t ordered_dependency_count = 0;
  std::size_t synchronous_source_dependency_count = 0;
  std::size_t device_wait_dependency_count = 0;
  std::size_t host_wait_dependency_count = 0;
};

// Exact, self-contained identity of one Graph specialization. The hash is an
// acceleration only: equality always compares the graph identity and every
// canonical binding (including computed derived dimensions), so a collision
// can never select the wrong Executable.
class SpecializationSignature final {
public:
  SpecializationSignature() noexcept = default;
  HYLE_GRAPH_API
  SpecializationSignature(const SpecializationSignature &other) noexcept;
  HYLE_GRAPH_API SpecializationSignature &
  operator=(const SpecializationSignature &other) noexcept;
  HYLE_GRAPH_API
  SpecializationSignature(SpecializationSignature &&other) noexcept;
  HYLE_GRAPH_API SpecializationSignature &
  operator=(SpecializationSignature &&other) noexcept;
  HYLE_GRAPH_API ~SpecializationSignature();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_GRAPH_API std::size_t binding_count() const noexcept;
  HYLE_GRAPH_API Result<ShapeBinding>
  binding(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API std::size_t hash() const noexcept;
  HYLE_GRAPH_API bool
  equals(const SpecializationSignature &other) const noexcept;

  friend bool operator==(const SpecializationSignature &left,
                         const SpecializationSignature &right) noexcept {
    return left.equals(right);
  }

private:
  friend struct detail::SpecializationSignatureAccess;
  explicit SpecializationSignature(detail::SignatureState *state) noexcept
      : state_(state) {}
  detail::SignatureState *state_ = nullptr;
};

enum class CompileStepKind : std::uint8_t {
  kOperationFallback = 0,
  kProviderRegion,
  kCopy,
};

enum class CompileFallbackReason : std::uint8_t {
  kNone = 0,
  kRegionSelectionDisabled,
  kNoEligibleRegion,
  kRegionNotSelected,
};

// One immutable step in the exact emitted execution plan. Synchronization is
// represented by segment dependencies rather than synthetic barrier steps.
// Provider-private layouts stay inside a region; the public plan emits a Copy
// only for a real canonical tensor boundary. Optional facts distinguish an
// opaque provider's unknown value from an exact zero.
struct CompileStepReport final {
  CompileStepKind kind = CompileStepKind::kOperationFallback;
  CompileFallbackReason fallback_reason = CompileFallbackReason::kNone;
  compute::ProviderIdentity provider{};
  ExecutionTargetId target{};
  ExecutionLaneId lane{};
  StorageDomainId workspace_storage{};
  StorageDomainId source_storage{};
  StorageDomainId destination_storage{};
  // Smallest canonical topological ordinal in this step. node_count is the
  // set cardinality; nodes_contiguous says whether begin/count also describe
  // the exact set. Use CompileReport::step_node() for general regions.
  std::uint32_t node_begin = 0;
  std::uint32_t node_count = 0;
  bool nodes_contiguous = true;
  compute::WorkspaceRequirement workspace{};
  std::optional<std::size_t> persistent_bytes{};
  std::size_t borrowed_constant_count = 0;
  std::size_t packed_constant_count = 0;
  std::size_t packed_constant_bytes = 0;
  std::size_t copy_bytes = 0;
  std::uint32_t synchronization_count = 0;
  // Backend execution episodes per submit: one synchronous CPU executor
  // fan-out or one device command launch counts as one. Workers/tasks/chunks
  // do not count separately. The built-in View metadata step reports zero;
  // a custom operation can emit a command even when it materializes no value.
  std::uint32_t dispatch_count = 1;
  std::uint32_t input_begin = 0;
  std::uint32_t output_begin = 0;
  std::uint32_t input_count = 0;
  std::uint32_t output_count = 0;
  compute::ProviderRuntimeProperties runtime{};
};

struct CompileTargetReport final {
  ExecutionTargetId target{};
  compute::Device device{};
  StorageDomainId default_storage{};
};

struct CompileStorageReport final {
  StorageDomainId storage{};
  mem::MemoryProperties properties{};
  std::size_t workspace_budget_bytes = SIZE_MAX;
  std::size_t workspace_bytes = 0;
  std::size_t workspace_backing_count = 0;
};

struct CompileLaneReport final {
  ExecutionLaneId lane{};
  ExecutionTargetId target{};
  ExecutionLaneRole role = ExecutionLaneRole::kCompute;
  // Shared lanes require an exact caller Queue. Delegated lanes are emitted
  // by Graph for provider-owned runtime islands and never accept QueueBinding.
  compute::ProviderRuntimeMode runtime_mode =
      compute::ProviderRuntimeMode::kSharedQueue;
  // Conservative Queue Completion capacity needed by one in-flight
  // Invocation. Deployments sharing a Queue across N concurrent lanes can
  // reserve N times this value before entering their warm path.
  std::size_t completion_slots_per_invocation = 0;
  bool queue_required = false;
  bool asynchronous_queue = false;
  bool capture_supported = false;
};

// One physical plan value. A logical ValueId can appear more than once when
// placement materializes it in several domains. The report owns descriptions
// and identities, but intentionally does not retain caller resource objects.
struct CompilePhysicalValueReport final {
  ValueId value{};
  StorageDomainId storage{};
  compute::TensorDesc description{};
  std::optional<std::uint32_t> alias_source{};
  bool external_input = false;
  bool external_output = false;
  bool constant = false;
  bool invocation_owned_output = false;
  bool workspace_backed = false;
};

// An execution segment is the unit of finalization. Shared segments may use
// Queue command capture; delegated segments are provider-owned islands.
// Dependencies are queried through CompileReport::segment_dependency().
struct CompileSegmentReport final {
  ExecutionLaneId lane{};
  compute::ProviderRuntimeMode runtime_mode =
      compute::ProviderRuntimeMode::kSharedQueue;
  std::uint32_t step_begin = 0;
  std::uint32_t step_count = 0;
  std::uint32_t dependency_begin = 0;
  std::uint32_t dependency_count = 0;
  bool terminal = false;
  bool contains_commands = false;
  bool capture_compatible = false;
  bool capture_supported = false;
};

enum class CompileDependencyRoute : std::uint8_t {
  // Both segments belong to one exact ordered execution endpoint; no explicit
  // wait is emitted.
  kOrderedQueue = 0,
  // The source domain's Queue completes inline, so a cross-domain data/control
  // edge needs no runtime wait even when Queue objects differ.
  kSourceCompletesInline,
  // Exact Queue/provider endpoints decide the route. Reusing one endpoint
  // needs no explicit wait; otherwise prepare_replay() fixes either a
  // non-blocking imported wait or an explicit host wait before steady
  // execution begins.
  kQueueBindingDependent,
};

struct CompileSegmentDependencyReport final {
  std::uint32_t source_segment = 0;
  std::uint32_t destination_segment = 0;
  CompileDependencyRoute route = CompileDependencyRoute::kOrderedQueue;
};

// Copyable cold-path report that owns a compact immutable copy of the emitted
// plan facts independently of the Executable and its heavyweight artifacts.
// Aggregate queries and step(index) are O(1); step returns a value, so no
// diagnostic view dangles after either handle moves.
class CompileReport final {
public:
  CompileReport() noexcept = default;
  HYLE_GRAPH_API CompileReport(const CompileReport &other) noexcept;
  HYLE_GRAPH_API CompileReport &operator=(const CompileReport &other) noexcept;
  HYLE_GRAPH_API CompileReport(CompileReport &&other) noexcept;
  HYLE_GRAPH_API CompileReport &operator=(CompileReport &&other) noexcept;
  HYLE_GRAPH_API ~CompileReport();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_GRAPH_API compute::BackendId backend() const noexcept;
  HYLE_GRAPH_API compute::ProviderIdentity provider() const noexcept;
  HYLE_GRAPH_API std::size_t workspace_bytes() const noexcept;
  HYLE_GRAPH_API std::size_t workspace_backing_count() const noexcept;
  HYLE_GRAPH_API std::size_t node_count() const noexcept;
  HYLE_GRAPH_API std::size_t target_count() const noexcept;
  HYLE_GRAPH_API std::size_t storage_domain_count() const noexcept;
  HYLE_GRAPH_API std::size_t lane_count() const noexcept;
  HYLE_GRAPH_API std::size_t step_count() const noexcept;
  HYLE_GRAPH_API std::size_t segment_count() const noexcept;
  HYLE_GRAPH_API std::size_t segment_dependency_count() const noexcept;
  HYLE_GRAPH_API std::size_t command_segment_count() const noexcept;
  HYLE_GRAPH_API std::size_t delegated_segment_count() const noexcept;
  // Ordering edges added only under kAllowSerialization to satisfy an exact
  // storage budget. Zero means Graph preserved the pre-existing concurrency.
  HYLE_GRAPH_API std::size_t serialization_edge_count() const noexcept;
  HYLE_GRAPH_API std::size_t capturable_segment_count() const noexcept;
  HYLE_GRAPH_API std::size_t physical_value_count() const noexcept;
  HYLE_GRAPH_API std::size_t region_count() const noexcept;
  HYLE_GRAPH_API std::size_t fallback_operation_count() const noexcept;
  HYLE_GRAPH_API std::size_t constant_count() const noexcept;
  HYLE_GRAPH_API std::size_t borrowed_constant_count() const noexcept;
  HYLE_GRAPH_API std::size_t packed_constant_count() const noexcept;
  HYLE_GRAPH_API std::size_t packed_constant_bytes() const noexcept;
  HYLE_GRAPH_API std::size_t invocation_owned_output_count() const noexcept;
  HYLE_GRAPH_API std::size_t copy_bytes() const noexcept;
  // Provider-declared synchronization plus potential cross-Queue dependency
  // sites. Concrete Queue bindings can collapse a cross-domain site when both
  // domains share one ordered Queue.
  HYLE_GRAPH_API std::uint32_t synchronization_count() const noexcept;
  HYLE_GRAPH_API bool supports_concurrent_invocations() const noexcept;
  HYLE_GRAPH_API bool mixed_providers() const noexcept;
  HYLE_GRAPH_API bool mixed_backends() const noexcept;
  HYLE_GRAPH_API SpecializationSignature specialization() const noexcept;
  HYLE_GRAPH_API Result<CompileTargetReport>
  target(std::size_t index) const noexcept;
  HYLE_GRAPH_API Result<CompileStorageReport>
  storage_domain(std::size_t index) const noexcept;
  HYLE_GRAPH_API Result<CompileLaneReport>
  lane(std::size_t index) const noexcept;
  HYLE_GRAPH_API Result<CompileSegmentReport>
  segment(std::size_t index) const noexcept;
  HYLE_GRAPH_API Result<CompileSegmentDependencyReport>
  segment_dependency(std::size_t segment_index,
                     std::size_t dependency_index) const noexcept;
  HYLE_GRAPH_API Result<CompilePhysicalValueReport>
  physical_value(std::size_t index) const noexcept;
  HYLE_GRAPH_API Result<std::uint32_t>
  step_input(std::size_t step_index, std::size_t input_index) const noexcept;
  HYLE_GRAPH_API Result<std::uint32_t>
  step_output(std::size_t step_index, std::size_t output_index) const noexcept;
  HYLE_GRAPH_API Result<NodeId>
  step_node(std::size_t step_index, std::size_t node_index) const noexcept;
  HYLE_GRAPH_API Result<CompileStepReport>
  step(std::size_t index) const noexcept;

private:
  friend struct detail::CompileReportAccess;
  explicit CompileReport(detail::ReportState *state) noexcept : state_(state) {}
  detail::ReportState *state_ = nullptr;
};

// A constant binding is conservatively borrowed by the resulting Executable.
// A provider may additionally own a packed representation reported by
// CompileReport, but Graph does not use that aggregate fact to shorten the source
// lifetime. The source storage must outlive the Executable and every
// completion submitted from it.
struct ConstantBinding final {
  ValueId value{};
  compute::ConstTensorView tensor{};
};

enum class RegionSelectionPolicy : std::uint8_t {
  kDisabled = 0,
  // Region providers are explicitly installed in Context. Prefer a
  // deterministic high-coverage set, then fewer backend execution episodes
  // only if every compared candidate declares that count, then earlier
  // provider order. Intervals use weighted interval selection;
  // general node sets use bounded greedy selection with contraction checks,
  // not a global maximum-coverage guarantee or a performance proof. Private
  // provider-relative timing scores are not accepted by this contract.
  kPreferProviderRegions,
};

enum class OperationFallbackPolicy : std::uint8_t {
  kAllow = 0,
  // Compile fails unless provider regions cover every logical node. This is
  // useful when a caller requires a captured/fused artifact and would rather
  // reject a specialization than silently execute single-operation steps.
  kReject,
};

enum class ProviderReplayPolicy : std::uint8_t {
  // Accept any valid declared class, including unknown. This does not prove
  // a zero-allocation or bounded-cost provider replay.
  kAllowAny = 0,
  // Only proposals declaring prepared replay may create artifacts. This is
  // independent of shared/delegated ownership and command capture support.
  kRequirePrepared,
};

enum class WorkspaceReusePolicy : std::uint8_t {
  // Reuse storage only when the existing lane/dependency schedule proves the
  // two lifetimes ordered. Unordered values never alias.
  kExistingOrderOnly = 0,
  // A compiler may add visible ordering edges to satisfy a hard storage
  // budget. Every added edge and its concurrency cost is reported.
  kAllowSerialization,
};

enum class OutputOwnership : std::uint8_t {
  // A materialized output uses a distinct caller-provided mutable binding.
  kCaller = 0,
  // A published input, constant or required View aliases storage already
  // supplied by its source binding. The output cannot be rebound; retrieve
  // the derived borrowed view from Invocation::output() after completion.
  kSourceAlias,
  // The Invocation owns the backing through its workspace. This is used when
  // a published zero-copy view is rooted in private graph storage; callers can
  // add an explicit Copy node when they require their own destination.
  kInvocation,
};

struct CompileOptions final {
  // Exactly one value per independent GraphBuilder dimension, in any order.
  // Derived IDs must not be bound. Empty is valid with no independent inputs.
  std::span<const ShapeBinding> shape{};
  std::span<const ConstantBinding> constants{};
  std::span<const NodePlacement> node_placements{};
  std::span<const ValuePlacement> value_placements{};
  std::span<const ExternalResourceContract> external_resources{};
  WorkspaceReusePolicy workspace_reuse =
      WorkspaceReusePolicy::kExistingOrderOnly;
  // Hard cap for node/domain preparation attempts during deterministic
  // placement. Exhaustion fails rather than silently selecting another plan.
  std::size_t max_placement_attempts = 4096;
  RegionSelectionPolicy region_selection =
      RegionSelectionPolicy::kPreferProviderRegions;
  OperationFallbackPolicy operation_fallback = OperationFallbackPolicy::kAllow;
  // Hard cold-path budget across all configured providers. Exhaustion fails
  // compilation instead of silently changing the selected plan.
  std::size_t max_region_proposals = 4096;
  // Independent cap offered to each provider. The effective writer capacity
  // is min(this value, remaining global budget); either limit is a hard error
  // rather than a silent truncation.
  std::size_t max_region_proposals_per_provider = 1024;
  // Hard aggregate storage budget for node identities copied from provider
  // proposals. Node-set exhaustion is a compile error, never truncation.
  std::size_t max_region_proposal_nodes = 65536;
  // Cold constraint for both operation and region providers. View/Copy and
  // caller-owned executor/callback costs are not provider replay contracts.
  ProviderReplayPolicy provider_replay = ProviderReplayPolicy::kAllowAny;
};

// Immutable, shareable compiled artifact. Copies share one allocator-owned
// plan. Mutable bindings and workspace never live here: every concurrent run
// creates a distinct Invocation.
class Executable final {
public:
  Executable() noexcept = default;
  HYLE_GRAPH_API Executable(const Executable &other) noexcept;
  HYLE_GRAPH_API Executable &operator=(const Executable &other) noexcept;
  HYLE_GRAPH_API Executable(Executable &&other) noexcept;
  HYLE_GRAPH_API Executable &operator=(Executable &&other) noexcept;
  HYLE_GRAPH_API ~Executable();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_GRAPH_API std::size_t input_count() const noexcept;
  HYLE_GRAPH_API std::size_t output_count() const noexcept;
  HYLE_GRAPH_API std::size_t node_count() const noexcept;
  HYLE_GRAPH_API std::size_t target_count() const noexcept;
  HYLE_GRAPH_API std::size_t storage_domain_count() const noexcept;
  HYLE_GRAPH_API std::size_t lane_count() const noexcept;
  HYLE_GRAPH_API std::size_t step_count() const noexcept;
  HYLE_GRAPH_API std::size_t segment_count() const noexcept;
  HYLE_GRAPH_API std::size_t segment_dependency_count() const noexcept;
  HYLE_GRAPH_API std::size_t required_queue_count() const noexcept;
  HYLE_GRAPH_API Result<bool>
  requires_queue(ExecutionLaneId lane) const noexcept;
  HYLE_GRAPH_API std::size_t region_count() const noexcept;
  HYLE_GRAPH_API std::size_t workspace_bytes() const noexcept;
  HYLE_GRAPH_API std::size_t workspace_backing_count() const noexcept;
  HYLE_GRAPH_API bool supports_concurrent_invocations() const noexcept;
  HYLE_GRAPH_API SpecializationSignature specialization() const noexcept;
  HYLE_GRAPH_API Result<ValueId> input_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<ValueId>
  output_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<OutputOwnership>
  output_ownership(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<Invocation> create_invocation() const noexcept;

private:
  friend struct detail::ExecutableAccess;
  explicit Executable(detail::PlanState *state) noexcept : state_(state) {}
  detail::PlanState *state_ = nullptr;
};

// Exclusive mutable run state: external bindings plus one Mem lifetime-plan
// instance. Creation may allocate; bind/reset/submit never grow metadata,
// search a provider, JIT, pack weights, or allocate workspace. bind() checks
// the candidate against already-bound external outputs (and, for a new output,
// external inputs), so all bindings together cost O(inputs*outputs+outputs^2)
// in the worst case while repeated submit remains O(emitted steps+bindings).
class Invocation final {
public:
  Invocation() noexcept = default;
  Invocation(const Invocation &) = delete;
  Invocation &operator=(const Invocation &) = delete;
  HYLE_GRAPH_API Invocation(Invocation &&other) noexcept;
  HYLE_GRAPH_API Invocation &operator=(Invocation &&other) noexcept;
  HYLE_GRAPH_API ~Invocation();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_GRAPH_API Status bind(ValueId value,
                             compute::ConstTensorView tensor) noexcept;
  HYLE_GRAPH_API Status bind(ValueId value,
                             compute::TensorView tensor) noexcept;
  HYLE_GRAPH_API bool in_flight() const noexcept;
  HYLE_GRAPH_API Status reset() noexcept;
  HYLE_GRAPH_API Status prepare_replay(std::span<const QueueBinding> queues,
                                       ReplayOptions options = {}) noexcept;
  // This is a cold-path commit point: all external bindings and transfer
  // routes are validated and physical Queue completion capacity is reserved
  // before the route becomes prepared. Rebinding/reset invalidates it.
  // Effective steady-state route selected by prepare_replay(). PreferCapture
  // may truthfully report commands, captured, or a per-segment hybrid.
  HYLE_GRAPH_API ReplayRoute replay_route() const noexcept;
  // Exact route selected by the latest successful preparation. This exposes
  // all host-blocking dependency sites before the first steady submission.
  HYLE_GRAPH_API Result<ReplayReport> replay_report() const noexcept;
  HYLE_GRAPH_API Result<ReplayDependencyRoute>
  dependency_route(std::size_t segment_index,
                   std::size_t dependency_index) const noexcept;
  // Returns a borrowed published output only after the latest submission has
  // reached a successful terminal state. Invocation-owned views remain valid
  // until reset, the next submit, or Invocation destruction.
  HYLE_GRAPH_API Result<compute::ConstTensorView>
  output(ValueId value) const noexcept;
  HYLE_GRAPH_API Result<compute::Completion> submit() noexcept;
  HYLE_GRAPH_API Result<compute::Completion>
  submit(compute::Queue &queue) noexcept;
  HYLE_GRAPH_API Result<compute::Completion>
  submit(std::span<const QueueBinding> queues) noexcept;

private:
  friend struct detail::InvocationAccess;
  explicit Invocation(detail::InvocationState *state) noexcept
      : state_(state) {}
  detail::InvocationState *state_ = nullptr;
};

struct CompiledGraph final {
  Executable executable{};
  CompileReport report{};
};

HYLE_GRAPH_API Result<CompiledGraph>
compile(const Graph &graph, const compute::Context &context,
        CompileOptions options = {}) noexcept;
HYLE_GRAPH_API Result<CompiledGraph>
compile(const Graph &graph, CompileEnvironment environment,
        CompileOptions options = {}) noexcept;

} // namespace hyle::graph
