#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/provider.h"
#include "hyle/compute/route_profile.h"
#include "hyle/compute/tensor.h"
#include "hyle/error/result.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/visibility.h"
#include "hyle/mem/buffer.h"
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

// Cold association with one actual configured Provider slot. instance is
// checked against Context, then discarded: only build/configuration and the
// selected provider's stable implementation key enter a persistent profile.
struct ProfileProviderBinding final {
  ExecutionTargetId target{};
  std::size_t provider_index = 0;
  compute::ProviderInstanceId instance{};
  compute::ProfileFingerprint build{};
  compute::ProfileFingerprint configuration{};
};

// Optional expert input from the owner of qualified measurements. Ordered
// targets/storage_domains correspond exactly to CompileEnvironment. Backend
// IDs and provider slots are checked; build/hardware/runtime/configuration
// fingerprints are owner assertions, never inferred from names or pointers.
//
// semantics identifies complete schema/instance semantics (especially custom
// descriptor attributes); Graph also hashes its actual builtin attributes,
// topology and shape program. replay_configuration covers the assumed Queue
// topology, command/capture mode, binding/resource class and host-wait policy.
// Selection predicts only this declared workload, not a later different replay
// setup. workload_data covers representative data AND measurement
// qualification. All-zero facts remain unknown; nonzero hashes alone prove no
// qualification.
struct CompileProfileContext final {
  compute::ProfileFingerprint hyle_build{};
  compute::ProfileFingerprint semantics{};
  compute::ProfileFingerprint workload_data{};
  compute::ProfileFingerprint replay_configuration{};
  std::span<const compute::ProfileTargetIdentity> targets{};
  std::span<const compute::ProfileFingerprint> storage_domains{};
  std::span<const ProfileProviderBinding> providers{};
  compute::WorkloadIntent workload{};
  // Finite fraction in [0,1). Zero still requires disjoint uncertainty
  // intervals. This is a caller's economic policy, not a benchmark threshold.
  double minimum_relative_improvement = 0;
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

// Hard constraint on the provider preparing a logical node, including any
// region containing it. An invalid instance accepts any configured slot of
// the required family; a valid one must match that exact Context slot. Never
// persist instance IDs. Fusion, fallback and resource retries cannot waive
// this constraint. Graph-inserted transfers are backend work, not providers.
struct NodeProviderConstraint final {
  NodeId node{};
  compute::ProviderId provider{};
  compute::ProviderInstanceId instance{};
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
  // Zero-copy metadata owned by Graph, not a prepared Provider artifact.
  kView,
};

enum class CompileFallbackReason : std::uint8_t {
  kNone = 0,
  kRegionSelectionDisabled,
  kNoEligibleRegion,
  kRegionNotSelected,
  // At least one region containing this node was transactionally rejected
  // during this compile; the final plan uses an allowed operation route.
  kRegionDeclined,
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
  // Configured slot identity, including duplicate installations of one
  // family. Non-owning and process-local; not a portable performance key.
  compute::ProviderInstanceId provider_instance{};
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
  // Provider-declared stable implementation component of a full profile key.
  // Unknown is not filled from the prepare token or provider instance. Graph
  // boundary copies have no Provider key. Nonzero is not timing qualification.
  compute::ProviderRouteKey profile_key{};
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

// Facts from cold discovery/preparation, not an optimality certificate. The
// returned Executable has passed all feasibility and resource checks even
// when candidate discovery was limited. A full but untruncated writer alone
// does not mean optimization_limited().
struct CompileSearchReport final {
  std::size_t operation_proposals = 0;
  std::size_t region_proposals = 0;
  std::size_t region_proposal_nodes = 0;
  std::size_t placement_attempts = 0;
  // Selected region attempts rejected by an explicit transactional decline or
  // incompatible resources. Other resource bindings may remain eligible.
  std::size_t region_route_rejections = 0;
  // Selected leaf attempts rejected by a transactional prepare decline.
  std::size_t operation_route_rejections = 0;
  // Cold node-cover visits and branching resource/copy-target/lane choices.
  // Replayed choice prefixes and decisions with only one legal value are free.
  std::size_t route_search_states = 0;
  // Complete node covers rejected by physical resource/copy/cycle checks or
  // final lane/workspace constraints. This rejects a combination, not each
  // participating provider independently. Real errors never count as declines.
  std::size_t route_plan_rejections = 0;
  std::size_t feasible_plans = 0;
  std::size_t profiled_plans = 0;
  std::size_t unprofiled_plans = 0;
  std::size_t profile_replacements = 0;
  std::optional<compute::ProfileTimingInterval> selected_profile_cost{};
  bool region_proposal_limit_reached = false;
  bool per_provider_proposal_limit_reached = false;
  bool region_node_limit_reached = false;
  bool operation_proposal_limit_reached = false;
  bool route_search_limit_reached = false;
  bool placement_limit_reached = false;

  constexpr bool optimization_limited() const noexcept {
    return route_search_limit_reached || placement_limit_reached ||
           operation_proposal_limit_reached || region_proposal_limit_reached ||
           per_provider_proposal_limit_reached || region_node_limit_reached;
  }
};

enum class ConstantPolicy : std::uint8_t {
  kBorrow = 0,
  // Compile makes one canonical copy in the constant's external storage
  // domain (default zero). It waits for that copy, not for user graph work.
  // The source may be released when compile returns; failure never owns it.
  kMaterialize,
};

struct CompileConstantReport final {
  ValueId value{};
  ConstantPolicy policy = ConstantPolicy::kBorrow;
  StorageDomainId storage{};
  compute::TensorDesc description{};
  // Canonical allocation owned by the Executable, not provider packing or
  // invocation workspace. The report does not retain this allocation.
  std::size_t materialized_bytes = 0;
};

// Storage retained by compiled artifacts, not total process/device memory.
// Counts requested canonical bytes and provider-reported persistent/packed
// storage. Excludes Graph/Compute control metadata, allocator overhead,
// borrowed external backing, Invocation workspace/capture and compile scratch.
// Provider artifacts are charged independently unless sharing is proven;
// unknown provider storage is never treated as zero.
struct CompiledStorageReport final {
  std::size_t materialized_constant_bytes = 0;
  std::size_t provider_persistent_bytes = 0; // Known portion only.
  std::size_t packed_constant_bytes = 0;
  std::size_t unknown_provider_count = 0;
  // Sum is checked before a report is published.
  constexpr std::size_t known_bytes() const noexcept {
    return materialized_constant_bytes + provider_persistent_bytes +
           packed_constant_bytes;
  }
  constexpr std::optional<std::size_t> total_bytes() const noexcept {
    return unknown_provider_count == 0
               ? std::optional<std::size_t>{known_bytes()}
               : std::nullopt;
  }
};

enum class UnknownStoragePolicy : std::uint8_t {
  kReject = 0,
  // An explicit partial budget: only known bytes are constrained. A report
  // marking unknown storage does NOT prove that total storage is within limit.
  kAllow,
};

struct CompiledStorageLimits final {
  // Covers the general route and every exact variant together. SIZE_MAX means
  // no limit, including no unknown-storage restriction. This is a retained
  // storage admission limit, NOT a peak-allocation quota during compilation.
  std::size_t max_bytes = SIZE_MAX;
  UnknownStoragePolicy unknown = UnknownStoragePolicy::kReject;
};

struct CompilationStorageLimits final {
  // Family-wide newly acquired storage: supplied-allocator requests, canonical
  // materializations and declared extra Provider private bounds. Excludes the
  // input Graph/external backing, Context/Queue/backend control overhead,
  // allocator rounding, Status diagnostics and Invocation/capture storage.
  std::size_t max_resident_bytes = SIZE_MAX;
  // Admission before each managed allocation/Provider preparation. Resident
  // is checked after cold scratch dies, before the complete family is returned.
  // Either finite limit rejects unknown private bounds unless kAllow is set.
  std::size_t max_peak_bytes = SIZE_MAX;
  UnknownStoragePolicy unknown = UnknownStoragePolicy::kReject;
};

struct CompilationStorageReport final {
  // Snapshot when the whole family was published, after cold scratch died.
  // Private Provider bounds can be conservative, not exact physical bytes.
  std::size_t resident_bytes = 0;
  std::size_t peak_bytes = 0;
  // Includes unknown private storage in attempted, discarded candidates.
  bool has_unknown_storage = false;
};

struct InvocationStorageLimits final {
  // One shared limit for all live invocations of this compiled family,
  // including independently retained exact variants. Counts allocator requests,
  // actual workspace backing requests and declared additional private upper
  // bounds. Not physical RSS, caller bindings, Context/Queue or compiled state.
  std::size_t max_bytes = SIZE_MAX;
  UnknownStoragePolicy unknown = UnknownStoragePolicy::kReject;
};

struct InvocationStorageUsage final {
  std::size_t live_bytes = 0;
  std::size_t peak_bytes = 0;
  std::size_t limit_bytes = SIZE_MAX;
  // Sticky: an attempted invocation/capture had unbounded private storage.
  // The counts then constrain only the known portion, even after its teardown.
  bool has_unknown_storage = false;
};

// Copyable cold-path report that owns a compact immutable copy of the emitted
// plan facts independently of the Executable and its heavyweight artifacts.
// Aggregate queries and indexed records are O(1). Provider name/version text
// is copied into the report; views returned by provider()/step() borrow its
// shared state and remain valid until the last report handle is destroyed.
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
  HYLE_GRAPH_API CompileSearchReport search() const noexcept;
  // Counts/limits across the general route and every requested exact route.
  // selected_profile_cost is empty: these are not one execution workload.
  HYLE_GRAPH_API CompileSearchReport total_search() const noexcept;
  HYLE_GRAPH_API std::size_t exact_variant_count() const noexcept;
  HYLE_GRAPH_API Result<CompileReport>
  exact_variant(std::size_t index) const noexcept;
  // Self-contained scalar facts plus spans borrowed from this report. Empty
  // unless compile received profile_context. The report owns no Context,
  // constant payload or prepared artifact. Pair with qualified full-Graph
  // timings to produce a compute::RouteProfile; never kernel-only timings.
  HYLE_GRAPH_API compute::RouteProfileIdentity
  profile_identity() const noexcept;
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
  HYLE_GRAPH_API std::size_t materialized_constant_count() const noexcept;
  HYLE_GRAPH_API std::size_t materialized_constant_bytes() const noexcept;
  // Single-route facts; total_* includes all retained variants, counting
  // actually shared canonical constants only once. An exact child's two
  // reports coincide and include the constants it independently keeps alive.
  HYLE_GRAPH_API CompiledStorageReport compiled_storage() const noexcept;
  HYLE_GRAPH_API CompiledStorageReport total_compiled_storage() const noexcept;
  // Absent unless compilation_storage was requested. Each child reports the
  // same family snapshot; retaining the report does not retain tensor payload.
  HYLE_GRAPH_API std::optional<CompilationStorageReport>
  compilation_storage() const noexcept;
  HYLE_GRAPH_API Result<CompileConstantReport>
  constant(std::size_t index = 0) const noexcept;
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
  // Domain reports contain conservative descriptor/workspace envelopes, not
  // an executable exact point. specialization() is invalid for such reports.
  HYLE_GRAPH_API bool is_shape_domain() const noexcept;
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

// Borrowed by default, through every Executable/Invocation/Completion lifetime.
// Materialize explicitly requests an Executable-owned canonical copy instead;
// the source must be ready and immutable only for the duration of compile().
// Aggregate provider packing never shortens a borrow. Resources themselves
// remain caller-owned even when Graph owns the allocation within them.
struct ConstantBinding final {
  ValueId value{};
  compute::ConstTensorView tensor{};
  ConstantPolicy policy = ConstantPolicy::kBorrow;
};

// Borrowed access footprint, not an execution argument or ownership transfer.
// memory() uses Mem's canonical allocation identity and checked subrange.
// object() is for non-addressable state; pass its canonical owner identity,
// never distinct wrapper addresses for the same resource. Addressable bytes
// must use memory(), so their overlap with graph tensor storage can be checked.
class EffectResourceFootprint final {
public:
  constexpr EffectResourceFootprint() noexcept = default;
  static EffectResourceFootprint memory(mem::ConstBufferView view) noexcept {
    EffectResourceFootprint result;
    result.memory_ = view;
    return result;
  }
  static EffectResourceFootprint object(const void *identity) noexcept {
    EffectResourceFootprint result;
    result.object_ = identity;
    return result;
  }
  bool valid() const noexcept { return memory_.valid() || object_ != nullptr; }
  bool is_memory() const noexcept { return memory_.valid(); }
  mem::ConstBufferView memory_range() const noexcept { return memory_; }
  const void *object_identity() const noexcept { return object_; }

private:
  mem::ConstBufferView memory_{};
  const void *object_ = nullptr;
};

struct EffectResourceBinding final {
  EffectResourceId resource{};
  EffectResourceFootprint footprint{};
};

enum class RegionSelectionPolicy : std::uint8_t {
  kDisabled = 0,
  // Region and leaf providers compete in the same bounded cold search.
  // Without qualified costs, configured target/provider order takes priority;
  // coverage/dispatch only break ties. This is not a performance certificate.
  kAllowProviderRegions,
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

enum class CompileSearchPolicy : std::uint8_t {
  // Finish and validate a plan using the admitted candidates; report any
  // discovery limit. Real errors or the absence of a feasible plan still fail.
  kAllowLimited = 0,
  // Reject discovery that omitted candidates/providers because of a limit.
  // This does not request or prove a globally optimal plan.
  kRequireUnexhausted,
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
  // A published materialized constant (or its View) borrows an immutable
  // Executable-owned allocation; it cannot be rebound as a mutable output.
  kExecutable,
};

enum class CompileShapePolicy : std::uint8_t {
  kExact,
  // Compile the complete bounds/guards declared by GraphBuilder. Every
  // dynamic node needs a Provider that proves and prepares the whole domain.
  // Exact-only implementations cannot stand in for an unproved domain.
  kDeclaredDomain,
};

struct ExactShapeVariant final {
  // One binding per independent dimension. Borrowed only during compile;
  // declaration order determines the stable variant index in the report.
  std::span<const ShapeBinding> shape{};
};

struct CompileOptions final {
  // Exactly one value per independent GraphBuilder dimension, in any order.
  // Derived IDs must not be bound. Empty is valid with no independent inputs.
  // Must be empty for kDeclaredDomain; bind at create_invocation() instead.
  std::span<const ShapeBinding> shape{};
  CompileShapePolicy shape_policy = CompileShapePolicy::kExact;
  // Only with kDeclaredDomain. Prepare every exact route now; unmatched valid
  // points use the complete general route. No lazy compilation or implicit
  // cache. Duplicate points and any unpreparable requested route fail compile.
  // All cumulative search/preparation caps below cover the whole family, not
  // each route separately. Workspace budgets remain per-route/per-Invocation.
  std::span<const ExactShapeVariant> exact_variants{};
  std::span<const ConstantBinding> constants{};
  CompiledStorageLimits compiled_storage{};
  std::optional<CompilationStorageLimits> compilation_storage{};
  // Opt-in; absent means no tracker or quota on the ordinary create path.
  // Metadata and workspace stay charged until their actual last owner frees
  // them, including abandoned asynchronous Completion or retained signatures.
  std::optional<InvocationStorageLimits> invocation_storage{};
  // Every used effect resource needs one binding. Backing is fixed/borrowed
  // for this Executable and all its Invocations/Completions. It must describe
  // what the operation implementation actually accesses; this does not pass
  // data to that implementation. Cross-Invocation ordering remains the
  // caller's.
  std::span<const EffectResourceBinding> effect_bindings{};
  std::span<const NodePlacement> node_placements{};
  std::span<const ValuePlacement> value_placements{};
  std::span<const ExternalResourceContract> external_resources{};
  // Borrowed for compile() only. Empty allow means unrestricted; deny wins.
  // Lists contain family IDs, not backend/target IDs; repeats are harmless.
  // A globally excluded provider is not queried or prepared by Graph.
  std::span<const compute::ProviderId> allowed_providers{};
  std::span<const compute::ProviderId> denied_providers{};
  // At most one declaration per graph-owned node, with a valid family ID.
  // No feasible route under these constraints is an error, not permission to
  // use a different implementation. This is independent of target placement.
  std::span<const NodeProviderConstraint> node_providers{};
  WorkspaceReusePolicy workspace_reuse =
      WorkspaceReusePolicy::kExistingOrderOnly;
  // Cumulative hard cap for selected leaf/region artifact preparation,
  // including retry prefixes. Exhaustion before a complete feasible plan fails;
  // later exhaustion may return the retained incumbent under kAllowLimited.
  std::size_t max_placement_attempts = 4096;
  RegionSelectionPolicy region_selection =
      RegionSelectionPolicy::kAllowProviderRegions;
  // Cumulative node-cover and branching storage/copy-target/lane state budget.
  // No recursion, Cartesian-product allocation, or warm search.
  // Exhaustion without a complete feasible route fails, never emits a partial
  // DAG.
  std::size_t max_route_search_states = 65536;
  OperationFallbackPolicy operation_fallback = OperationFallbackPolicy::kAllow;
  // Independent storage cap for admitted operation proposals across all
  // nodes/targets/providers. Discovery does not prepare artifacts or spend
  // max_placement_attempts. A limit may not leave the final plan incomplete.
  // Unused when operation_fallback is kReject.
  std::size_t max_operation_proposals = 16384;
  // Hard cold-path admission/storage caps. Exhaustion stops further discovery
  // and is reported; search_policy decides whether a feasible limited plan
  // is acceptable. It never waives placement, coverage or workspace checks.
  std::size_t max_region_proposals = 4096;
  // Independent cap offered to each provider. The effective writer capacity
  // is min(this value, remaining global budget). Reaching this provider's
  // limit does not prevent discovery from a later provider with room left.
  std::size_t max_region_proposals_per_provider = 1024;
  // Hard aggregate storage budget for node identities copied from provider
  // proposals. A refused node set is never partially admitted.
  std::size_t max_region_proposal_nodes = 65536;
  // Cold constraint for operation and region providers, including a region
  // containing Views. Graph metadata Views, inserted backend transfers and
  // caller-owned executor/callback costs are not provider replay contracts.
  ProviderReplayPolicy provider_replay = ProviderReplayPolicy::kAllowAny;
  CompileSearchPolicy search_policy = CompileSearchPolicy::kAllowLimited;
  // Borrowed only through compile(). No implicit measurement or warm lookup.
  // Context alone exports a plan key; profiles also enable bounded cost search
  // when the first feasible route has a qualified complete cost. Unknown or
  // overlapping costs never displace the deterministic incumbent.
  const CompileProfileContext *profile_context = nullptr;
  std::span<const compute::RouteProfile> profiles{};
};

// Immutable, shareable compiled artifact. Copies share one allocator-owned
// plan. Mutable tensor bindings and workspace never live here: every concurrent
// run creates a distinct Invocation. Fixed effect footprints are shared
// borrowed declarations; callers still synchronize conflicting external
// resource access across Invocations, even when the artifacts support
// concurrent invocations.
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
  HYLE_GRAPH_API bool is_shape_domain() const noexcept;
  HYLE_GRAPH_API std::size_t exact_variant_count() const noexcept;
  // nullopt if no budget was requested (or this handle is invalid). Individual
  // counters may observe different instants during concurrent create/free;
  // admission itself never exceeds the shared hard limit.
  HYLE_GRAPH_API std::optional<InvocationStorageUsage>
  invocation_storage_usage() const noexcept;
  // An independently owning route handle. Top-level plan queries describe the
  // general route; query a child for its exact workspace/ownership contract.
  HYLE_GRAPH_API Result<Executable>
  exact_variant(std::size_t index) const noexcept;
  // Invalid for a domain; the concrete signature belongs to its Invocation.
  HYLE_GRAPH_API SpecializationSignature specialization() const noexcept;
  HYLE_GRAPH_API Result<ValueId> input_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<ValueId>
  output_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<OutputOwnership>
  output_ownership(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<Invocation> create_invocation() const noexcept;
  // Cold binding of all independent dimensions for a declared domain. Selects
  // a precompiled exact variant if present, otherwise the general route. Does
  // not recompile or select Providers. Query the returned Invocation's actual
  // executable for its workspace/queue contract. Exact executables accept
  // only an empty span.
  HYLE_GRAPH_API Result<Invocation>
  create_invocation(std::span<const ShapeBinding> shape) const noexcept;

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
// in the worst case, plus O(effect resources) per bind for tensor/effect alias
// validation. Repeated submit remains O(emitted steps+bindings), with no
// separate scan of effect resources.
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
  // Exact values fixed at creation, including derived dimensions. reset()
  // clears tensor bindings, not this signature; another shape needs a new run.
  HYLE_GRAPH_API SpecializationSignature specialization() const noexcept;
  // The actual selected route, retained independently of the parent table.
  HYLE_GRAPH_API Executable executable() const noexcept;
  // nullopt for general fallback or an ordinary standalone exact Executable.
  HYLE_GRAPH_API std::optional<std::size_t>
  exact_variant_index() const noexcept;
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
