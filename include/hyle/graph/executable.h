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

class ExecutionDomainId final {
public:
  constexpr ExecutionDomainId() noexcept = default;
  constexpr explicit ExecutionDomainId(std::uint32_t ordinal) noexcept
      : ordinal_(ordinal) {}

  constexpr bool valid() const noexcept { return ordinal_ != UINT32_MAX; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  constexpr std::uint32_t ordinal() const noexcept { return ordinal_; }

  friend constexpr bool operator==(ExecutionDomainId,
                                   ExecutionDomainId) noexcept = default;

private:
  std::uint32_t ordinal_ = UINT32_MAX;
};

// Borrowed cold compile input. Context is retained internally by the emitted
// plan; the MemoryResource provider remains caller-owned and must outlive the
// Executable, its Invocations, and all submitted work.
struct ExecutionDomain final {
  const compute::Context *context = nullptr;
  mem::MemoryResourceRef workspace_resource{};
};

struct CompileEnvironment final {
  std::span<const ExecutionDomain> domains{};
};

enum class PlacementStrength : std::uint8_t {
  kPreferred = 0,
  kRequired,
};

struct NodePlacement final {
  NodeId node{};
  ExecutionDomainId domain{};
  PlacementStrength strength = PlacementStrength::kPreferred;
};

struct ValuePlacement final {
  ValueId value{};
  ExecutionDomainId domain{};
  PlacementStrength strength = PlacementStrength::kPreferred;
};

// Exact compile-time resource locality for a caller-bound graph value. The
// allocation itself may change between replays, but its resource/domain may
// not. Omitted source contracts use domain zero; an omitted produced output
// stays with its producer. Both defer exact resource validation to the owning
// backend, preserving the compact single-domain convenience path. An external
// value that crosses a domain boundary requires an exact contract so compile
// can select a real transfer executor instead of guessing from a workspace
// resource.
struct ExternalResourceContract final {
  ValueId value{};
  ExecutionDomainId domain{};
  mem::MemoryResourceRef resource{};
};

struct QueueBinding final {
  ExecutionDomainId domain{};
  compute::Queue *queue = nullptr;
};

enum class ReplayMode : std::uint8_t {
  kCommands = 0,
  kPreferCapture,
  kRequireCapture,
};

enum class HostWaitPolicy : std::uint8_t {
  // Permit prepare_replay() to select an explicit host wait when the exact
  // destination Queue cannot consume a source Queue's asynchronous prefix.
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
  // The destination Queue consumes the source completion without blocking
  // the submitting host thread.
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
// canonical binding, so a collision can never select the wrong Executable.
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
  ExecutionDomainId domain{};
  ExecutionDomainId source_domain{};
  ExecutionDomainId destination_domain{};
  std::uint32_t node_begin = 0;
  std::uint32_t node_count = 0;
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
  bool blocks_host = false;
  bool uses_private_threads = false;
  bool supports_concurrent_invocations = true;
  bool capture_compatible = false;
};

struct CompileDomainReport final {
  ExecutionDomainId domain{};
  compute::Device device{};
  std::size_t workspace_bytes = 0;
  std::size_t workspace_backing_count = 0;
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
  ExecutionDomainId domain{};
  compute::TensorDesc description{};
  std::optional<std::uint32_t> alias_source{};
  bool external_input = false;
  bool external_output = false;
  bool constant = false;
  bool invocation_owned_output = false;
  bool workspace_backed = false;
};

// A Queue segment is the unit of finalization and optional command capture.
// Dependencies are queried through CompileReport::segment_dependency().
struct CompileSegmentReport final {
  ExecutionDomainId domain{};
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
  // Both segments belong to one domain and therefore use the same ordered
  // Queue; no explicit wait is emitted.
  kOrderedQueue = 0,
  // The source domain's Queue completes inline, so a cross-domain data/control
  // edge needs no runtime wait even when Queue objects differ.
  kSourceCompletesInline,
  // The concrete Queue bindings decide the route. Reusing one Queue needs no
  // explicit wait; otherwise prepare_replay() fixes either a device-side or
  // explicit host wait before steady execution begins.
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
  HYLE_GRAPH_API std::size_t domain_count() const noexcept;
  HYLE_GRAPH_API std::size_t step_count() const noexcept;
  HYLE_GRAPH_API std::size_t segment_count() const noexcept;
  HYLE_GRAPH_API std::size_t segment_dependency_count() const noexcept;
  HYLE_GRAPH_API std::size_t command_segment_count() const noexcept;
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
  HYLE_GRAPH_API Result<CompileDomainReport>
  domain(std::size_t index) const noexcept;
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
  // deterministic maximum-coverage set, then fewer backend execution
  // episodes, then earlier provider order. A provider may omit a timing
  // estimate, but an explicit zero estimate is not selected.
  kPreferProviderRegions,
  // Accept only proposals with a positive provider-declared portable net
  // saving estimate. This is for deployments that require quantified cost
  // evidence rather than explicit provider opt-in.
  kRequireEstimatedBenefit,
};

enum class OperationFallbackPolicy : std::uint8_t {
  kAllow = 0,
  // Compile fails unless provider regions cover every logical node. This is
  // useful when a caller requires a captured/fused artifact and would rather
  // reject a specialization than silently execute single-operation steps.
  kReject,
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
  // Exact-once values for every bounded symbol declared by GraphBuilder.
  // Empty is valid only for a graph with no symbolic dimensions.
  std::span<const ShapeBinding> shape{};
  std::span<const ConstantBinding> constants{};
  std::span<const NodePlacement> node_placements{};
  std::span<const ValuePlacement> value_placements{};
  std::span<const ExternalResourceContract> external_resources{};
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
  HYLE_GRAPH_API std::size_t domain_count() const noexcept;
  HYLE_GRAPH_API std::size_t step_count() const noexcept;
  HYLE_GRAPH_API std::size_t segment_count() const noexcept;
  HYLE_GRAPH_API std::size_t segment_dependency_count() const noexcept;
  HYLE_GRAPH_API std::size_t required_queue_count() const noexcept;
  HYLE_GRAPH_API Result<bool>
  requires_queue(ExecutionDomainId domain) const noexcept;
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
