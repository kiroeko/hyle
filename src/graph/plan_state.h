#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "hyle/compute/detail/command_capture.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/compute/detail/provider_extension.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/container/small_vector.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/region_provider.h"
#include "hyle/mem/lifetime_plan.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"

namespace hyle::graph::detail {

inline constexpr std::uint32_t kNoWorkspaceRequest = UINT32_MAX;
inline constexpr std::uint32_t kNoInPlaceSource = UINT32_MAX;
inline constexpr std::uint8_t kExternalInput = 1u << 0;
inline constexpr std::uint8_t kExternalOutput = 1u << 1;
inline constexpr std::uint8_t kConstant = 1u << 2;
inline constexpr std::uint8_t kInvocationOwnedOutput = 1u << 3;
inline constexpr std::uint8_t kRetainedOutputRoot = 1u << 4;
inline constexpr std::uint8_t kSourceAliasedOutput = 1u << 5;

class PreparedRegion;

class RegionInvocation final {
public:
  RegionInvocation() noexcept = default;
  RegionInvocation(const RegionInvocation &) = delete;
  RegionInvocation &operator=(const RegionInvocation &) = delete;
  RegionInvocation(RegionInvocation &&other) noexcept
      : prepared_(other.prepared_), state_(other.state_) {
    other.prepared_ = nullptr;
    other.state_ = nullptr;
  }
  RegionInvocation &operator=(RegionInvocation &&other) noexcept;
  ~RegionInvocation();

  bool valid() const noexcept { return prepared_ != nullptr; }
  compute::detail::EnqueueResult
  enqueue(compute::detail::QueueSequence &sequence,
          std::span<const compute::ConstTensorView> inputs,
          std::span<const compute::TensorView> outputs) noexcept;

private:
  void reset() noexcept;
  friend class PreparedRegion;
  RegionInvocation(const PreparedRegion *prepared, void *state) noexcept
      : prepared_(prepared), state_(state) {}

  const PreparedRegion *prepared_ = nullptr;
  void *state_ = nullptr;
};

// Graph-owned wrapper around a provider's compiled region. The extension
// handle retains the exact Context/provider state; provider metadata and the
// per-invocation workspace remain separately owned.
class PreparedRegion final {
public:
  PreparedRegion() noexcept = default;
  PreparedRegion(compute::detail::ProviderExtensionRef extension,
                 ProviderRegionPrepared prepared,
                 mem::AllocatorRef allocator) noexcept
      : extension_(std::move(extension)), prepared_(prepared),
        allocator_(allocator) {}
  PreparedRegion(const PreparedRegion &) = delete;
  PreparedRegion &operator=(const PreparedRegion &) = delete;
  PreparedRegion(PreparedRegion &&other) noexcept
      : extension_(std::move(other.extension_)), prepared_(other.prepared_),
        allocator_(other.allocator_) {
    other.prepared_ = {};
    other.allocator_ = {};
  }
  PreparedRegion &operator=(PreparedRegion &&other) noexcept {
    if (this != &other) {
      reset();
      extension_ = std::move(other.extension_);
      prepared_ = other.prepared_;
      allocator_ = other.allocator_;
      other.prepared_ = {};
      other.allocator_ = {};
    }
    return *this;
  }
  ~PreparedRegion() { reset(); }

  bool valid() const noexcept {
    return extension_.valid() && prepared_.vtable != nullptr;
  }
  compute::ProviderIdentity provider() const noexcept {
    return extension_.provider();
  }
  RegionArtifactProperties properties() const noexcept {
    return prepared_.properties;
  }
  Result<compute::AliasSupport>
  alias_support(std::size_t output_index,
                std::size_t input_index) const noexcept {
    if (!valid() || prepared_.vtable->alias_support == nullptr) {
      return std::unexpected(
          Status::Internal("compiled graph region alias contract is invalid"));
    }
    return prepared_.vtable->alias_support(prepared_.state, output_index,
                                           input_index);
  }

  Result<RegionInvocation>
  create_invocation(mem::BufferView workspace) const noexcept;
  compute::detail::EnqueueResult
  enqueue(void *invocation, compute::detail::QueueSequence &sequence,
          std::span<const compute::ConstTensorView> inputs,
          std::span<const compute::TensorView> outputs) const noexcept;

private:
  friend class RegionInvocation;
  void reset() noexcept {
    if (prepared_.vtable != nullptr &&
        prepared_.vtable->destroy_prepared != nullptr) {
      prepared_.vtable->destroy_prepared(prepared_.state, allocator_);
    }
    prepared_ = {};
  }

  compute::detail::ProviderExtensionRef extension_{};
  ProviderRegionPrepared prepared_{};
  mem::AllocatorRef allocator_{};
};

inline RegionInvocation &
RegionInvocation::operator=(RegionInvocation &&other) noexcept {
  if (this != &other) {
    reset();
    prepared_ = other.prepared_;
    state_ = other.state_;
    other.prepared_ = nullptr;
    other.state_ = nullptr;
  }
  return *this;
}

inline RegionInvocation::~RegionInvocation() { reset(); }

inline void RegionInvocation::reset() noexcept {
  if (prepared_ != nullptr && prepared_->valid() &&
      prepared_->prepared_.vtable->destroy_invocation != nullptr) {
    prepared_->prepared_.vtable->destroy_invocation(state_,
                                                    prepared_->allocator_);
  }
  prepared_ = nullptr;
  state_ = nullptr;
}

inline compute::detail::EnqueueResult RegionInvocation::enqueue(
    compute::detail::QueueSequence &sequence,
    std::span<const compute::ConstTensorView> inputs,
    std::span<const compute::TensorView> outputs) noexcept {
  if (!valid()) {
    return compute::detail::EnqueueResult::rejected(
        Status::FailedPrecondition("graph region invocation is invalid"));
  }
  return prepared_->enqueue(state_, sequence, inputs, outputs);
}

inline Result<RegionInvocation>
PreparedRegion::create_invocation(mem::BufferView workspace) const noexcept {
  if (!valid() || prepared_.vtable->create_invocation == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compiled graph region is invalid"));
  }
  struct Episode final {
    const PreparedRegion *region = nullptr;
    mem::BufferView workspace{};
    void *state = nullptr;
  } episode{this, workspace, nullptr};
  const Status status = compute::detail::ProviderExtensionAccess::call(
      extension_,
      +[](void *, const void *, void *opaque) noexcept -> Status {
        auto &call = *static_cast<Episode *>(opaque);
        return call.region->prepared_.vtable->create_invocation(
            call.region->prepared_.state, call.region->allocator_,
            call.workspace, &call.state);
      },
      &episode);
  if (!status.ok()) {
    // A failing provider is still responsible for returning either nullptr or
    // a destroyable partial invocation. Cleaning the latter here makes OOM and
    // injected-failure rollback symmetric with prepared-region compilation.
    if (episode.state != nullptr &&
        prepared_.vtable->destroy_invocation != nullptr) {
      prepared_.vtable->destroy_invocation(episode.state, allocator_);
    }
    return std::unexpected(status);
  }
  return RegionInvocation(this, episode.state);
}

inline compute::detail::EnqueueResult PreparedRegion::enqueue(
    void *invocation, compute::detail::QueueSequence &sequence,
    std::span<const compute::ConstTensorView> inputs,
    std::span<const compute::TensorView> outputs) const noexcept {
  struct Episode final {
    const PreparedRegion *region = nullptr;
    void *invocation = nullptr;
    std::span<const compute::ConstTensorView> inputs{};
    std::span<const compute::TensorView> outputs{};
  } episode{this, invocation, inputs, outputs};
  return compute::detail::ProviderExtensionAccess::enqueue_admitted(
      extension_, sequence, inputs, outputs,
      +[](void *, const void *, void *opaque,
          compute::Queue &submitted_queue) noexcept
          -> compute::detail::EnqueueResult {
        auto &call = *static_cast<Episode *>(opaque);
        return call.region->prepared_.vtable->submit(
            call.region->prepared_.state, call.invocation, submitted_queue,
            call.inputs, call.outputs);
      },
      &episode);
}

enum class PlanStepKind : std::uint8_t {
  kOperation = 0,
  kRegion,
  kCopy,
};

struct PlanStep final {
  PlanStep() noexcept = default;
  PlanStep(compute::PreparedOperation prepared_value,
           std::uint32_t node_begin_value, std::uint32_t input_begin_value,
           std::uint32_t input_count_value, std::uint32_t output_begin_value,
           std::uint32_t output_count_value, bool materializes_value,
           CompileFallbackReason fallback_reason_value,
           ExecutionTargetId target_value = ExecutionTargetId{0},
           StorageDomainId workspace_storage_value =
               StorageDomainId{0}) noexcept
      : operation(std::move(prepared_value)), node_begin(node_begin_value),
        input_begin(input_begin_value), output_begin(output_begin_value),
        input_count(input_count_value), output_count(output_count_value),
        materializes(materializes_value),
        fallback_reason(fallback_reason_value), target(target_value),
        workspace_storage(workspace_storage_value) {}

  PlanStep(PreparedRegion region_value, std::uint32_t node_begin_value,
           std::uint32_t node_count_value, std::uint32_t input_begin_value,
           std::uint32_t input_count_value, std::uint32_t output_begin_value,
           std::uint32_t output_count_value,
           ExecutionTargetId target_value = ExecutionTargetId{0},
           StorageDomainId workspace_storage_value =
               StorageDomainId{0}) noexcept
      : region(std::move(region_value)), node_begin(node_begin_value),
        node_count(node_count_value), input_begin(input_begin_value),
        output_begin(output_begin_value), input_count(input_count_value),
        output_count(output_count_value), kind(PlanStepKind::kRegion),
        target(target_value), workspace_storage(workspace_storage_value) {}

  static PlanStep copy(std::uint32_t input_begin_value,
                       std::uint32_t output_begin_value,
                       std::size_t bytes_value,
                       StorageDomainId source_storage_value,
                       StorageDomainId destination_storage_value,
                       ExecutionTargetId execution_target_value) noexcept {
    PlanStep result;
    result.input_begin = input_begin_value;
    result.output_begin = output_begin_value;
    result.input_count = 1;
    result.output_count = 1;
    result.kind = PlanStepKind::kCopy;
    result.materializes = true;
    result.copy_bytes = bytes_value;
    result.source_storage = source_storage_value;
    result.target = execution_target_value;
    result.destination_storage = destination_storage_value;
    return result;
  }

  PlanStep(const PlanStep &) = delete;
  PlanStep &operator=(const PlanStep &) = delete;
  PlanStep(PlanStep &&) noexcept = default;
  PlanStep &operator=(PlanStep &&) noexcept = default;

  compute::ProviderIdentity provider() const noexcept {
    if (kind == PlanStepKind::kRegion)
      return region.provider();
    if (kind == PlanStepKind::kOperation)
      return operation.provider();
    return {};
  }
  compute::WorkspaceRequirement workspace_requirement() const noexcept {
    if (kind == PlanStepKind::kRegion)
      return region.properties().workspace;
    if (kind == PlanStepKind::kOperation)
      return operation.workspace_requirement();
    return {};
  }
  // Output storage and execution are separate concerns. View is the one
  // built-in metadata-only operation: it derives an alias during binding
  // preparation and never enters a Queue. Custom operations may legally have
  // no outputs (for example an external side effect), so materializes must
  // never be used as a proxy for whether the provider has work to submit.
  bool emits_command() const noexcept {
    return kind != PlanStepKind::kOperation ||
           operation.kind() != compute::OperationKind::kView;
  }
  bool capture_compatible() const noexcept {
    if (!emits_command())
      return true;
    if (kind == PlanStepKind::kCopy)
      return true;
    if (kind == PlanStepKind::kRegion)
      return region.properties().capture_compatible;
    return operation.capture_compatible();
  }
  compute::ProviderRuntimeProperties runtime_properties() const noexcept {
    if (kind == PlanStepKind::kRegion) {
      const RegionArtifactProperties properties = region.properties();
      return compute::ProviderRuntimeProperties{
          .mode = compute::ProviderRuntimeMode::kSharedQueue,
          .workspace_owner = compute::ProviderWorkspaceOwner::kCaller,
          .blocks_host = properties.blocks_host,
          .uses_private_threads = properties.uses_private_threads,
          .supports_concurrent_invocations =
              properties.supports_concurrent_invocations,
          .capture_compatible = properties.capture_compatible,
      };
    }
    if (kind == PlanStepKind::kOperation)
      return operation.runtime_properties();
    return compute::ProviderRuntimeProperties{
        .mode = compute::ProviderRuntimeMode::kSharedQueue,
        .workspace_owner = compute::ProviderWorkspaceOwner::kCaller,
        .capture_compatible = true,
    };
  }

  compute::PreparedOperation operation{};
  PreparedRegion region{};
  std::uint32_t node_begin = 0;
  std::uint32_t node_count = 1;
  std::uint32_t input_begin = 0;
  std::uint32_t output_begin = 0;
  std::uint32_t input_count = 0;
  std::uint32_t output_count = 0;
  PlanStepKind kind = PlanStepKind::kOperation;
  bool materializes = true;
  CompileFallbackReason fallback_reason = CompileFallbackReason::kNone;
  ExecutionTargetId target{};
  ExecutionLaneId lane{};
  StorageDomainId workspace_storage{};
  StorageDomainId source_storage{};
  StorageDomainId destination_storage{};
  std::size_t copy_bytes = 0;
};

using PlanSteps = container::SmallVector<PlanStep, 8, mem::AllocatorRef>;
struct PlanSegment final {
  ExecutionLaneId lane{};
  std::uint32_t step_begin = 0;
  std::uint32_t step_count = 0;
  std::uint32_t dependency_begin = 0;
  std::uint32_t dependency_count = 0;
};
using PlanSegments = container::SmallVector<PlanSegment, 4, mem::AllocatorRef>;
using ValueIds = container::SmallVector<ValueId, 8, mem::AllocatorRef>;
using TensorDescriptions =
    container::SmallVector<compute::TensorDesc, 16, mem::AllocatorRef>;
using ValueOrdinals =
    container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
using BindingRoles =
    container::SmallVector<std::uint8_t, 16, mem::AllocatorRef>;
struct StepInvocation final {
  compute::OperationInvocation operation{};
  RegionInvocation region{};
};
using StepInvocations =
    container::SmallVector<StepInvocation, 8, mem::AllocatorRef>;
using ReadableBindings =
    container::SmallVector<compute::ConstTensorView, 16, mem::AllocatorRef>;
using WritableBindings =
    container::SmallVector<compute::TensorView, 16, mem::AllocatorRef>;
using CompileStepReports =
    container::SmallVector<CompileStepReport, 8, mem::AllocatorRef>;
using CompileSegmentReports =
    container::SmallVector<CompileSegmentReport, 4, mem::AllocatorRef>;
using CompilePhysicalValueReports =
    container::SmallVector<CompilePhysicalValueReport, 16, mem::AllocatorRef>;
using ShapeBindings =
    container::SmallVector<ShapeBinding, 4, mem::AllocatorRef>;

struct StorageDomainRecord final {
  mem::MemoryResourceRef resource{};
  std::size_t workspace_budget_bytes = SIZE_MAX;
};

struct ExecutionTargetRecord final {
  compute::detail::ContextRef context{};
  StorageDomainId default_storage{};
};

struct ExecutionLaneRecord final {
  ExecutionTargetId target{};
  ExecutionLaneRole role = ExecutionLaneRole::kCompute;
};

using StorageDomainRecords =
    container::SmallVector<StorageDomainRecord, 2, mem::AllocatorRef>;
using ExecutionTargetRecords =
    container::SmallVector<ExecutionTargetRecord, 2, mem::AllocatorRef>;
using ExecutionLaneRecords =
    container::SmallVector<ExecutionLaneRecord, 2, mem::AllocatorRef>;
using CompileTargetReports =
    container::SmallVector<CompileTargetReport, 2, mem::AllocatorRef>;
using CompileStorageReports =
    container::SmallVector<CompileStorageReport, 2, mem::AllocatorRef>;
using CompileLaneReports =
    container::SmallVector<CompileLaneReport, 2, mem::AllocatorRef>;

struct SignatureState final : util::RefCounted<SignatureState> {
  SignatureState(mem::AllocatorRef allocator_value,
                 std::uint64_t generation_value) noexcept
      : allocator(allocator_value), generation(generation_value),
        bindings(allocator_value) {}

  void destroy() noexcept {
    const mem::AllocatorRef owner = allocator;
    this->~SignatureState();
    owner.deallocate(this, sizeof(SignatureState), alignof(SignatureState));
  }

  mem::AllocatorRef allocator{};
  std::uint64_t generation = 0;
  ShapeBindings bindings;
  std::size_t hash = 0;
};

inline void retain(SignatureState *state) noexcept {
  if (state != nullptr)
    state->incref();
}

inline void release(SignatureState *state) noexcept {
  if (state != nullptr)
    state->decref();
}

struct PlanState final : util::RefCounted<PlanState> {
  PlanState(mem::AllocatorRef allocator_value,
            std::uint64_t generation_value) noexcept
      : allocator(allocator_value), generation(generation_value),
        storage_domains(allocator_value), targets(allocator_value),
        lanes(allocator_value), steps(allocator_value),
        segments(allocator_value), segment_dependencies(allocator_value),
        step_segments(allocator_value),
        terminal_segments(allocator_value),
        completion_segments(allocator_value),
        lane_segment_counts(allocator_value), input_ids(allocator_value),
        constant_ids(allocator_value), output_ids(allocator_value),
        constant_bindings(allocator_value), descriptions(allocator_value),
        instance_descriptions(allocator_value),
        instance_values(allocator_value), instance_storage(allocator_value),
        instance_roles(allocator_value), external_instances(allocator_value),
        external_storage(allocator_value),
        node_inputs(allocator_value), node_outputs(allocator_value),
        workspace_requests(allocator_value),
        step_workspace_requests(allocator_value),
        storage_workspace_slots(allocator_value),
        workspace_resources(allocator_value),
        workspace_storage(allocator_value), node_steps(allocator_value),
        value_materialized(allocator_value), alias_sources(allocator_value),
        alias_offsets(allocator_value), binding_roles(allocator_value) {}

  void destroy() noexcept {
    const mem::AllocatorRef owner = allocator;
    this->~PlanState();
    owner.deallocate(this, sizeof(PlanState), alignof(PlanState));
  }

  mem::AllocatorRef allocator{};
  std::uint64_t generation = 0;
  StorageDomainRecords storage_domains;
  ExecutionTargetRecords targets;
  ExecutionLaneRecords lanes;
  PlanSteps steps;
  PlanSegments segments;
  ValueOrdinals segment_dependencies;
  ValueOrdinals step_segments;
  // Logical DAG sinks are report facts. Runtime completion must additionally
  // observe the latest ordered prefix from every active execution lane so a
  // late backend error cannot disappear merely because another domain owns
  // the final graph sink.
  ValueOrdinals terminal_segments;
  ValueOrdinals completion_segments;
  // Exact number of runtime Queue segments emitted for each execution lane.
  // A zero entry means the lane participates in compile-time
  // placement/reporting only and must not require a Queue at replay.
  ValueOrdinals lane_segment_counts;
  ValueIds input_ids;
  ValueIds constant_ids;
  ValueIds output_ids;
  ReadableBindings constant_bindings;
  // Public ValueId metadata is logical. Plan step ports address physical
  // instances, because one logical value may be materialized in several
  // storage domains after boundary-copy insertion.
  TensorDescriptions descriptions;
  TensorDescriptions instance_descriptions;
  ValueOrdinals instance_values;
  ValueOrdinals instance_storage;
  BindingRoles instance_roles;
  ValueOrdinals external_instances;
  ValueOrdinals external_storage;
  ValueOrdinals node_inputs;
  ValueOrdinals node_outputs;
  ValueOrdinals workspace_requests;
  ValueOrdinals step_workspace_requests;
  // LifetimePlanner requires dense resource slots while storage-domain
  // ordinals may be sparse among values that actually need workspace.
  ValueOrdinals storage_workspace_slots;
  container::SmallVector<mem::MemoryResourceRef, 2, mem::AllocatorRef>
      workspace_resources;
  // Dense LifetimePlanner resource slot -> public storage-domain ordinal.
  ValueOrdinals workspace_storage;
  ValueOrdinals node_steps;
  BindingRoles value_materialized;
  // For an operation output, identifies the input value selected for exact
  // in-place storage. External pairs constrain bindings; internal pairs become
  // LifetimeRequest aliases.
  ValueOrdinals alias_sources;
  container::SmallVector<std::size_t, 16, mem::AllocatorRef> alias_offsets;
  BindingRoles binding_roles;
  mem::LifetimePlan workspace_plan{};
  std::size_t workspace_bytes = 0;
  std::size_t workspace_backing_count = 0;
  std::size_t logical_node_count = 0;
  std::size_t region_count = 0;
  std::size_t required_queue_count = 0;
  std::size_t serialization_edge_count = 0;
  WorkspaceReusePolicy workspace_reuse =
      WorkspaceReusePolicy::kExistingOrderOnly;
  bool supports_concurrent_invocations = true;
  std::atomic<bool> exclusive_in_flight{false};
  SpecializationSignature specialization{};
};

inline void retain(PlanState *state) noexcept {
  if (state != nullptr)
    state->incref();
}

inline void release(PlanState *state) noexcept {
  if (state != nullptr)
    state->decref();
}

struct ReportState final : util::RefCounted<ReportState> {
  explicit ReportState(mem::AllocatorRef allocator_value) noexcept
      : allocator(allocator_value), steps(allocator_value),
        targets(allocator_value), storage_domains(allocator_value),
        lanes(allocator_value), segments(allocator_value),
        physical_values(allocator_value), step_inputs(allocator_value),
        step_outputs(allocator_value), segment_dependencies(allocator_value) {}

  void destroy() noexcept {
    const mem::AllocatorRef owner = allocator;
    this->~ReportState();
    owner.deallocate(this, sizeof(ReportState), alignof(ReportState));
  }

  mem::AllocatorRef allocator{};
  CompileStepReports steps;
  CompileTargetReports targets;
  CompileStorageReports storage_domains;
  CompileLaneReports lanes;
  CompileSegmentReports segments;
  CompilePhysicalValueReports physical_values;
  ValueOrdinals step_inputs;
  ValueOrdinals step_outputs;
  ValueOrdinals segment_dependencies;
  compute::BackendId backend{};
  compute::ProviderIdentity provider{};
  std::size_t workspace_bytes = 0;
  std::size_t workspace_backing_count = 0;
  std::size_t node_count = 0;
  std::size_t segment_count = 0;
  std::size_t segment_dependency_count = 0;
  std::size_t command_segment_count = 0;
  std::size_t capturable_segment_count = 0;
  std::size_t region_count = 0;
  std::size_t fallback_operation_count = 0;
  std::size_t constant_count = 0;
  std::size_t borrowed_constant_count = 0;
  std::size_t packed_constant_count = 0;
  std::size_t packed_constant_bytes = 0;
  std::size_t invocation_owned_output_count = 0;
  std::size_t copy_bytes = 0;
  std::uint32_t synchronization_count = 0;
  bool supports_concurrent_invocations = true;
  bool mixed_providers = false;
  bool mixed_backends = false;
  SpecializationSignature specialization{};
};

inline void retain(ReportState *state) noexcept {
  if (state != nullptr)
    state->incref();
}

inline void release(ReportState *state) noexcept {
  if (state != nullptr)
    state->decref();
}

struct AggregateCompletionState;

struct AggregateChildState final {
  AggregateCompletionState *aggregate = nullptr;
  compute::detail::CompletionLifetime lifetime{};
};

using AggregateChildStates =
    container::SmallVector<AggregateChildState, 4, mem::AllocatorRef>;
using ReplayDependencyRoutes =
    container::SmallVector<ReplayDependencyRoute, 4, mem::AllocatorRef>;

// Preallocated mutable state for a public Completion that represents several
// independent terminal Queue segments. It is activated only while an
// Invocation is in flight; no metadata allocation occurs in submit().
struct AggregateCompletionState final {
  compute::Completion *completions = nullptr;
  const std::uint32_t *completion_segments = nullptr;
  AggregateChildState *children = nullptr;
  std::size_t completion_count = 0;
  std::atomic<std::size_t> remaining{0};
  util::SpinLock lock{};
  Status terminal{};
  compute::detail::CompletionLifetime outer_lifetime{};
};

struct InvocationState final : util::RefCounted<InvocationState> {
  InvocationState(PlanState *plan_value,
                  mem::PlannedBuffers workspace_value) noexcept
      : allocator(plan_value->allocator), plan(plan_value), readable(allocator),
        writable(allocator), bound(allocator),
        workspace(std::move(workspace_value)), invocations(allocator),
        submission_inputs(allocator), submission_outputs(allocator),
        queues(allocator), queue_seen(allocator),
        segment_completions(allocator), captured_segments(allocator),
        dependency_routes(allocator), candidate_dependency_routes(allocator),
        aggregate_segments(allocator), aggregate_children(allocator) {
    retain(plan);
  }

  void destroy() noexcept {
    const mem::AllocatorRef owner = allocator;
    PlanState *retained_plan = plan;
    this->~InvocationState();
    release(retained_plan);
    owner.deallocate(this, sizeof(InvocationState), alignof(InvocationState));
  }

  mem::AllocatorRef allocator{};
  PlanState *plan = nullptr;
  ReadableBindings readable;
  WritableBindings writable;
  BindingRoles bound;
  // Provider invocation state may borrow its assigned view until destruction,
  // so backing storage must be declared first and therefore destroyed last.
  mem::PlannedBuffers workspace{};
  StepInvocations invocations;
  ReadableBindings submission_inputs;
  WritableBindings submission_outputs;
  container::SmallVector<compute::Queue *, 2, mem::AllocatorRef> queues;
  BindingRoles queue_seen;
  container::SmallVector<compute::Completion, 4, mem::AllocatorRef>
      segment_completions;
  container::SmallVector<compute::detail::CapturedCommands, 4,
                         mem::AllocatorRef>
      captured_segments;
  // The active vector is immutable throughout every warm submit. Preparation
  // writes a preallocated candidate vector and commits it only after all Queue,
  // transfer and completion-capacity checks succeed.
  ReplayDependencyRoutes dependency_routes;
  ReplayDependencyRoutes candidate_dependency_routes;
  ValueOrdinals aggregate_segments;
  AggregateChildStates aggregate_children;
  AggregateCompletionState aggregate_completion{};
  compute::detail::CompletionLifetime completion_lifetime{};
  // Only one run may use mutable bindings/provider invocation state at a time.
  // A Completion lifetime attachment returns this flag to idle at the true
  // terminal point and retains the state if the public Invocation is released.
  std::atomic<bool> in_flight{false};
  std::atomic<bool> outputs_ready{false};
  ReplayRoute replay_route = ReplayRoute::kUnprepared;
  ReplayReport replay_report{};
  bool replay_prepared = false;
  bool bindings_prepared = false;
  bool transfers_prepared = false;
  bool dependencies_prepared = false;
};

inline void retain(InvocationState *state) noexcept {
  if (state != nullptr)
    state->incref();
}

inline void release(InvocationState *state) noexcept {
  if (state != nullptr)
    state->decref();
}

struct ExecutableAccess final {
  static Executable adopt(PlanState *state) noexcept {
    return Executable(state);
  }
};

struct CompileReportAccess final {
  static CompileReport adopt(ReportState *state) noexcept {
    return CompileReport(state);
  }
};

struct InvocationAccess final {
  static Invocation adopt(InvocationState *state) noexcept {
    return Invocation(state);
  }
};

struct SpecializationSignatureAccess final {
  static SpecializationSignature adopt(SignatureState *state) noexcept {
    return SpecializationSignature(state);
  }
  static const SignatureState *
  state(const SpecializationSignature &signature) noexcept {
    return signature.state_;
  }
};

} // namespace hyle::graph::detail
