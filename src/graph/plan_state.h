#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "constant_artifact.h"
#include "graph_state.h"
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
#include "storage_budget.h"

namespace hyle::graph::detail {

enum class InvocationPhase : std::uint8_t { kIdle, kRunning, kFailed };
static_assert(sizeof(std::atomic<InvocationPhase>) ==
              sizeof(std::atomic<bool>));
static_assert(alignof(std::atomic<InvocationPhase>) ==
              alignof(std::atomic<bool>));

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
      : prepared_(other.prepared_), state_(other.state_),
        allocator_(other.allocator_) {
    other.prepared_ = nullptr;
    other.state_ = nullptr;
  }
  RegionInvocation &operator=(RegionInvocation &&other) noexcept;
  ~RegionInvocation();

  bool valid() const noexcept { return prepared_ != nullptr; }
  compute::detail::EnqueueResult
  enqueue_prevalidated(compute::detail::QueueSequence &sequence,
                       std::span<const compute::ConstTensorView> inputs,
                       std::span<const compute::TensorView> outputs) noexcept;
  compute::detail::CompletionEndpoint delegated_endpoint() const noexcept;
  bool can_import_endpoint(
      compute::detail::CompletionEndpoint source) const noexcept;
  Result<compute::Completion> submit_delegated(
      std::span<const compute::detail::CompletionDependency> dependencies,
      std::span<const compute::ConstTensorView> inputs,
      std::span<const compute::TensorView> outputs) noexcept;

private:
  void reset() noexcept;
  friend class PreparedRegion;
  RegionInvocation(const PreparedRegion *prepared, void *state,
                   const mem::AllocatorRef &allocator) noexcept
      : prepared_(prepared), state_(state), allocator_(&allocator) {}

  const PreparedRegion *prepared_ = nullptr;
  void *state_ = nullptr;
  // Borrows the enclosing Graph Invocation's stable allocator member.
  const mem::AllocatorRef *allocator_ = nullptr;
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
  compute::ProviderInstanceId provider_instance_id() const noexcept {
    return extension_.provider_instance_id();
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
  create_invocation(mem::BufferView workspace,
                    const mem::AllocatorRef &allocator) const noexcept;
  compute::detail::EnqueueResult
  enqueue_prevalidated(
      void *invocation, compute::detail::QueueSequence &sequence,
      std::span<const compute::ConstTensorView> inputs,
      std::span<const compute::TensorView> outputs) const noexcept;
  compute::detail::CompletionEndpoint
  delegated_endpoint(void *invocation) const noexcept;
  bool can_import_endpoint(
      void *invocation,
      compute::detail::CompletionEndpoint source) const noexcept;
  Result<compute::Completion> submit_delegated(
      void *invocation,
      std::span<const compute::detail::CompletionDependency> dependencies,
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
    allocator_ = other.allocator_;
    other.prepared_ = nullptr;
    other.state_ = nullptr;
  }
  return *this;
}

inline RegionInvocation::~RegionInvocation() { reset(); }

inline void RegionInvocation::reset() noexcept {
  if (prepared_ != nullptr && prepared_->valid() &&
      prepared_->prepared_.vtable->destroy_invocation != nullptr) {
    prepared_->prepared_.vtable->destroy_invocation(state_, *allocator_);
  }
  prepared_ = nullptr;
  state_ = nullptr;
}

inline compute::detail::EnqueueResult RegionInvocation::enqueue_prevalidated(
    compute::detail::QueueSequence &sequence,
    std::span<const compute::ConstTensorView> inputs,
    std::span<const compute::TensorView> outputs) noexcept {
  if (!valid()) {
    return compute::detail::EnqueueResult::rejected(
        Status::FailedPrecondition("graph region invocation is invalid"));
  }
  return prepared_->enqueue_prevalidated(state_, sequence, inputs, outputs);
}

inline compute::detail::CompletionEndpoint
RegionInvocation::delegated_endpoint() const noexcept {
  return valid() ? prepared_->delegated_endpoint(state_)
                 : compute::detail::CompletionEndpoint{};
}

inline bool RegionInvocation::can_import_endpoint(
    compute::detail::CompletionEndpoint source) const noexcept {
  return valid() && prepared_->can_import_endpoint(state_, source);
}

inline Result<compute::Completion> RegionInvocation::submit_delegated(
    std::span<const compute::detail::CompletionDependency> dependencies,
    std::span<const compute::ConstTensorView> inputs,
    std::span<const compute::TensorView> outputs) noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("graph region invocation is invalid"));
  }
  return prepared_->submit_delegated(state_, dependencies, inputs, outputs);
}

inline Result<RegionInvocation> PreparedRegion::create_invocation(
    mem::BufferView workspace,
    const mem::AllocatorRef &allocator) const noexcept {
  if (!valid() || prepared_.vtable->create_invocation == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compiled graph region is invalid"));
  }
  struct Episode final {
    const PreparedRegion *region = nullptr;
    mem::BufferView workspace{};
    mem::AllocatorRef allocator{};
    void *state = nullptr;
  } episode{this, workspace, allocator, nullptr};
  const Status status = compute::detail::ProviderExtensionAccess::call(
      extension_,
      +[](void *, const void *, void *opaque) noexcept -> Status {
        auto &call = *static_cast<Episode *>(opaque);
        return call.region->prepared_.vtable->create_invocation(
            call.region->prepared_.state, call.allocator, call.workspace,
            &call.state);
      },
      &episode);
  if (!status.ok()) {
    // A failing provider is still responsible for returning either nullptr or
    // a destroyable partial invocation. Cleaning the latter here makes OOM and
    // injected-failure rollback symmetric with prepared-region compilation.
    if (episode.state != nullptr &&
        prepared_.vtable->destroy_invocation != nullptr) {
      prepared_.vtable->destroy_invocation(episode.state, allocator);
    }
    return std::unexpected(status);
  }
  return RegionInvocation(this, episode.state, allocator);
}

inline compute::detail::EnqueueResult PreparedRegion::enqueue_prevalidated(
    void *invocation, compute::detail::QueueSequence &sequence,
    std::span<const compute::ConstTensorView> inputs,
    std::span<const compute::TensorView> outputs) const noexcept {
  if (!valid() || !sequence.valid() ||
      prepared_.properties.runtime.mode !=
          compute::ProviderRuntimeMode::kSharedQueue ||
      prepared_.vtable->shared_submit == nullptr) {
    return compute::detail::EnqueueResult::rejected(
        Status::Internal("prevalidated graph region route is invalid"));
  }
  // Compile/bind established the exact port resources and prepare_replay()
  // cached these views. Both replay preparation and submit check the lane's
  // current Queue owner; the sequence retains Context/Queue admission. The
  // extension still owns the provider lifetime, so replay needs only the
  // already-bound region call, not another per-port resource validation.
  return prepared_.vtable->shared_submit(prepared_.state, invocation,
                                        sequence.queue(), inputs, outputs);
}

inline compute::detail::CompletionEndpoint
PreparedRegion::delegated_endpoint(void *invocation) const noexcept {
  if (!valid() ||
      prepared_.properties.runtime.mode !=
          compute::ProviderRuntimeMode::kDelegated ||
      prepared_.vtable->endpoint == nullptr) {
    return {};
  }
  struct Episode final {
    const PreparedRegion *region = nullptr;
    void *invocation = nullptr;
    compute::detail::CompletionEndpoint endpoint{};
  } episode{this, invocation, {}};
  const Status status = compute::detail::ProviderExtensionAccess::call(
      extension_,
      +[](void *, const void *, void *opaque) noexcept -> Status {
        auto &call = *static_cast<Episode *>(opaque);
        call.endpoint = call.region->prepared_.vtable->endpoint(
            call.region->prepared_.state, call.invocation);
        return Status::Ok();
      },
      &episode);
  return status.ok() ? episode.endpoint
                     : compute::detail::CompletionEndpoint{};
}

inline bool PreparedRegion::can_import_endpoint(
    void *invocation,
    compute::detail::CompletionEndpoint source) const noexcept {
  if (!source.valid() || !valid() ||
      prepared_.properties.runtime.mode !=
          compute::ProviderRuntimeMode::kDelegated ||
      prepared_.vtable->can_import_endpoint == nullptr) {
    return false;
  }
  struct Episode final {
    const PreparedRegion *region = nullptr;
    void *invocation = nullptr;
    compute::detail::CompletionEndpoint source{};
    bool accepted = false;
  } episode{this, invocation, source, false};
  const Status status = compute::detail::ProviderExtensionAccess::call(
      extension_,
      +[](void *, const void *, void *opaque) noexcept -> Status {
        auto &call = *static_cast<Episode *>(opaque);
        call.accepted = call.region->prepared_.vtable->can_import_endpoint(
            call.region->prepared_.state, call.invocation, call.source);
        return Status::Ok();
      },
      &episode);
  return status.ok() && episode.accepted;
}

inline Result<compute::Completion> PreparedRegion::submit_delegated(
    void *invocation,
    std::span<const compute::detail::CompletionDependency> dependencies,
    std::span<const compute::ConstTensorView> inputs,
    std::span<const compute::TensorView> outputs) const noexcept {
  if (!valid() ||
      prepared_.properties.runtime.mode !=
          compute::ProviderRuntimeMode::kDelegated ||
      prepared_.vtable->delegated_submit == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("graph region is not delegated"));
  }
  Result<compute::Completion> submitted = std::unexpected(
      Status::Internal("delegated graph region submission did not run"));
  struct Episode final {
    const PreparedRegion *region = nullptr;
    void *invocation = nullptr;
    std::span<const compute::detail::CompletionDependency> dependencies{};
    std::span<const compute::ConstTensorView> inputs{};
    std::span<const compute::TensorView> outputs{};
    Result<compute::Completion> *submitted = nullptr;
  } episode{this, invocation, dependencies, inputs, outputs, &submitted};
  const Status status = compute::detail::ProviderExtensionAccess::call(
      extension_,
      +[](void *, const void *, void *opaque) noexcept -> Status {
        auto &call = *static_cast<Episode *>(opaque);
        *call.submitted = call.region->prepared_.vtable->delegated_submit(
            call.region->prepared_.state, call.invocation, call.dependencies,
            call.inputs, call.outputs);
        return Status::Ok();
      },
      &episode);
  if (!status.ok())
    return std::unexpected(status);
  if (!submitted)
    return std::unexpected(std::move(submitted).error());
  if (!submitted->valid()) {
    return std::unexpected(Status::Internal(
        "delegated graph region returned an invalid completion"));
  }
  return submitted;
}

enum class PlanStepKind : std::uint8_t {
  kOperation = 0,
  kRegion,
  kCopy,
  kView,
};

struct PlanStep final {
  PlanStep() noexcept = default;
  PlanStep(
      compute::PreparedOperation prepared_value, std::uint32_t node_begin_value,
      std::uint32_t node_set_begin_value, std::uint32_t input_begin_value,
      std::uint32_t input_count_value, std::uint32_t output_begin_value,
      std::uint32_t output_count_value, bool materializes_value,
      CompileFallbackReason fallback_reason_value,
      ExecutionTargetId target_value = ExecutionTargetId{0},
      StorageDomainId workspace_storage_value = StorageDomainId{0}) noexcept
      : operation(std::move(prepared_value)), node_begin(node_begin_value),
        node_set_begin(node_set_begin_value), input_begin(input_begin_value),
        output_begin(output_begin_value), input_count(input_count_value),
        output_count(output_count_value), materializes(materializes_value),
        fallback_reason(fallback_reason_value), target(target_value),
        workspace_storage(workspace_storage_value) {}

  PlanStep(
      PreparedRegion region_value, std::uint32_t node_begin_value,
      std::uint32_t node_set_begin_value, std::uint32_t node_count_value,
      bool nodes_contiguous_value, std::uint32_t input_begin_value,
      std::uint32_t input_count_value, std::uint32_t output_begin_value,
      std::uint32_t output_count_value,
      ExecutionTargetId target_value = ExecutionTargetId{0},
      StorageDomainId workspace_storage_value = StorageDomainId{0}) noexcept
      : region(std::move(region_value)), node_begin(node_begin_value),
        node_set_begin(node_set_begin_value), node_count(node_count_value),
        input_begin(input_begin_value), output_begin(output_begin_value),
        input_count(input_count_value), output_count(output_count_value),
        kind(PlanStepKind::kRegion), nodes_contiguous(nodes_contiguous_value),
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
    result.node_count = 0;
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
  compute::ProviderInstanceId provider_instance_id() const noexcept {
    if (kind == PlanStepKind::kRegion)
      return region.provider_instance_id();
    if (kind == PlanStepKind::kOperation)
      return operation.provider_instance_id();
    return {};
  }
  compute::WorkspaceRequirement workspace_requirement() const noexcept {
    if (kind == PlanStepKind::kRegion &&
        region.properties().runtime.workspace_owner ==
            compute::ProviderWorkspaceOwner::kCaller)
      return region.properties().workspace;
    if (kind == PlanStepKind::kOperation &&
        operation.runtime_properties().workspace_owner ==
            compute::ProviderWorkspaceOwner::kCaller)
      return operation.workspace_requirement();
    return {};
  }
  compute::WorkspaceRequirement reported_workspace_requirement() const noexcept {
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
  bool emits_command() const noexcept { return kind != PlanStepKind::kView; }
  bool capture_compatible() const noexcept {
    if (!emits_command())
      return true;
    if (kind == PlanStepKind::kCopy)
      return true;
    if (kind == PlanStepKind::kRegion)
      return region.properties().runtime.capture_compatible;
    return operation.capture_compatible();
  }
  compute::ProviderRuntimeProperties runtime_properties() const noexcept {
    if (kind == PlanStepKind::kRegion) {
      return region.properties().runtime;
    }
    if (kind == PlanStepKind::kOperation)
      return operation.runtime_properties();
    return compute::ProviderRuntimeProperties{
        .mode = compute::ProviderRuntimeMode::kSharedQueue,
        .workspace_owner = compute::ProviderWorkspaceOwner::kCaller,
        .replay_class = kind == PlanStepKind::kView
                            ? compute::ProviderReplayClass::kPrepared
                            : compute::ProviderReplayClass::kUnknown,
        .capture_compatible = true,
    };
  }

  compute::PreparedOperation operation{};
  PreparedRegion region{};
  std::uint32_t node_begin = 0;
  std::uint32_t node_set_begin = 0;
  std::uint32_t node_count = 1;
  std::uint32_t input_begin = 0;
  std::uint32_t output_begin = 0;
  std::uint32_t input_count = 0;
  std::uint32_t output_count = 0;
  PlanStepKind kind = PlanStepKind::kOperation;
  bool materializes = true;
  bool nodes_contiguous = true;
  CompileFallbackReason fallback_reason = CompileFallbackReason::kNone;
  ExecutionTargetId target{};
  ExecutionLaneId lane{};
  StorageDomainId workspace_storage{};
  StorageDomainId source_storage{};
  StorageDomainId destination_storage{};
  union {
    std::size_t copy_bytes = 0;
    std::size_t view_offset;
  };
  // Stable identity across cold re-scheduling. Serialization edges name this
  // ordinal rather than the current PlanSteps position.
  std::uint32_t schedule_identity = kNoNode;
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
using NodeIds = container::SmallVector<NodeId, 8, mem::AllocatorRef>;
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
  compute::ProviderRuntimeMode runtime_mode =
      compute::ProviderRuntimeMode::kSharedQueue;
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

// Created only when a compiled family shares owned canonical constants. There
// is no Plan back-pointer: an exact child may outlive the parent table.
struct SharedConstants final : util::RefCounted<SharedConstants> {
  explicit SharedConstants(mem::AllocatorRef owner) noexcept
      : allocator(owner), charges(owner), buffers(owner) {}
  void destroy() noexcept {
    const auto owner = allocator;
    this->~SharedConstants();
    owner.deallocate(this, sizeof(SharedConstants), alignof(SharedConstants));
  }
  mem::AllocatorRef allocator;
  container::SmallVector<mem::QuotaAllocator::Reservation, 0, mem::AllocatorRef>
      charges;
  container::SmallVector<mem::Buffer, 0, mem::AllocatorRef> buffers;
};

struct PlanState final : util::RefCounted<PlanState> {
  PlanState(mem::AllocatorRef allocator_value,
            std::uint64_t generation_value) noexcept
      : allocator(allocator_value), generation(generation_value),
        storage_domains(allocator_value), targets(allocator_value),
        lanes(allocator_value), constant_charges(allocator_value),
        artifact_charges(allocator_value),
        materialized_constants(allocator_value),
        constant_artifacts(allocator_value), steps(allocator_value),
        exact_variants(allocator_value), variant_order(allocator_value),
        segments(allocator_value), segment_dependencies(allocator_value),
        serialization_edges(allocator_value), step_segments(allocator_value),
        terminal_segments(allocator_value),
        completion_segments(allocator_value),
        lane_segment_counts(allocator_value), input_ids(allocator_value),
        constant_ids(allocator_value), output_ids(allocator_value),
        constant_bindings(allocator_value), effect_bindings(allocator_value),
        descriptions(allocator_value), instance_descriptions(allocator_value),
        instance_values(allocator_value), instance_storage(allocator_value),
        instance_alignments(allocator_value),
        instance_roles(allocator_value), external_instances(allocator_value),
        external_storage(allocator_value), node_inputs(allocator_value),
        node_outputs(allocator_value), step_nodes(allocator_value),
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

  const mem::Buffer *owned_constant(std::size_t slot) const noexcept {
    const auto &buffers =
        shared_constants ? shared_constants->buffers : materialized_constants;
    return slot < buffers.size() && buffers[slot].valid() ? &buffers[slot]
                                                          : nullptr;
  }

  mem::AllocatorRef allocator{};
  std::uint64_t generation = 0;
  compute::WorkloadContext workload{};
  // Empty on the exact path. A domain retains its immutable shape program and
  // specs once; concrete Invocations share this Plan and its concurrency gate.
  Graph domain_graph{};
  // One opt-in owner shared by general/exact Plans; no mutable run data in
  // Plan.
  util::Ref<StorageBudgetState> invocation_storage;
  util::Ref<StorageBudgetState> compilation_storage;
  mem::AllocatorRef runtime_allocator() const noexcept {
    return compilation_storage ? compilation_storage->backing() : allocator;
  }
  StorageDomainRecords storage_domains;
  ExecutionTargetRecords targets;
  ExecutionLaneRecords lanes;
  // Declared before the actual owners: return charges only after their storage
  // is destroyed. Empty/no allocation on ordinary unbudgeted compilation.
  container::SmallVector<mem::QuotaAllocator::Reservation, 0, mem::AllocatorRef>
      constant_charges, artifact_charges;
  // Declared before steps: artifacts die before the constants they may borrow,
  // and buffers die before the retained Context/resource backend leases.
  // Empty on the default borrow path; otherwise one slot per logical constant.
  container::SmallVector<mem::Buffer, 0, mem::AllocatorRef>
      materialized_constants;
  util::Ref<SharedConstants> shared_constants;
  container::SmallVector<util::Ref<ConstantArtifactState>, 0, mem::AllocatorRef>
      constant_artifacts;
  // Subset already owned by an earlier selected Plan in this compiled family.
  CompiledStorageReport shared_artifact_storage{};
  PlanSteps steps;
  container::SmallVector<Executable, 0, mem::AllocatorRef> exact_variants;
  // Lexicographic signature order -> caller's declaration index. Lookup is
  // O(D log K), including adversarial hash collisions; never used by submit.
  container::SmallVector<std::uint32_t, 0, mem::AllocatorRef> variant_order;
  std::optional<std::size_t> exact_variant_index{};
  PlanSegments segments;
  ValueOrdinals segment_dependencies;
  // Packed stable step identities: source in the high word, destination in
  // the low word. These are explicit memory-for-concurrency tradeoff edges.
  container::SmallVector<std::uint64_t, 4, mem::AllocatorRef>
      serialization_edges;
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
  container::SmallVector<EffectResourceBinding, 0, mem::AllocatorRef>
      effect_bindings;
  // Public ValueId metadata is logical. Plan step ports address physical
  // instances, because one logical value may be materialized in several
  // storage domains after boundary-copy insertion.
  TensorDescriptions descriptions;
  TensorDescriptions instance_descriptions;
  ValueOrdinals instance_values;
  ValueOrdinals instance_storage;
  container::SmallVector<std::size_t, 16, mem::AllocatorRef>
      instance_alignments;
  BindingRoles instance_roles;
  ValueOrdinals external_instances;
  ValueOrdinals external_storage;
  ValueOrdinals node_inputs;
  ValueOrdinals node_outputs;
  // Canonical logical node ordinals owned by each operation/region step.
  // PlanStep::node_set_begin/node_count address this packed arena.
  ValueOrdinals step_nodes;
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
  CompileSearchReport search{};
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
  ReportState(mem::AllocatorRef allocator_value,
              std::size_t allocation_bytes_value) noexcept
      : allocator(allocator_value), allocation_bytes(allocation_bytes_value),
        nodes(allocator_value), steps(allocator_value),
        targets(allocator_value), storage_domains(allocator_value),
        lanes(allocator_value), segments(allocator_value),
        physical_values(allocator_value), constants(allocator_value),
        step_inputs(allocator_value), step_outputs(allocator_value),
        step_nodes(allocator_value), step_node_begins(allocator_value),
        segment_dependencies(allocator_value), exact_variants(allocator_value) {
  }

  void destroy() noexcept {
    const mem::AllocatorRef owner = allocator;
    const std::size_t bytes = allocation_bytes;
    this->~ReportState();
    owner.deallocate(this, bytes, alignof(ReportState));
  }

  mem::AllocatorRef allocator{};
  // Provider identity text is owned in this allocation's immutable tail.
  std::size_t allocation_bytes = 0;
  container::SmallVector<CompileNodeReport, 0, mem::AllocatorRef> nodes;
  CompileStepReports steps;
  CompileTargetReports targets;
  CompileStorageReports storage_domains;
  CompileLaneReports lanes;
  CompileSegmentReports segments;
  CompilePhysicalValueReports physical_values;
  container::SmallVector<CompileConstantReport, 0, mem::AllocatorRef> constants;
  ValueOrdinals step_inputs;
  ValueOrdinals step_outputs;
  NodeIds step_nodes;
  ValueOrdinals step_node_begins;
  ValueOrdinals segment_dependencies;
  container::SmallVector<CompileReport, 0, mem::AllocatorRef> exact_variants;
  compute::BackendId backend{};
  compute::ProviderIdentity provider{};
  std::size_t workspace_bytes = 0;
  std::size_t workspace_backing_count = 0;
  std::size_t node_count = 0;
  std::size_t segment_count = 0;
  std::size_t segment_dependency_count = 0;
  std::size_t command_segment_count = 0;
  std::size_t delegated_segment_count = 0;
  std::size_t serialization_edge_count = 0;
  std::size_t capturable_segment_count = 0;
  std::size_t region_count = 0;
  std::size_t fallback_operation_count = 0;
  std::size_t constant_count = 0;
  std::size_t borrowed_constant_count = 0;
  std::size_t materialized_constant_count = 0;
  std::size_t materialized_constant_bytes = 0;
  CompiledStorageReport compiled_storage{};
  CompiledStorageReport total_compiled_storage{};
  std::optional<CompilationStorageReport> compilation_storage{};
  std::size_t packed_constant_count = 0;
  std::size_t packed_constant_bytes = 0;
  std::size_t invocation_owned_output_count = 0;
  std::size_t copy_bytes = 0;
  std::uint32_t synchronization_count = 0;
  bool supports_concurrent_invocations = true;
  bool mixed_providers = false;
  bool mixed_backends = false;
  bool is_shape_domain = false;
  CompileSearchReport search{};
  CompileSearchReport total_search{};
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
using CompletionEndpoints =
    container::SmallVector<compute::detail::CompletionEndpoint, 4,
                           mem::AllocatorRef>;
using CompletionDependencies =
    container::SmallVector<compute::detail::CompletionDependency, 4,
                           mem::AllocatorRef>;
using ConcreteDescriptions =
    container::SmallVector<compute::TensorDesc, 0, mem::AllocatorRef>;

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
  InvocationState(PlanState *plan_value, mem::PlannedBuffers workspace_value,
                  mem::AllocatorRef allocator_value) noexcept
      : allocator(allocator_value), plan(plan_value),
        concrete_descriptions(allocator), readable(allocator),
        writable(allocator), bound(allocator),
        workspace(std::move(workspace_value)), invocations(allocator),
        submission_inputs(allocator), submission_outputs(allocator),
        queues(allocator), queue_seen(allocator),
        segment_completions(allocator), captured_segments(allocator),
        dependency_routes(allocator), candidate_dependency_routes(allocator),
        lane_endpoints(allocator), candidate_lane_endpoints(allocator),
        completion_dependencies(allocator), aggregate_segments(allocator),
        aggregate_children(allocator) {
    retain(plan);
  }

  void destroy() noexcept {
    const mem::AllocatorRef owner = allocator;
    PlanState *retained_plan = plan;
    const bool budgeted = static_cast<bool>(plan->invocation_storage);
    auto *reservation = storage_reservation();
    this->~InvocationState();
    // Provider state and workspace have actually gone before returning their
    // reservation. Keep the Plan and allocator alive through this final free.
    if (budgeted)
      reservation->~Reservation();
    owner.deallocate(this, allocation_bytes(budgeted),
                     alignof(InvocationState));
    release(retained_plan);
  }

  using Reservation = mem::QuotaAllocator::Reservation;
  static constexpr std::size_t allocation_bytes(bool budgeted) noexcept {
    return sizeof(InvocationState) + (budgeted ? sizeof(Reservation) : 0);
  }
  Reservation *storage_reservation() noexcept {
    return reinterpret_cast<Reservation *>(this + 1);
  }

  mem::AllocatorRef allocator{};
  PlanState *plan = nullptr;
  // Empty and allocation-free for exact plans. Domain instances own one
  // concrete description per logical value, not a copy of the physical plan.
  ConcreteDescriptions concrete_descriptions;
  SpecializationSignature specialization{};
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
  CompletionEndpoints lane_endpoints;
  CompletionEndpoints candidate_lane_endpoints;
  CompletionDependencies completion_dependencies;
  ValueOrdinals aggregate_segments;
  AggregateChildStates aggregate_children;
  AggregateCompletionState aggregate_completion{};
  compute::detail::CompletionLifetime completion_lifetime{};
  // Only one run may use mutable bindings/provider invocation state at a time.
  // A Completion attachment publishes idle only after success, failed after
  // failure, and retains the state if the public Invocation is released.
  std::atomic<InvocationPhase> phase{InvocationPhase::kIdle};
  std::atomic<bool> outputs_ready{false};
  ReplayRoute replay_route = ReplayRoute::kUnprepared;
  ReplayReport replay_report{};
  bool replay_prepared = false;
  bool bindings_prepared = false;
  bool transfers_prepared = false;
  bool dependencies_prepared = false;
};

static_assert(alignof(InvocationState) >=
              alignof(InvocationState::Reservation));
static_assert(sizeof(InvocationState) % alignof(InvocationState::Reservation) ==
              0);

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
  static PlanState *state(const Executable &executable) noexcept {
    return executable.state_;
  }
};

struct CompileReportAccess final {
  static CompileReport adopt(ReportState *state) noexcept {
    return CompileReport(state);
  }
  static ReportState *state(const CompileReport &report) noexcept {
    return report.state_;
  }
};

struct InvocationAccess final {
  static Invocation adopt(InvocationState *state) noexcept {
    return Invocation(state);
  }
  static InvocationState *state(const Invocation &invocation) noexcept {
    return invocation.state_;
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
