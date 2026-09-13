#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/provider_extension.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/container/small_vector.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/region_provider.h"
#include "hyle/mem/lifetime_plan.h"
#include "hyle/util/ref_counted.h"

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
};

struct PlanStep final {
  PlanStep(compute::PreparedOperation prepared_value,
           std::uint32_t node_begin_value, std::uint32_t input_begin_value,
           std::uint32_t input_count_value, std::uint32_t output_begin_value,
           std::uint32_t output_count_value, bool materializes_value,
           CompileFallbackReason fallback_reason_value) noexcept
      : operation(std::move(prepared_value)), node_begin(node_begin_value),
        input_begin(input_begin_value), output_begin(output_begin_value),
        input_count(input_count_value), output_count(output_count_value),
        materializes(materializes_value),
        fallback_reason(fallback_reason_value) {}

  PlanStep(PreparedRegion region_value, std::uint32_t node_begin_value,
           std::uint32_t node_count_value, std::uint32_t input_begin_value,
           std::uint32_t input_count_value, std::uint32_t output_begin_value,
           std::uint32_t output_count_value) noexcept
      : region(std::move(region_value)), node_begin(node_begin_value),
        node_count(node_count_value), input_begin(input_begin_value),
        output_begin(output_begin_value), input_count(input_count_value),
        output_count(output_count_value), kind(PlanStepKind::kRegion) {}

  PlanStep(const PlanStep &) = delete;
  PlanStep &operator=(const PlanStep &) = delete;
  PlanStep(PlanStep &&) noexcept = default;
  PlanStep &operator=(PlanStep &&) noexcept = default;

  compute::ProviderIdentity provider() const noexcept {
    return kind == PlanStepKind::kRegion ? region.provider()
                                         : operation.provider();
  }
  compute::WorkspaceRequirement workspace_requirement() const noexcept {
    return kind == PlanStepKind::kRegion ? region.properties().workspace
                                         : operation.workspace_requirement();
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
};

using PlanSteps = container::SmallVector<PlanStep, 8, mem::AllocatorRef>;
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

struct PlanState final : util::RefCounted<PlanState> {
  PlanState(mem::AllocatorRef allocator_value,
            std::uint64_t generation_value) noexcept
      : allocator(allocator_value), generation(generation_value),
        steps(allocator_value), input_ids(allocator_value),
        constant_ids(allocator_value), output_ids(allocator_value),
        constant_bindings(allocator_value), descriptions(allocator_value),
        node_inputs(allocator_value), node_outputs(allocator_value),
        workspace_requests(allocator_value),
        step_workspace_requests(allocator_value), node_steps(allocator_value),
        value_materialized(allocator_value), alias_sources(allocator_value),
        alias_offsets(allocator_value), binding_roles(allocator_value) {}

  void destroy() noexcept {
    const mem::AllocatorRef owner = allocator;
    this->~PlanState();
    owner.deallocate(this, sizeof(PlanState), alignof(PlanState));
  }

  mem::AllocatorRef allocator{};
  std::uint64_t generation = 0;
  PlanSteps steps;
  ValueIds input_ids;
  ValueIds constant_ids;
  ValueIds output_ids;
  ReadableBindings constant_bindings;
  TensorDescriptions descriptions;
  ValueOrdinals node_inputs;
  ValueOrdinals node_outputs;
  ValueOrdinals workspace_requests;
  ValueOrdinals step_workspace_requests;
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
      : allocator(allocator_value), steps(allocator_value) {}

  void destroy() noexcept {
    const mem::AllocatorRef owner = allocator;
    this->~ReportState();
    owner.deallocate(this, sizeof(ReportState), alignof(ReportState));
  }

  mem::AllocatorRef allocator{};
  CompileStepReports steps;
  compute::BackendId backend{};
  compute::ProviderIdentity provider{};
  std::size_t workspace_bytes = 0;
  std::size_t workspace_backing_count = 0;
  std::size_t node_count = 0;
  std::size_t region_count = 0;
  std::size_t fallback_operation_count = 0;
  std::size_t constant_count = 0;
  std::size_t borrowed_constant_count = 0;
  std::size_t packed_constant_count = 0;
  std::size_t packed_constant_bytes = 0;
  std::size_t invocation_owned_output_count = 0;
  std::size_t copy_bytes = 0;
  std::uint32_t synchronization_count = 0;
  bool mixed_providers = false;
};

inline void retain(ReportState *state) noexcept {
  if (state != nullptr)
    state->incref();
}

inline void release(ReportState *state) noexcept {
  if (state != nullptr)
    state->decref();
}

struct InvocationState final : util::RefCounted<InvocationState> {
  InvocationState(PlanState *plan_value,
                  mem::PlannedBuffers workspace_value) noexcept
      : allocator(plan_value->allocator), plan(plan_value), readable(allocator),
        writable(allocator), bound(allocator),
        workspace(std::move(workspace_value)), invocations(allocator),
        submission_inputs(allocator), submission_outputs(allocator) {
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
  compute::detail::CompletionLifetime completion_lifetime{};
  // Only one run may use mutable bindings/provider invocation state at a time.
  // A Completion lifetime attachment returns this flag to idle at the true
  // terminal point and retains the state if the public Invocation is released.
  std::atomic<bool> in_flight{false};
  std::atomic<bool> outputs_ready{false};
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

} // namespace hyle::graph::detail
