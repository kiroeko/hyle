#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"
#include "hyle/compute/operation.h"
#include "hyle/compute/provider.h"
#include "hyle/compute/tensor.h"
#include "hyle/compute/visibility.h"
#include "hyle/compute/workload.h"
#include "hyle/error/status.h"

namespace hyle::compute {

namespace detail {
struct InvocationState;
struct OperationPreparationAccess;
struct OperationInvocationAccess;
struct PreparedState;
} // namespace detail

struct WorkspaceRequirement final {
  std::size_t size = 0;
  std::size_t alignment = 1;
};

// Upper bounds outside the supplied compilation allocator. Includes private
// packing/code and prepare/failure scratch, but not caller constants,
// Context/Queue/driver control overhead or per-invocation storage. Resident
// bounds the retained artifact through final destruction (including teardown
// scratch); peak additionally covers prepare and partial-failure cleanup.
// Peak includes resident; unknown is not zero. Declared before prepare.
struct ProviderCompilationStorage final {
  std::optional<std::size_t> resident_bytes{};
  std::optional<std::size_t> peak_bytes{};

  constexpr bool valid() const noexcept {
    return !resident_bytes || !peak_bytes || *resident_bytes <= *peak_bytes;
  }
};

// Cold structural facts at one provider's canonical operation/region
// boundary. Disengaged means unknown; engaged zero is an exact zero, never a
// timing score. Runtime ownership/replay/threading remain separate facts.
struct ProviderRouteCostFacts final {
  std::optional<WorkspaceRequirement> workspace{};
  std::optional<std::size_t> persistent_bytes{};
  std::optional<std::size_t> copy_bytes{};
  std::optional<std::size_t> repack_bytes{};
  std::optional<std::uint32_t> synchronization_count{};
  // Backend execution episodes, not workers, tasks or vector chunks. A
  // materialized provider route must declare at least one episode if known.
  std::optional<std::uint32_t> dispatch_count{};
  ProviderCompilationStorage compilation_storage{};

  constexpr bool valid() const noexcept {
    return (!workspace || std::has_single_bit(workspace->alignment)) &&
           (!dispatch_count || *dispatch_count != 0) &&
           compilation_storage.valid();
  }
};

enum class ProviderPrepareOutcomeKind : std::uint8_t {
  kPrepared = 0,
  kDeclinedUnsupported,
  kDeclinedResourceBudget,
  kError,
};

// Shared operation/region cold-prepare result. A decline promises no accepted
// user work or external side effects; the common owner must first destroy any
// returned partial artifact before another route is considered. An arbitrary
// Error Status (including Unimplemented/OOM) never grants permission to retry.
struct ProviderPrepareOutcome final {
  ProviderPrepareOutcomeKind kind = ProviderPrepareOutcomeKind::kError;
  Status status = Status::Internal("provider prepare outcome is unset");

  static ProviderPrepareOutcome prepared() noexcept {
    return {ProviderPrepareOutcomeKind::kPrepared, Status::Ok()};
  }
  static ProviderPrepareOutcome unsupported(Status reason) noexcept {
    return {ProviderPrepareOutcomeKind::kDeclinedUnsupported,
            std::move(reason)};
  }
  static ProviderPrepareOutcome resource_declined(Status reason) noexcept {
    return {ProviderPrepareOutcomeKind::kDeclinedResourceBudget,
            std::move(reason)};
  }
  static ProviderPrepareOutcome error(Status reason) noexcept {
    return {ProviderPrepareOutcomeKind::kError, std::move(reason)};
  }

  bool valid() const noexcept {
    switch (kind) {
    case ProviderPrepareOutcomeKind::kPrepared:
      return status.ok();
    case ProviderPrepareOutcomeKind::kDeclinedUnsupported:
      return status.IsUnimplemented();
    case ProviderPrepareOutcomeKind::kDeclinedResourceBudget:
      return status.IsResourceExhausted();
    case ProviderPrepareOutcomeKind::kError:
      return !status.ok();
    }
    return false;
  }
};

// Exact MemoryResource selection for one prepared route. The spans are
// borrowed only for prepare(); PreparedOperation copies their identities.
// An empty workspace_resource selects the first output resource, then the
// first input resource. It is only valid when the selected provider uses
// caller-owned workspace; provider-owned workspace ignores this field.
struct OperationResourceBindings final {
  std::span<const mem::MemoryResourceRef> inputs{};
  std::span<const mem::MemoryResourceRef> outputs{};
  mem::MemoryResourceRef workspace_resource{};
};

// Provider-specific refinement of the broad alias permission declared by the
// operation schema. byte_offset is relative to the selected input storage;
// optional in-place aliases use zero, while a non-materializing view may name
// a proper subrange. Querying this is cold planning work and adds no submit
// dispatch.
struct AliasSupport final {
  bool supported = false;
  std::size_t byte_offset = 0;
};

class PreparedOperation;
class OperationInvocation;

// Descriptor preparation is the executable capability check for the selected
// device, dtype/shape and resources. It may compile/allocate program state but
// does not read or execute tensor payloads, and needs no Queue or tensor
// storage. Retain the result for execution; schema inference alone proves no
// support.

HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Operation operation,
        std::span<const TensorDesc> inputs,
        std::span<const TensorDesc> outputs,
        OperationResourceBindings resources,
        const WorkloadContext &workload = {}) noexcept;

// Single-resource convenience for CPU/unified-memory work. Every binding and
// caller-owned workspace is fixed to `resource` during the cold prepare.
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Operation operation,
        std::span<const TensorDesc> inputs,
        std::span<const TensorDesc> outputs,
        mem::MemoryResourceRef resource,
        const WorkloadContext &workload = {}) noexcept;

// Natural direct-use route: derive exact descriptions and resources from the
// supplied views. The views remain caller-owned and are not retained.
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Operation operation,
        std::span<const ConstTensorView> inputs,
        std::span<const TensorView> outputs,
        mem::MemoryResourceRef workspace_resource = {},
        const WorkloadContext &workload = {}) noexcept;

class PreparedOperation final {
public:
  PreparedOperation() noexcept = default;
  HYLE_COMPUTE_API PreparedOperation(const PreparedOperation &other) noexcept;
  HYLE_COMPUTE_API PreparedOperation &
  operator=(const PreparedOperation &other) noexcept;
  HYLE_COMPUTE_API PreparedOperation(PreparedOperation &&other) noexcept;
  HYLE_COMPUTE_API PreparedOperation &
  operator=(PreparedOperation &&other) noexcept;
  HYLE_COMPUTE_API ~PreparedOperation();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  // Immutable facts bound during preparation; this cold query retains their owner.
  HYLE_COMPUTE_API WorkloadContext workload() const noexcept;
  HYLE_COMPUTE_API OperationKind kind() const noexcept;
  HYLE_COMPUTE_API ProviderIdentity provider() const noexcept;
  HYLE_COMPUTE_API ProviderInstanceId provider_instance_id() const noexcept;
  HYLE_COMPUTE_API ProviderRuntimeProperties runtime_properties() const noexcept;
  HYLE_COMPUTE_API WorkspaceRequirement workspace_requirement() const noexcept;
  HYLE_COMPUTE_API std::optional<std::size_t> persistent_bytes() const noexcept;
  // Per-invocation upper bound outside the supplied metadata allocator and
  // caller workspace, including cold domain instantiation and all subsequent
  // create/replay/teardown. A domain artifact covers every accepted shape.
  // nullopt is unknown; zero promises no such additional private storage.
  HYLE_COMPUTE_API std::optional<std::size_t>
  invocation_private_bytes() const noexcept;
  HYLE_COMPUTE_API bool capture_compatible() const noexcept;
  HYLE_COMPUTE_API std::size_t input_count() const noexcept;
  HYLE_COMPUTE_API std::size_t output_count() const noexcept;
  HYLE_COMPUTE_API Result<const TensorDesc &>
  input_description(std::size_t index = 0) const noexcept;
  HYLE_COMPUTE_API Result<const TensorDesc &>
  output_description(std::size_t index = 0) const noexcept;
  HYLE_COMPUTE_API Result<AliasSupport>
  alias_support(std::size_t output_index,
                std::size_t input_index) const noexcept;

  HYLE_COMPUTE_API Result<OperationInvocation>
  create_invocation(mem::BufferView workspace = {}) const noexcept;

private:
  friend class PreparedDelegatedOperation;
  friend HYLE_COMPUTE_API Result<PreparedOperation>
  prepare(const Context &, Operation, std::span<const TensorDesc>,
          std::span<const TensorDesc>, OperationResourceBindings,
          const WorkloadContext &) noexcept;
  friend struct detail::OperationPreparationAccess;

  explicit PreparedOperation(detail::PreparedState *state) noexcept
      : state_(state) {}

  detail::PreparedState *state_ = nullptr;
};

// Exclusive mutable state for one execution lane. Create it once per
// concurrent run, then rebind and submit repeatedly without provider search,
// descriptor construction or workspace allocation. A pending Completion
// retains the provider invocation even if this handle is released; the caller
// still owns tensor payload, external workspace and Queue lifetimes.
// If runtime_properties().supports_concurrent_invocations is false, direct
// callers must serialize every Invocation of that PreparedOperation through
// terminal Completion. Direct submit does not add a shared artifact lock.
// An accepted run that ends in failure retires this invocation: in_flight()
// becomes false, but subsequent submit fails. Create a fresh invocation;
// provider/device recovery may additionally require rebuilding its owners.
// A rejection before acceptance leaves the invocation reusable.
class OperationInvocation final {
public:
  OperationInvocation() noexcept = default;
  OperationInvocation(const OperationInvocation &) = delete;
  OperationInvocation &operator=(const OperationInvocation &) = delete;
  HYLE_COMPUTE_API OperationInvocation(OperationInvocation &&other) noexcept;
  HYLE_COMPUTE_API OperationInvocation &
  operator=(OperationInvocation &&other) noexcept;
  HYLE_COMPUTE_API ~OperationInvocation();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_API ProviderIdentity provider() const noexcept;
  HYLE_COMPUTE_API ProviderInstanceId provider_instance_id() const noexcept;
  HYLE_COMPUTE_API bool in_flight() const noexcept;

  HYLE_COMPUTE_API Result<Completion>
  submit(Queue &queue, std::span<const ConstTensorView> inputs,
         std::span<const TensorView> outputs) noexcept;
  HYLE_COMPUTE_API Result<Completion>
  submit(Queue &queue, std::span<const ConstTensorView> inputs,
         const TensorView &output) noexcept;
  HYLE_COMPUTE_API Result<Completion> submit(Queue &queue,
                                             const TensorView &output) noexcept;
  HYLE_COMPUTE_API Result<Completion> submit(Queue &queue,
                                             const ConstTensorView &input,
                                             const TensorView &output) noexcept;
  HYLE_COMPUTE_API Result<Completion> submit(Queue &queue,
                                             const ConstTensorView &input,
                                             const ConstTensorView &weight,
                                             const TensorView &output) noexcept;
  HYLE_COMPUTE_API Result<Completion> submit(Queue &queue,
                                             const ConstTensorView &first,
                                             const ConstTensorView &second,
                                             const ConstTensorView &third,
                                             const TensorView &output) noexcept;
  HYLE_COMPUTE_API Result<Completion>
  submit(Queue &queue, const ConstTensorView &first,
         const ConstTensorView &second, const ConstTensorView &third,
         const ConstTensorView &fourth, const TensorView &output) noexcept;
  HYLE_COMPUTE_API Result<Completion>
  submit(Queue &queue, const ConstTensorView &input,
         const TensorView &first_output,
         const TensorView &second_output) noexcept;

private:
  friend class DelegatedInvocation;
  friend class PreparedOperation;
  friend struct detail::OperationPreparationAccess;
  friend struct detail::OperationInvocationAccess;
  explicit OperationInvocation(detail::InvocationState *state) noexcept
      : state_(state) {}

  detail::InvocationState *state_ = nullptr;
};

HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, View operation, TensorDesc input,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Copy operation, TensorDesc input,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Fill operation,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Cast operation, TensorDesc input,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Unary operation, TensorDesc input,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Binary operation, TensorDesc left,
        TensorDesc right,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Select operation, TensorDesc condition,
        TensorDesc when_true, TensorDesc when_false,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Reduce operation, TensorDesc input,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;

HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Scan operation, TensorDesc input,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, LayerNorm operation, TensorDesc input,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, LayerNorm operation, TensorDesc input,
        TensorDesc scale,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, LayerNorm operation, TensorDesc input,
        TensorDesc scale, TensorDesc bias,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, RmsNorm operation, TensorDesc input,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, RmsNorm operation, TensorDesc input,
        TensorDesc weight,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, MatMul operation, TensorDesc left,
        TensorDesc right,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, MatMul operation, TensorDesc left,
        TensorDesc right, TensorDesc output,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, GatedActivation operation, TensorDesc input,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, GatedActivation operation, TensorDesc gate,
        TensorDesc value,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Gather operation, TensorDesc source,
        TensorDesc indices,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Scatter operation, TensorDesc input,
        TensorDesc indices, TensorDesc updates,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Concat operation,
        std::span<const TensorDesc> inputs,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, RoPE operation, TensorDesc input,
        TensorDesc positions, TensorDesc frequencies,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Softmax operation, TensorDesc input,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Softmax operation, TensorDesc input,
        TensorDesc mask,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, TopK operation, TensorDesc input,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, ScaledDotProductAttention operation,
        TensorDesc query, TensorDesc key, TensorDesc value,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, ScaledDotProductAttention operation,
        TensorDesc query, TensorDesc key, TensorDesc value, TensorDesc mask,
        mem::MemoryResourceRef resource = mem::system_resource(),
        const WorkloadContext &workload = {}) noexcept;

} // namespace hyle::compute
