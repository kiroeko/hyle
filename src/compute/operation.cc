#include "hyle/compute/prepared_operation.h"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <utility>

#include "backend_internal.h"
#include "context_state.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "prepared_state.h"

namespace hyle::compute {
namespace {

constexpr std::size_t kInlineInferredOutputs = 4;

constexpr std::size_t align_up(std::size_t value,
                               std::size_t alignment) noexcept {
  return (value + alignment - 1) & ~(alignment - 1);
}

void mark_operation_terminal(void *opaque, const Status &) noexcept {
  static_cast<detail::InvocationState *>(opaque)->in_flight.store(
      false, std::memory_order_release);
}

void release_operation_lifetime(void *opaque) noexcept {
  detail::release(static_cast<detail::InvocationState *>(opaque));
}

Result<std::size_t>
prepared_allocation_size(std::size_t input_count,
                         std::size_t output_count) noexcept {
  constexpr std::size_t descriptor_alignment = alignof(TensorDesc);
  static_assert((descriptor_alignment & (descriptor_alignment - 1)) == 0);
  static_assert(alignof(detail::PreparedState) >= descriptor_alignment);
  const std::size_t offset =
      align_up(sizeof(detail::PreparedState), descriptor_alignment);
  if (input_count > (std::numeric_limits<std::size_t>::max)() - output_count) {
    return std::unexpected(
        Status::OutOfRange("prepared descriptor count overflows"));
  }
  const std::size_t count = input_count + output_count;
  if (count > ((std::numeric_limits<std::size_t>::max)() - offset) /
                  sizeof(TensorDesc)) {
    return std::unexpected(
        Status::OutOfRange("prepared descriptor storage overflows"));
  }
  return offset + count * sizeof(TensorDesc);
}

Status validate_inferred_outputs(Operation operation,
                                 std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> outputs,
                                 mem::AllocatorRef allocator) noexcept {
  std::array<TensorDesc, kInlineInferredOutputs> local{};
  TensorDesc *inferred = local.data();
  bool allocated = false;
  std::size_t bytes = 0;
  if (outputs.size() > local.size()) {
    if (outputs.size() >
        (std::numeric_limits<std::size_t>::max)() / sizeof(TensorDesc)) {
      return Status::OutOfRange("inferred output storage overflows");
    }
    bytes = outputs.size() * sizeof(TensorDesc);
    void *storage = allocator.allocate(bytes, alignof(TensorDesc));
    if (storage == nullptr)
      return Status::ResourceExhausted("output inference allocation failed");
    inferred = static_cast<TensorDesc *>(storage);
    for (std::size_t index = 0; index < outputs.size(); ++index)
      std::construct_at(inferred + index);
    allocated = true;
  }

  Status status = infer_output_descriptions(
      operation, inputs, std::span<TensorDesc>(inferred, outputs.size()));
  if (status.ok()) {
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      if (outputs[index] != inferred[index]) {
        status = Status::InvalidArgument(
            "operation output description does not match inferred shape");
        break;
      }
    }
  }
  if (allocated) {
    for (std::size_t index = outputs.size(); index != 0; --index)
      std::destroy_at(inferred + index - 1);
    allocator.deallocate(inferred, bytes, alignof(TensorDesc));
  }
  return status;
}

Result<detail::PreparedState *>
make_prepared_state(detail::ContextState *context, Operation operation,
                    std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs,
                    ProviderIdentity provider,
                    detail::ProviderPrepared provider_prepared) noexcept {
  auto bytes = prepared_allocation_size(inputs.size(), outputs.size());
  if (!bytes) {
    provider_prepared.vtable->destroy_prepared(provider_prepared.state,
                                               context->allocator);
    return std::unexpected(std::move(bytes).error());
  }
  void *storage =
      context->allocator.allocate(*bytes, alignof(detail::PreparedState));
  if (storage == nullptr) {
    provider_prepared.vtable->destroy_prepared(provider_prepared.state,
                                               context->allocator);
    return std::unexpected(
        Status::ResourceExhausted("prepared operation allocation failed"));
  }
  constexpr std::size_t descriptor_offset =
      align_up(sizeof(detail::PreparedState), alignof(TensorDesc));
  auto *state = ::new (storage)
      detail::PreparedState(context, operation, provider, *bytes,
                            descriptor_offset, inputs.size(), outputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index)
    std::construct_at(state->inputs() + index, inputs[index]);
  for (std::size_t index = 0; index < outputs.size(); ++index)
    std::construct_at(state->outputs() + index, outputs[index]);
  state->provider_prepared = provider_prepared;
  return state;
}

struct ByteRange final {
  mem::MemoryResourceRef resource{};
  void *handle = nullptr;
  std::size_t offset = 0;
  std::size_t size = 0;
};

ByteRange byte_range(mem::ConstBufferView view) noexcept {
  return {view.resource(), view.native_handle(), view.offset(), view.size()};
}
ByteRange byte_range(mem::BufferView view) noexcept {
  return {view.resource(), view.native_handle(), view.offset(), view.size()};
}
bool same_allocation(const ByteRange &left, const ByteRange &right) noexcept {
  return left.resource == right.resource && left.handle == right.handle;
}
bool equal_range(const ByteRange &left, const ByteRange &right) noexcept {
  return same_allocation(left, right) && left.offset == right.offset &&
         left.size == right.size;
}
bool contains(const ByteRange &outer, const ByteRange &inner) noexcept {
  return same_allocation(outer, inner) && inner.offset >= outer.offset &&
         inner.offset - outer.offset <= outer.size &&
         inner.size <= outer.size - (inner.offset - outer.offset);
}
bool overlaps(const ByteRange &left, const ByteRange &right) noexcept {
  if (!same_allocation(left, right) || left.size == 0 || right.size == 0)
    return false;
  if (left.offset <= right.offset)
    return right.offset - left.offset < left.size;
  return left.offset - right.offset < right.size;
}

bool matches_alias_offset(const ByteRange &input, const ByteRange &output,
                          std::size_t byte_offset) noexcept {
  return same_allocation(input, output) && output.offset >= input.offset &&
         output.offset - input.offset == byte_offset &&
         byte_offset <= input.size && output.size <= input.size - byte_offset;
}

Status validate_aliases(const detail::PreparedState &prepared,
                        std::span<const ConstTensorView> inputs,
                        std::span<const TensorView> outputs) noexcept {
  const OperationTraits traits = operation_traits(prepared.operation);
  for (std::size_t left = 0; left < outputs.size(); ++left) {
    const ByteRange output = byte_range(outputs[left].storage());
    // Empty tensors touch no byte. Several valid zero-sized allocations use a
    // null native handle, which is not a unique allocation identity and must
    // therefore never manufacture an alias conflict between unrelated views.
    if (output.size == 0)
      continue;
    for (std::size_t right = left + 1; right < outputs.size(); ++right) {
      if (overlaps(output, byte_range(outputs[right].storage()))) {
        return Status::InvalidArgument(
            "operation output bindings overlap unsafely");
      }
    }
    bool aliases_input0 = false;
    for (std::size_t input_index = 0; input_index < inputs.size();
         ++input_index) {
      const ByteRange input = byte_range(inputs[input_index].storage());
      if (!overlaps(input, output) && !equal_range(input, output))
        continue;
      bool allowed = false;
      const AliasSupport provider_support =
          prepared.provider_prepared.vtable->alias_support(
              prepared.provider_prepared.state, left, input_index);
      switch (traits.alias) {
      case AliasPolicy::kNone:
        break;
      case AliasPolicy::kMayAliasInput0:
        allowed = input_index == 0 && equal_range(input, output) &&
                  provider_support.supported &&
                  provider_support.byte_offset == 0;
        break;
      case AliasPolicy::kMayAliasAnyInput:
        allowed = equal_range(input, output) && provider_support.supported &&
                  provider_support.byte_offset == 0;
        break;
      case AliasPolicy::kMustAliasInput0:
        allowed =
            input_index == 0 && contains(input, output) &&
            provider_support.supported &&
            matches_alias_offset(input, output, provider_support.byte_offset);
        aliases_input0 = aliases_input0 || allowed;
        break;
      }
      if (!allowed) {
        return Status::InvalidArgument(
            "operation input and output bindings overlap unsafely");
      }
    }
    if (traits.alias == AliasPolicy::kMustAliasInput0 && !aliases_input0) {
      return Status::InvalidArgument(
          "view output must alias storage from its first input");
    }
  }
  return Status::Ok();
}

Status
validate_binding_resources(const detail::ContextState &context,
                           std::span<const ConstTensorView> inputs,
                           std::span<const TensorView> outputs) noexcept {
  for (const ConstTensorView &input : inputs) {
    Status status = context.backend_vtable->validate_resource(
        context.device, input.storage().resource());
    if (!status.ok())
      return status;
  }
  for (const TensorView &output : outputs) {
    Status status = context.backend_vtable->validate_resource(
        context.device, output.storage().resource());
    if (!status.ok())
      return status;
  }
  return Status::Ok();
}

} // namespace

PreparedOperation::PreparedOperation(const PreparedOperation &other) noexcept
    : state_(other.state_) {
  detail::retain(state_);
}

PreparedOperation &
PreparedOperation::operator=(const PreparedOperation &other) noexcept {
  if (state_ != other.state_) {
    detail::retain(other.state_);
    detail::release(state_);
    state_ = other.state_;
  }
  return *this;
}

PreparedOperation::PreparedOperation(PreparedOperation &&other) noexcept
    : state_(other.state_) {
  other.state_ = nullptr;
}

PreparedOperation &
PreparedOperation::operator=(PreparedOperation &&other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

PreparedOperation::~PreparedOperation() { detail::release(state_); }

OperationKind PreparedOperation::kind() const noexcept {
  return state_ != nullptr ? state_->operation.kind() : OperationKind::kUnknown;
}

ProviderIdentity PreparedOperation::provider() const noexcept {
  return state_ != nullptr ? state_->provider : ProviderIdentity{};
}

WorkspaceRequirement PreparedOperation::workspace_requirement() const noexcept {
  return state_ != nullptr ? state_->provider_prepared.workspace
                           : WorkspaceRequirement{};
}

bool PreparedOperation::capture_compatible() const noexcept {
  return state_ != nullptr && state_->provider_prepared.capture_compatible;
}

std::size_t PreparedOperation::input_count() const noexcept {
  return state_ != nullptr ? state_->input_count : 0;
}

std::size_t PreparedOperation::output_count() const noexcept {
  return state_ != nullptr ? state_->output_count : 0;
}

Result<const TensorDesc &>
PreparedOperation::input_description(std::size_t index) const noexcept {
  if (!valid())
    return std::unexpected(
        Status::FailedPrecondition("prepared operation is invalid"));
  if (index >= state_->input_count)
    return std::unexpected(
        Status::OutOfRange("prepared input index is out of range"));
  return state_->inputs()[index];
}

Result<const TensorDesc &>
PreparedOperation::output_description(std::size_t index) const noexcept {
  if (!valid())
    return std::unexpected(
        Status::FailedPrecondition("prepared operation is invalid"));
  if (index >= state_->output_count)
    return std::unexpected(
        Status::OutOfRange("prepared output index is out of range"));
  return state_->outputs()[index];
}

Result<AliasSupport>
PreparedOperation::alias_support(std::size_t output_index,
                                 std::size_t input_index) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("prepared operation is invalid"));
  }
  if (output_index >= state_->output_count ||
      input_index >= state_->input_count) {
    return std::unexpected(
        Status::OutOfRange("prepared alias index is out of range"));
  }
  return state_->provider_prepared.vtable->alias_support(
      state_->provider_prepared.state, output_index, input_index);
}

Result<PreparedOperation>
prepare(const Context &context, Operation operation,
        std::span<const TensorDesc> inputs,
        std::span<const TensorDesc> outputs) noexcept {
  detail::ContextState *state = detail::ContextAccess::state(context);
  if (state == nullptr)
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  if (!state->admission.try_enter())
    return std::unexpected(Status::FailedPrecondition("context is closed"));
  struct ContextAdmissionGuard final {
    detail::AdmissionGate *gate = nullptr;
    ~ContextAdmissionGuard() { gate->leave(); }
  } admission_guard{&state->admission};

  const OperationTraits traits = operation_traits(operation);
  if (!operation.valid() || !traits.inputs.accepts(inputs.size()) ||
      outputs.size() != traits.outputs) {
    return std::unexpected(Status::InvalidArgument(
        "operation binding arity does not match its schema"));
  }
  Status inference =
      validate_inferred_outputs(operation, inputs, outputs, state->allocator);
  if (!inference.ok())
    return std::unexpected(std::move(inference));

  const detail::ContextState::ProviderSlot *slots = state->providers();
  for (std::size_t index = 0; index < state->provider_count; ++index) {
    const detail::ContextState::ProviderSlot &slot = slots[index];
    const detail::ProviderVTable *table =
        detail::ProviderAccess::vtable(slot.provider);
    detail::ProviderPrepared provider_prepared{};
    Status provider_status =
        table->prepare(slot.state, operation, inputs, outputs, state->allocator,
                       &provider_prepared);
    if (provider_status.IsUnimplemented())
      continue;
    if (!provider_status.ok())
      return std::unexpected(std::move(provider_status));
    if (provider_prepared.vtable == nullptr ||
        provider_prepared.vtable->destroy_prepared == nullptr ||
        provider_prepared.vtable->alias_support == nullptr ||
        provider_prepared.vtable->create_invocation == nullptr ||
        provider_prepared.vtable->destroy_invocation == nullptr ||
        provider_prepared.vtable->submit == nullptr ||
        provider_prepared.workspace.alignment == 0 ||
        (provider_prepared.workspace.alignment &
         (provider_prepared.workspace.alignment - 1)) != 0) {
      if (provider_prepared.vtable != nullptr &&
          provider_prepared.vtable->destroy_prepared != nullptr) {
        provider_prepared.vtable->destroy_prepared(provider_prepared.state,
                                                   state->allocator);
      }
      return std::unexpected(
          Status::Internal("compute provider returned an invalid artifact"));
    }
    if (traits.alias == AliasPolicy::kMustAliasInput0 && inputs.empty()) {
      provider_prepared.vtable->destroy_prepared(provider_prepared.state,
                                                 state->allocator);
      return std::unexpected(Status::Internal(
          "compute provider accepted an impossible required alias"));
    }
    if (traits.alias == AliasPolicy::kMustAliasInput0) {
      for (std::size_t output_index = 0; output_index < outputs.size();
           ++output_index) {
        const AliasSupport support = provider_prepared.vtable->alias_support(
            provider_prepared.state, output_index, 0);
        auto input_bytes = inputs[0].required_bytes();
        auto output_bytes = outputs[output_index].required_bytes();
        if (!support.supported || !input_bytes || !output_bytes ||
            support.byte_offset > *input_bytes ||
            *output_bytes > *input_bytes - support.byte_offset) {
          provider_prepared.vtable->destroy_prepared(provider_prepared.state,
                                                     state->allocator);
          return std::unexpected(Status::Internal(
              "compute provider cannot satisfy required output alias"));
        }
      }
    }
    auto prepared = make_prepared_state(state, operation, inputs, outputs,
                                        table->identity, provider_prepared);
    if (!prepared)
      return std::unexpected(std::move(prepared).error());
    return PreparedOperation(*prepared);
  }
  return std::unexpected(Status::Unimplemented(
      "no configured compute provider supports this operation"));
}

Result<PreparedOperation> prepare(const Context &context, View operation,
                                  TensorDesc input) noexcept {
  const std::array inputs{input};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context, Copy operation,
                                  TensorDesc input) noexcept {
  const std::array inputs{input};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context,
                                  Fill operation) noexcept {
  const std::array<TensorDesc, 0> inputs{};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context, Cast operation,
                                  TensorDesc input) noexcept {
  const std::array inputs{input};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context, Unary operation,
                                  TensorDesc input) noexcept {
  const std::array inputs{input};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context, Binary operation,
                                  TensorDesc left, TensorDesc right) noexcept {
  const std::array inputs{left, right};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context, Select operation,
                                  TensorDesc condition, TensorDesc when_true,
                                  TensorDesc when_false) noexcept {
  const std::array inputs{condition, when_true, when_false};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context, Reduce operation,
                                  TensorDesc input) noexcept {
  const std::array inputs{input};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context, LayerNorm operation,
                                  TensorDesc input) noexcept {
  const std::array inputs{input};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context, LayerNorm operation,
                                  TensorDesc input, TensorDesc scale) noexcept {
  const std::array inputs{input, scale};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context, LayerNorm operation,
                                  TensorDesc input, TensorDesc scale,
                                  TensorDesc bias) noexcept {
  const std::array inputs{input, scale, bias};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context, RmsNorm operation,
                                  TensorDesc input,
                                  TensorDesc output) noexcept {
  const std::array inputs{input};
  const std::array outputs{output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context, RmsNorm operation,
                                  TensorDesc input, TensorDesc weight,
                                  TensorDesc output) noexcept {
  const std::array inputs{input, weight};
  const std::array outputs{output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context, MatMul operation,
                                  TensorDesc left, TensorDesc right,
                                  TensorDesc output) noexcept {
  const std::array inputs{left, right};
  const std::array outputs{output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context &context,
                                  GatedActivation operation, TensorDesc input,
                                  TensorDesc output) noexcept {
  const std::array inputs{input};
  const std::array outputs{output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<OperationInvocation>
PreparedOperation::create_invocation(mem::BufferView workspace) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(
        Status::FailedPrecondition("prepared operation is invalid"));
  const WorkspaceRequirement requirement = workspace_requirement();
  if (requirement.size != 0) {
    if (!workspace.valid() || workspace.size() < requirement.size ||
        workspace.offset() % requirement.alignment != 0) {
      return std::unexpected(Status::InvalidArgument(
          "operation workspace does not satisfy size/alignment"));
    }
    Status resource = state_->context->backend_vtable->validate_resource(
        state_->context->device, workspace.resource());
    if (!resource.ok())
      return std::unexpected(std::move(resource));
  }

  void *provider_state = nullptr;
  const detail::ProviderPrepared &provider = state_->provider_prepared;
  Status status = provider.vtable->create_invocation(
      provider.state, state_->allocator, workspace, &provider_state);
  if (!status.ok())
    return std::unexpected(std::move(status));

  void *storage = state_->allocator.allocate(sizeof(detail::InvocationState),
                                             alignof(detail::InvocationState));
  if (storage == nullptr) {
    provider.vtable->destroy_invocation(provider_state, state_->allocator);
    return std::unexpected(
        Status::ResourceExhausted("operation invocation allocation failed"));
  }
  auto *invocation =
      ::new (storage) detail::InvocationState(state_, provider_state);
  return OperationInvocation(invocation);
}

OperationInvocation::OperationInvocation(OperationInvocation &&other) noexcept
    : state_(other.state_) {
  other.state_ = nullptr;
}

OperationInvocation &
OperationInvocation::operator=(OperationInvocation &&other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

OperationInvocation::~OperationInvocation() { detail::release(state_); }

ProviderIdentity OperationInvocation::provider() const noexcept {
  return state_ != nullptr ? state_->prepared->provider : ProviderIdentity{};
}

bool OperationInvocation::in_flight() const noexcept {
  return state_ != nullptr && state_->in_flight.load(std::memory_order_acquire);
}

Result<Completion>
OperationInvocation::submit(Queue &queue,
                            std::span<const ConstTensorView> inputs,
                            std::span<const TensorView> outputs) noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("operation invocation is invalid"));
  }
  bool idle = false;
  if (!state_->in_flight.compare_exchange_strong(
          idle, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
    return std::unexpected(Status::FailedPrecondition(
        "operation invocation already has work in flight"));
  }
  struct RunRollback final {
    detail::InvocationState *state = nullptr;
    bool armed = true;
    ~RunRollback() {
      if (armed)
        state->in_flight.store(false, std::memory_order_release);
    }
    void disarm() noexcept { armed = false; }
  } rollback{state_};

  auto sequence = detail::QueueSequence::begin(queue);
  if (!sequence)
    return std::unexpected(std::move(sequence).error());
  detail::EnqueueResult enqueued = detail::OperationInvocationAccess::enqueue(
      *this, *sequence, inputs, outputs);
  sequence->record(std::move(enqueued));
  auto completion = sequence->finish();
  if (!completion)
    return std::unexpected(std::move(completion).error());
  if (!detail::CompletionAccess::has_async_state(*completion)) {
    state_->in_flight.store(false, std::memory_order_release);
    rollback.disarm();
    return completion;
  }

  detail::retain(state_);
  state_->completion_lifetime = detail::CompletionLifetime{
      .state = state_,
      .terminal = &mark_operation_terminal,
      .release = &release_operation_lifetime,
  };
  Status attached = detail::CompletionAccess::attach_lifetime(
      *completion, state_->completion_lifetime);
  if (!attached.ok())
    return std::unexpected(std::move(attached));
  rollback.disarm();
  return completion;
}

detail::EnqueueResult detail::OperationInvocationAccess::enqueue(
    OperationInvocation &invocation, detail::QueueSequence &sequence,
    std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs) noexcept {
  if (!invocation.valid())
    return EnqueueResult::rejected(
        Status::FailedPrecondition("operation invocation is invalid"));
  if (!sequence.valid())
    return EnqueueResult::rejected(
        Status::FailedPrecondition("compute queue sequence is invalid"));
  Queue &queue = sequence.queue();
  if (!queue.valid())
    return EnqueueResult::rejected(
        Status::FailedPrecondition("queue is invalid"));
  detail::PreparedState &prepared = *invocation.state_->prepared;
  detail::QueueState *queue_state = detail::ContextAccess::state(queue);
  if (queue_state->context != prepared.context) {
    return EnqueueResult::rejected(Status::InvalidArgument(
        "queue and operation invocation belong to different contexts"));
  }
  if (inputs.size() != prepared.input_count ||
      outputs.size() != prepared.output_count) {
    return EnqueueResult::rejected(Status::InvalidArgument(
        "binding count does not match the prepared operation"));
  }
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    if (!inputs[index].valid() ||
        inputs[index].description() != prepared.inputs()[index]) {
      return EnqueueResult::rejected(Status::InvalidArgument(
          "input binding does not match its prepared description"));
    }
  }
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (!outputs[index].valid() ||
        outputs[index].description() != prepared.outputs()[index]) {
      return EnqueueResult::rejected(Status::InvalidArgument(
          "output binding does not match its prepared description"));
    }
  }
  Status resource =
      validate_binding_resources(*prepared.context, inputs, outputs);
  if (!resource.ok())
    return EnqueueResult::rejected(std::move(resource));
  Status alias = validate_aliases(prepared, inputs, outputs);
  if (!alias.ok())
    return EnqueueResult::rejected(std::move(alias));

  return prepared.provider_prepared.vtable->submit(
      prepared.provider_prepared.state, invocation.state_->provider_state,
      queue, inputs, outputs);
}

detail::EnqueueResult detail::OperationInvocationAccess::enqueue_prevalidated(
    OperationInvocation &invocation, detail::QueueSequence &sequence,
    std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs) noexcept {
  if (!invocation.valid())
    return EnqueueResult::rejected(
        Status::FailedPrecondition("operation invocation is invalid"));
  if (!sequence.valid())
    return EnqueueResult::rejected(
        Status::FailedPrecondition("compute queue sequence is invalid"));
  detail::PreparedState &prepared = *invocation.state_->prepared;
  detail::QueueState *queue_state =
      detail::ContextAccess::state(sequence.queue());
  if (queue_state == nullptr || queue_state->context != prepared.context ||
      inputs.size() != prepared.input_count ||
      outputs.size() != prepared.output_count) {
    return EnqueueResult::rejected(
        Status::Internal("prevalidated graph operation route is invalid"));
  }
  return prepared.provider_prepared.vtable->submit(
      prepared.provider_prepared.state, invocation.state_->provider_state,
      sequence.queue(), inputs, outputs);
}

Result<Completion>
OperationInvocation::submit(Queue &queue, const TensorView &output) noexcept {
  return submit(queue, {}, std::span<const TensorView>(&output, 1));
}

Result<Completion>
OperationInvocation::submit(Queue &queue, const ConstTensorView &input,
                            const TensorView &output) noexcept {
  return submit(queue, std::span<const ConstTensorView>(&input, 1),
                std::span<const TensorView>(&output, 1));
}

Result<Completion>
OperationInvocation::submit(Queue &queue, const ConstTensorView &first,
                            const ConstTensorView &second,
                            const TensorView &output) noexcept {
  const std::array inputs{first, second};
  return submit(queue, inputs, std::span<const TensorView>(&output, 1));
}

Result<Completion> OperationInvocation::submit(
    Queue &queue, const ConstTensorView &first, const ConstTensorView &second,
    const ConstTensorView &third, const TensorView &output) noexcept {
  const std::array inputs{first, second, third};
  return submit(queue, inputs, std::span<const TensorView>(&output, 1));
}

} // namespace hyle::compute
