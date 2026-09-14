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
constexpr std::size_t kInlineBindings = 8;

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

struct PreparedAllocationLayout final {
  std::size_t size = 0;
  std::size_t descriptor_offset = 0;
  std::size_t resource_offset = 0;
};

Result<PreparedAllocationLayout>
prepared_allocation_layout(std::size_t input_count,
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
  const std::size_t descriptor_end = offset + count * sizeof(TensorDesc);
  constexpr std::size_t resource_alignment = alignof(mem::MemoryResourceRef);
  static_assert((resource_alignment & (resource_alignment - 1)) == 0);
  static_assert(alignof(detail::PreparedState) >= resource_alignment);
  if (descriptor_end >
      (std::numeric_limits<std::size_t>::max)() - resource_alignment + 1) {
    return std::unexpected(
        Status::OutOfRange("prepared resource offset overflows"));
  }
  const std::size_t resource_offset =
      align_up(descriptor_end, resource_alignment);
  if (count > ((std::numeric_limits<std::size_t>::max)() - resource_offset) /
                  sizeof(mem::MemoryResourceRef)) {
    return std::unexpected(
        Status::OutOfRange("prepared resource storage overflows"));
  }
  return PreparedAllocationLayout{
      .size = resource_offset + count * sizeof(mem::MemoryResourceRef),
      .descriptor_offset = offset,
      .resource_offset = resource_offset,
  };
}

template <class T, std::size_t InlineCapacity> class ScratchStorage final {
public:
  ScratchStorage(std::size_t size, mem::AllocatorRef allocator) noexcept
      : size_(size), allocator_(allocator), data_(local_.data()) {
    if (size <= InlineCapacity)
      return;
    if (size > (std::numeric_limits<std::size_t>::max)() / sizeof(T)) {
      data_ = nullptr;
      return;
    }
    bytes_ = size * sizeof(T);
    data_ = static_cast<T *>(allocator_.allocate(bytes_, alignof(T)));
    if (data_ == nullptr)
      return;
    for (std::size_t index = 0; index < size_; ++index)
      std::construct_at(data_ + index);
  }

  ScratchStorage(const ScratchStorage &) = delete;
  ScratchStorage &operator=(const ScratchStorage &) = delete;

  ~ScratchStorage() {
    if (bytes_ == 0 || data_ == nullptr)
      return;
    for (std::size_t index = size_; index != 0; --index)
      std::destroy_at(data_ + index - 1);
    allocator_.deallocate(data_, bytes_, alignof(T));
  }

  bool valid() const noexcept { return data_ != nullptr; }
  std::span<T> span() noexcept { return {data_, size_}; }
  std::span<const T> span() const noexcept { return {data_, size_}; }
  T &operator[](std::size_t index) noexcept { return data_[index]; }

private:
  std::array<T, InlineCapacity> local_{};
  std::size_t size_ = 0;
  mem::AllocatorRef allocator_{};
  T *data_ = nullptr;
  std::size_t bytes_ = 0;
};

constexpr bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

Status validate_requirement(
    const detail::ResourceBindingRequirement &requirement) noexcept {
  if (!is_power_of_two(requirement.minimum_alignment)) {
    return Status::Internal(
        "compute provider returned an invalid resource alignment");
  }
  const auto required =
      static_cast<std::uint32_t>(requirement.required_properties);
  const auto forbidden =
      static_cast<std::uint32_t>(requirement.forbidden_properties);
  if ((required & forbidden) != 0) {
    return Status::Internal(
        "compute provider resource properties contradict each other");
  }
  if (requirement.requires_exact_domain &&
      requirement.exact_domain.authority == nullptr) {
    return Status::Internal(
        "compute provider returned an invalid exact memory domain");
  }
  return Status::Ok();
}

Status match_requirement(
    const detail::ResourceBindingRequirement &requirement,
    mem::MemoryResourceRef resource) noexcept {
  if (!resource.valid())
    return Status::InvalidArgument("operation resource binding is invalid");
  const mem::MemoryProperties properties = resource.properties();
  const auto flags = static_cast<std::uint32_t>(properties.flags);
  const auto required =
      static_cast<std::uint32_t>(requirement.required_properties);
  const auto forbidden =
      static_cast<std::uint32_t>(requirement.forbidden_properties);
  if ((flags & required) != required || (flags & forbidden) != 0 ||
      (requirement.exact_resource.valid() &&
       requirement.exact_resource != resource) ||
      (requirement.requires_exact_domain &&
       requirement.exact_domain != properties.domain)) {
    return Status::Unimplemented(
        "compute provider does not accept the selected memory resource");
  }
  return Status::Ok();
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
                    std::span<const mem::MemoryResourceRef> input_resources,
                    std::span<const mem::MemoryResourceRef> output_resources,
                    mem::MemoryResourceRef workspace_resource,
                    ProviderIdentity provider,
                    detail::ProviderPrepared provider_prepared) noexcept {
  auto layout = prepared_allocation_layout(inputs.size(), outputs.size());
  if (!layout) {
    provider_prepared.vtable->destroy_prepared(provider_prepared.state,
                                               context->allocator);
    return std::unexpected(std::move(layout).error());
  }
  void *storage = context->allocator.allocate(layout->size,
                                              alignof(detail::PreparedState));
  if (storage == nullptr) {
    provider_prepared.vtable->destroy_prepared(provider_prepared.state,
                                               context->allocator);
    return std::unexpected(
        Status::ResourceExhausted("prepared operation allocation failed"));
  }
  auto *state = ::new (storage)
      detail::PreparedState(context, operation, provider, layout->size,
                            layout->descriptor_offset, layout->resource_offset,
                            inputs.size(), outputs.size(), workspace_resource);
  for (std::size_t index = 0; index < inputs.size(); ++index)
    std::construct_at(state->inputs() + index, inputs[index]);
  for (std::size_t index = 0; index < outputs.size(); ++index)
    std::construct_at(state->outputs() + index, outputs[index]);
  for (std::size_t index = 0; index < input_resources.size(); ++index) {
    std::construct_at(state->input_resources() + index,
                      input_resources[index]);
  }
  for (std::size_t index = 0; index < output_resources.size(); ++index) {
    std::construct_at(state->output_resources() + index,
                      output_resources[index]);
  }
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

Status validate_bindings(const detail::PreparedState &prepared,
                         std::span<const ConstTensorView> inputs,
                         std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != prepared.input_count ||
      outputs.size() != prepared.output_count) {
    return Status::InvalidArgument(
        "binding count does not match the prepared operation");
  }
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const ConstTensorView &input = inputs[index];
    if (!input.valid() || input.description() != prepared.inputs()[index] ||
        input.storage().resource() != prepared.input_resources()[index]) {
      return Status::InvalidArgument(
          "input binding does not match its prepared description/resource");
    }
    Status status = prepared.context->backend_vtable->validate_resource(
        prepared.context->device, input.storage().resource());
    if (!status.ok())
      return status;
  }
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const TensorView &output = outputs[index];
    if (!output.valid() || output.description() != prepared.outputs()[index] ||
        output.storage().resource() != prepared.output_resources()[index]) {
      return Status::InvalidArgument(
          "output binding does not match its prepared description/resource");
    }
    Status status = prepared.context->backend_vtable->validate_resource(
        prepared.context->device, output.storage().resource());
    if (!status.ok())
      return status;
  }
  return validate_aliases(prepared, inputs, outputs);
}

Result<PreparedOperation>
prepare_inferred_single(const Context &context, Operation operation,
                        std::span<const TensorDesc> inputs,
                        mem::MemoryResourceRef resource) noexcept {
  auto output = infer_single_output(operation, inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, operation, inputs, outputs, resource);
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

ProviderRuntimeProperties
PreparedOperation::runtime_properties() const noexcept {
  return state_ != nullptr ? state_->provider_prepared.runtime
                           : ProviderRuntimeProperties{};
}

WorkspaceRequirement PreparedOperation::workspace_requirement() const noexcept {
  return state_ != nullptr ? state_->provider_prepared.workspace
                           : WorkspaceRequirement{};
}

std::optional<std::size_t> PreparedOperation::persistent_bytes() const noexcept {
  return state_ != nullptr ? state_->provider_prepared.persistent_bytes
                           : std::optional<std::size_t>{};
}

bool PreparedOperation::capture_compatible() const noexcept {
  return state_ != nullptr &&
         state_->provider_prepared.runtime.capture_compatible;
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

Result<detail::ProviderOperationProposal>
detail::OperationPreparationAccess::propose(
    const Context &context, std::size_t provider_index, Operation operation,
    std::span<const TensorDesc> inputs,
    std::span<const TensorDesc> outputs,
    std::span<ResourceBindingRequirement> input_requirements,
    std::span<ResourceBindingRequirement> output_requirements,
    ProviderShapeDomain shape) noexcept {
  detail::ContextState *state = detail::ContextAccess::state(context);
  if (state == nullptr)
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  if (input_requirements.size() != inputs.size() ||
      output_requirements.size() != outputs.size()) {
    return std::unexpected(Status::InvalidArgument(
        "provider proposal requirement counts do not match bindings"));
  }
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
  Status domain_status = validate_shape_domain(shape, inputs, outputs);
  if (!domain_status.ok())
    return std::unexpected(std::move(domain_status));
  // A derived/guarded domain's axis-wise envelope need not be an admitted
  // concrete shape. Only the provider's whole-domain proof can qualify it.
  if (shape.exact()) {
    Status inference =
        validate_inferred_outputs(operation, inputs, outputs, state->allocator);
    if (!inference.ok())
      return std::unexpected(std::move(inference));
  }

  if (provider_index >= state->provider_count) {
    return std::unexpected(
        Status::OutOfRange("compute provider index is out of range"));
  }
  for (ResourceBindingRequirement &requirement : input_requirements)
    requirement = {};
  for (ResourceBindingRequirement &requirement : output_requirements)
    requirement = {};
  const detail::ContextState::ProviderSlot &slot =
      state->providers()[provider_index];
  const detail::ProviderVTable *table =
      detail::ProviderAccess::vtable(slot.provider);
  detail::ProviderOperationProposal proposal{};
  const ProviderOperationRequest request{operation, inputs, outputs, shape};
  Status status = table->propose(slot.state, request,
                                 input_requirements, output_requirements,
                                 &proposal);
  if (!status.ok())
    return std::unexpected(std::move(status));
  if (proposal.shape_coverage != ProviderShapeCoverage::kExactPoint &&
      proposal.shape_coverage != ProviderShapeCoverage::kEntireDomain) {
    return std::unexpected(Status::Internal(
        "compute provider returned invalid shape coverage"));
  }
  if (!shape.exact() &&
      proposal.shape_coverage != ProviderShapeCoverage::kEntireDomain) {
    return std::unexpected(Status::Unimplemented(
        "compute provider only supports an exact shape point"));
  }
  proposal.context = state;
  proposal.provider = table->identity;
  proposal.provider_index = provider_index;
  if (!proposal.runtime.valid()) {
    return std::unexpected(Status::Internal(
        "compute provider returned invalid runtime properties"));
  }
  if (!proposal.cost.valid()) {
    return std::unexpected(Status::Internal(
        "compute provider returned invalid structural cost facts"));
  }
  status = validate_requirement(proposal.workspace_resource);
  if (!status.ok())
    return std::unexpected(std::move(status));
  for (const ResourceBindingRequirement &requirement : input_requirements) {
    status = validate_requirement(requirement);
    if (!status.ok())
      return std::unexpected(std::move(status));
  }
  for (const ResourceBindingRequirement &requirement : output_requirements) {
    status = validate_requirement(requirement);
    if (!status.ok())
      return std::unexpected(std::move(status));
  }
  return proposal;
}

Status detail::OperationPreparationAccess::match_resource(
    const ResourceBindingRequirement &requirement,
    mem::MemoryResourceRef resource) noexcept {
  Status status = validate_requirement(requirement);
  if (!status.ok())
    return status;
  return match_requirement(requirement, resource);
}

Result<detail::ProviderPrepareAttempt>
detail::OperationPreparationAccess::prepare_candidate(
    const Context &context, const ProviderOperationProposal &proposal,
    Operation operation, std::span<const TensorDesc> inputs,
    std::span<const TensorDesc> outputs,
    std::span<const ResourceBindingRequirement> input_requirements,
    std::span<const ResourceBindingRequirement> output_requirements,
    OperationResourceBindings resources) noexcept {
  detail::ContextState *state = detail::ContextAccess::state(context);
  if (state == nullptr)
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  if (resources.inputs.size() != inputs.size() ||
      resources.outputs.size() != outputs.size() ||
      input_requirements.size() != inputs.size() ||
      output_requirements.size() != outputs.size()) {
    return std::unexpected(Status::InvalidArgument(
        "operation resource counts do not match its bindings"));
  }
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
  Status status =
      validate_inferred_outputs(operation, inputs, outputs, state->allocator);
  if (!status.ok())
    return std::unexpected(std::move(status));
  if (proposal.provider_index >= state->provider_count) {
    return std::unexpected(
        Status::OutOfRange("compute provider index is out of range"));
  }
  const detail::ContextState::ProviderSlot &slot =
      state->providers()[proposal.provider_index];
  const detail::ProviderVTable *table =
      detail::ProviderAccess::vtable(slot.provider);
  if (proposal.context != state || proposal.provider != table->identity ||
      !proposal.runtime.valid()) {
    return std::unexpected(Status::InvalidArgument(
        "operation proposal does not belong to the selected provider"));
  }

  for (std::size_t index = 0; index < resources.inputs.size(); ++index) {
    status = validate_requirement(input_requirements[index]);
    if (!status.ok())
      return std::unexpected(std::move(status));
    status = match_requirement(input_requirements[index],
                               resources.inputs[index]);
    if (!status.ok())
      return std::unexpected(std::move(status));
    status = state->backend_vtable->validate_resource(
        state->device, resources.inputs[index]);
    if (!status.ok())
      return std::unexpected(std::move(status));
  }
  for (std::size_t index = 0; index < resources.outputs.size(); ++index) {
    status = validate_requirement(output_requirements[index]);
    if (!status.ok())
      return std::unexpected(std::move(status));
    status = match_requirement(output_requirements[index],
                               resources.outputs[index]);
    if (!status.ok())
      return std::unexpected(std::move(status));
    status = state->backend_vtable->validate_resource(
        state->device, resources.outputs[index]);
    if (!status.ok())
      return std::unexpected(std::move(status));
  }

  mem::MemoryResourceRef workspace_resource = resources.workspace_resource;
  if (proposal.runtime.workspace_owner == ProviderWorkspaceOwner::kCaller &&
      !workspace_resource.valid()) {
    if (!resources.outputs.empty())
      workspace_resource = resources.outputs.front();
    else if (!resources.inputs.empty())
      workspace_resource = resources.inputs.front();
  }
  if (proposal.runtime.workspace_owner == ProviderWorkspaceOwner::kCaller &&
      workspace_resource.valid()) {
    status = match_requirement(proposal.workspace_resource,
                               workspace_resource);
    if (!status.ok())
      return std::unexpected(std::move(status));
    status = state->backend_vtable->validate_resource(state->device,
                                                      workspace_resource);
    if (!status.ok())
      return std::unexpected(std::move(status));
  } else if (proposal.runtime.workspace_owner ==
             ProviderWorkspaceOwner::kProvider) {
    workspace_resource = {};
  }

  detail::ProviderPrepared provider_prepared{};
  const detail::ProviderPrepareRequest request{
      .operation = operation,
      .inputs = inputs,
      .outputs = outputs,
      .input_resources = resources.inputs,
      .output_resources = resources.outputs,
      .workspace_resource = workspace_resource,
      .route = proposal.route,
  };
  const auto destroy_invalid = [&]() noexcept {
    if (provider_prepared.vtable != nullptr &&
        provider_prepared.vtable->destroy_prepared != nullptr) {
      provider_prepared.vtable->destroy_prepared(provider_prepared.state,
                                                 state->allocator);
    }
  };
  detail::ProviderPrepareOutcome outcome = table->prepare(
      slot.state, request, state->allocator, &provider_prepared);
  if (outcome.kind != detail::ProviderPrepareOutcomeKind::kPrepared) {
    const bool valid_decline =
        (outcome.kind ==
             detail::ProviderPrepareOutcomeKind::kDeclinedUnsupported &&
         outcome.status.IsUnimplemented()) ||
        (outcome.kind ==
             detail::ProviderPrepareOutcomeKind::kDeclinedResourceBudget &&
         outcome.status.IsResourceExhausted()) ||
        (outcome.kind == detail::ProviderPrepareOutcomeKind::kError &&
         !outcome.status.ok());
    if (provider_prepared.state != nullptr &&
        (provider_prepared.vtable == nullptr ||
         provider_prepared.vtable->destroy_prepared == nullptr)) {
      return std::unexpected(Status::Internal(
          "compute provider returned an unowned partial artifact"));
    }
    destroy_invalid();
    if (!valid_decline) {
      return std::unexpected(Status::Internal(
          "compute provider returned an invalid prepare outcome"));
    }
    if (outcome.kind == detail::ProviderPrepareOutcomeKind::kError)
      return std::unexpected(std::move(outcome.status));
    return detail::ProviderPrepareAttempt{
        .decline = outcome.kind ==
                           detail::ProviderPrepareOutcomeKind::kDeclinedUnsupported
                       ? detail::ProviderPrepareDecline::kUnsupported
                       : detail::ProviderPrepareDecline::kResourceBudget,
        .reason = std::move(outcome.status),
    };
  }
  if (!outcome.status.ok()) {
    destroy_invalid();
    return std::unexpected(Status::Internal(
        "compute provider returned an error with a prepared outcome"));
  }
  const bool shared =
      provider_prepared.runtime.mode == ProviderRuntimeMode::kSharedQueue;
  if (provider_prepared.vtable == nullptr ||
      provider_prepared.vtable->destroy_prepared == nullptr ||
      provider_prepared.vtable->alias_support == nullptr ||
      provider_prepared.vtable->create_invocation == nullptr ||
      provider_prepared.vtable->destroy_invocation == nullptr ||
      (shared &&
       (provider_prepared.vtable->shared_submit == nullptr ||
        provider_prepared.vtable->delegated_submit != nullptr ||
        provider_prepared.vtable->endpoint != nullptr ||
        provider_prepared.vtable->can_import_endpoint != nullptr)) ||
      (!shared &&
       (provider_prepared.vtable->shared_submit != nullptr ||
        provider_prepared.vtable->delegated_submit == nullptr ||
        provider_prepared.vtable->endpoint == nullptr ||
        provider_prepared.vtable->can_import_endpoint == nullptr)) ||
      !provider_prepared.runtime.valid() ||
      provider_prepared.runtime != proposal.runtime ||
      !is_power_of_two(provider_prepared.workspace.alignment) ||
      (proposal.cost.workspace.has_value() &&
       (provider_prepared.workspace.size != proposal.cost.workspace->size ||
        provider_prepared.workspace.alignment !=
            proposal.cost.workspace->alignment)) ||
      (proposal.cost.persistent_bytes.has_value() &&
       provider_prepared.persistent_bytes !=
           proposal.cost.persistent_bytes)) {
    destroy_invalid();
    return std::unexpected(
        Status::Internal("compute provider returned an invalid artifact"));
  }
  if (provider_prepared.runtime.workspace_owner ==
          ProviderWorkspaceOwner::kCaller &&
      provider_prepared.workspace.size != 0 && !workspace_resource.valid()) {
    destroy_invalid();
    return std::unexpected(Status::InvalidArgument(
        "operation requires an explicit caller workspace resource"));
  }
  if (traits.alias == AliasPolicy::kMustAliasInput0 && inputs.empty()) {
    destroy_invalid();
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
        destroy_invalid();
        return std::unexpected(Status::Internal(
            "compute provider cannot satisfy required output alias"));
      }
    }
  }
  auto prepared = make_prepared_state(
      state, operation, inputs, outputs, resources.inputs, resources.outputs,
      workspace_resource, table->identity, provider_prepared);
  if (!prepared)
    return std::unexpected(std::move(prepared).error());
  return detail::ProviderPrepareAttempt{
      .prepared = PreparedOperation(*prepared),
  };
}

Result<PreparedOperation> detail::OperationPreparationAccess::prepare(
    const Context &context, const ProviderOperationProposal &proposal,
    Operation operation, std::span<const TensorDesc> inputs,
    std::span<const TensorDesc> outputs,
    std::span<const ResourceBindingRequirement> input_requirements,
    std::span<const ResourceBindingRequirement> output_requirements,
    OperationResourceBindings resources) noexcept {
  auto attempt = prepare_candidate(context, proposal, operation, inputs,
                                   outputs, input_requirements,
                                   output_requirements, resources);
  if (!attempt)
    return std::unexpected(std::move(attempt).error());
  if (!attempt->succeeded())
    return std::unexpected(std::move(attempt->reason));
  return std::move(attempt->prepared);
}

Result<PreparedOperation>
prepare(const Context &context, Operation operation,
        std::span<const TensorDesc> inputs,
        std::span<const TensorDesc> outputs,
        OperationResourceBindings resources) noexcept {
  detail::ContextState *state = detail::ContextAccess::state(context);
  if (state == nullptr)
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  if (resources.inputs.size() != inputs.size() ||
      resources.outputs.size() != outputs.size()) {
    return std::unexpected(Status::InvalidArgument(
        "operation resource counts do not match its bindings"));
  }
  ScratchStorage<detail::ResourceBindingRequirement, kInlineBindings>
      input_requirements(inputs.size(), state->allocator);
  ScratchStorage<detail::ResourceBindingRequirement, kInlineBindings>
      output_requirements(outputs.size(), state->allocator);
  if (!input_requirements.valid() || !output_requirements.valid()) {
    return std::unexpected(Status::ResourceExhausted(
        "operation proposal requirement allocation failed"));
  }

  for (std::size_t index = 0; index < state->provider_count; ++index) {
    auto proposal = detail::OperationPreparationAccess::propose(
        context, index, operation, inputs, outputs, input_requirements.span(),
        output_requirements.span());
    if (!proposal && proposal.error().IsUnimplemented())
      continue;
    if (!proposal)
      return std::unexpected(std::move(proposal).error());
    // The public direct route promises execution on the caller's Queue.
    // Delegated artifacts are selected only by Graph through the internal
    // proposal/exact-prepare seam; returning one here would hand a direct
    // caller an OperationInvocation that it cannot submit.
    if (proposal->runtime.mode != ProviderRuntimeMode::kSharedQueue)
      continue;
    auto prepared = detail::OperationPreparationAccess::prepare(
        context, *proposal, operation, inputs, outputs,
        input_requirements.span(), output_requirements.span(), resources);
    if (!prepared && prepared.error().IsUnimplemented())
      continue;
    return prepared;
  }
  return std::unexpected(Status::Unimplemented(
      "no configured compute provider supports this operation"));
}

Result<PreparedOperation>
prepare(const Context &context, Operation operation,
        std::span<const TensorDesc> inputs,
        std::span<const TensorDesc> outputs,
        mem::MemoryResourceRef resource) noexcept {
  detail::ContextState *state = detail::ContextAccess::state(context);
  if (state == nullptr)
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  if (!resource.valid()) {
    return std::unexpected(
        Status::InvalidArgument("operation memory resource is invalid"));
  }
  ScratchStorage<mem::MemoryResourceRef, kInlineBindings> input_resources(
      inputs.size(), state->allocator);
  ScratchStorage<mem::MemoryResourceRef, kInlineBindings> output_resources(
      outputs.size(), state->allocator);
  if (!input_resources.valid() || !output_resources.valid()) {
    return std::unexpected(Status::ResourceExhausted(
        "operation resource binding allocation failed"));
  }
  for (mem::MemoryResourceRef &binding : input_resources.span())
    binding = resource;
  for (mem::MemoryResourceRef &binding : output_resources.span())
    binding = resource;
  return prepare(context, operation, inputs, outputs,
                 OperationResourceBindings{
                     .inputs = input_resources.span(),
                     .outputs = output_resources.span(),
                     .workspace_resource = resource,
                 });
}

Result<PreparedOperation>
prepare(const Context &context, Operation operation,
        std::span<const ConstTensorView> inputs,
        std::span<const TensorView> outputs,
        mem::MemoryResourceRef workspace_resource) noexcept {
  detail::ContextState *state = detail::ContextAccess::state(context);
  if (state == nullptr)
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  ScratchStorage<TensorDesc, kInlineBindings> input_descriptions(
      inputs.size(), state->allocator);
  ScratchStorage<TensorDesc, kInlineBindings> output_descriptions(
      outputs.size(), state->allocator);
  ScratchStorage<mem::MemoryResourceRef, kInlineBindings> input_resources(
      inputs.size(), state->allocator);
  ScratchStorage<mem::MemoryResourceRef, kInlineBindings> output_resources(
      outputs.size(), state->allocator);
  if (!input_descriptions.valid() || !output_descriptions.valid() ||
      !input_resources.valid() || !output_resources.valid()) {
    return std::unexpected(Status::ResourceExhausted(
        "operation binding metadata allocation failed"));
  }
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    if (!inputs[index].valid()) {
      return std::unexpected(
          Status::InvalidArgument("operation input view is invalid"));
    }
    input_descriptions[index] = inputs[index].description();
    input_resources[index] = inputs[index].storage().resource();
  }
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (!outputs[index].valid()) {
      return std::unexpected(
          Status::InvalidArgument("operation output view is invalid"));
    }
    output_descriptions[index] = outputs[index].description();
    output_resources[index] = outputs[index].storage().resource();
  }
  return prepare(context, operation, input_descriptions.span(),
                 output_descriptions.span(),
                 OperationResourceBindings{
                     .inputs = input_resources.span(),
                     .outputs = output_resources.span(),
                     .workspace_resource = workspace_resource,
                 });
}

Result<PreparedOperation> prepare(const Context &context, View operation,
                                  TensorDesc input,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs, resource);
}

Result<PreparedOperation> prepare(const Context &context, Copy operation,
                                  TensorDesc input,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs, resource);
}

Result<PreparedOperation> prepare(const Context &context,
                                  Fill operation,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array<TensorDesc, 0> inputs{};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs, resource);
}

Result<PreparedOperation> prepare(const Context &context, Cast operation,
                                  TensorDesc input,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs, resource);
}

Result<PreparedOperation> prepare(const Context &context, Unary operation,
                                  TensorDesc input,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs, resource);
}

Result<PreparedOperation> prepare(const Context &context, Binary operation,
                                  TensorDesc left, TensorDesc right,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{left, right};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs, resource);
}

Result<PreparedOperation> prepare(const Context &context, Select operation,
                                  TensorDesc condition, TensorDesc when_true,
                                  TensorDesc when_false,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{condition, when_true, when_false};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs, resource);
}

Result<PreparedOperation> prepare(const Context &context, Reduce operation,
                                  TensorDesc input,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs, resource);
}

Result<PreparedOperation> prepare(const Context &context, LayerNorm operation,
                                  TensorDesc input,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs, resource);
}

Result<PreparedOperation> prepare(const Context &context, LayerNorm operation,
                                  TensorDesc input, TensorDesc scale,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input, scale};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs, resource);
}

Result<PreparedOperation> prepare(const Context &context, LayerNorm operation,
                                  TensorDesc input, TensorDesc scale,
                                  TensorDesc bias,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input, scale, bias};
  auto output = infer_single_output(Operation(operation), inputs);
  if (!output)
    return std::unexpected(std::move(output).error());
  const std::array outputs{*output};
  return prepare(context, Operation(operation), inputs, outputs, resource);
}

Result<PreparedOperation> prepare(const Context &context, RmsNorm operation,
                                  TensorDesc input,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input};
  return prepare_inferred_single(context, Operation(operation), inputs,
                                 resource);
}

Result<PreparedOperation> prepare(const Context &context, RmsNorm operation,
                                  TensorDesc input, TensorDesc weight,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input, weight};
  return prepare_inferred_single(context, Operation(operation), inputs,
                                 resource);
}

Result<PreparedOperation> prepare(const Context &context, MatMul operation,
                                  TensorDesc left, TensorDesc right,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{left, right};
  return prepare_inferred_single(context, Operation(operation), inputs,
                                 resource);
}

Result<PreparedOperation> prepare(const Context &context, MatMul operation,
                                  TensorDesc left, TensorDesc right,
                                  TensorDesc output,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{left, right};
  const std::array outputs{output};
  return prepare(context, Operation(operation), inputs, outputs, resource);
}

Result<PreparedOperation> prepare(const Context &context,
                                  GatedActivation operation, TensorDesc input,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input};
  return prepare_inferred_single(context, Operation(operation), inputs,
                                 resource);
}

Result<PreparedOperation> prepare(const Context &context,
                                  GatedActivation operation, TensorDesc gate,
                                  TensorDesc value,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{gate, value};
  return prepare_inferred_single(context, Operation(operation), inputs,
                                 resource);
}

Result<PreparedOperation> prepare(const Context &context, Gather operation,
                                  TensorDesc source, TensorDesc indices,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{source, indices};
  return prepare_inferred_single(context, Operation(operation), inputs,
                                 resource);
}

Result<PreparedOperation> prepare(const Context &context, Scatter operation,
                                  TensorDesc input, TensorDesc indices,
                                  TensorDesc updates,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input, indices, updates};
  return prepare_inferred_single(context, Operation(operation), inputs,
                                 resource);
}

Result<PreparedOperation> prepare(const Context &context, Concat operation,
                                  std::span<const TensorDesc> inputs,
                                  mem::MemoryResourceRef resource) noexcept {
  return prepare_inferred_single(context, Operation(operation), inputs,
                                 resource);
}

Result<PreparedOperation> prepare(const Context &context, RoPE operation,
                                  TensorDesc input, TensorDesc positions,
                                  TensorDesc frequencies,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input, positions, frequencies};
  return prepare_inferred_single(context, Operation(operation), inputs,
                                 resource);
}

Result<PreparedOperation> prepare(const Context &context, Softmax operation,
                                  TensorDesc input,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input};
  return prepare_inferred_single(context, Operation(operation), inputs,
                                 resource);
}

Result<PreparedOperation> prepare(const Context &context, Softmax operation,
                                  TensorDesc input, TensorDesc mask,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input, mask};
  return prepare_inferred_single(context, Operation(operation), inputs,
                                 resource);
}

Result<PreparedOperation> prepare(const Context &context, TopK operation,
                                  TensorDesc input,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{input};
  std::array<TensorDesc, 2> outputs{};
  Status status =
      infer_output_descriptions(Operation(operation), inputs, outputs);
  if (!status.ok())
    return std::unexpected(std::move(status));
  return prepare(context, Operation(operation), inputs, outputs, resource);
}

Result<PreparedOperation>
prepare(const Context &context, ScaledDotProductAttention operation,
        TensorDesc query, TensorDesc key, TensorDesc value,
        mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{query, key, value};
  return prepare_inferred_single(context, Operation(operation), inputs,
                                 resource);
}

Result<PreparedOperation>
prepare(const Context &context, ScaledDotProductAttention operation,
        TensorDesc query, TensorDesc key, TensorDesc value, TensorDesc mask,
        mem::MemoryResourceRef resource) noexcept {
  const std::array inputs{query, key, value, mask};
  return prepare_inferred_single(context, Operation(operation), inputs,
                                 resource);
}

Result<OperationInvocation>
PreparedOperation::create_invocation(mem::BufferView workspace) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(
        Status::FailedPrecondition("prepared operation is invalid"));
  const WorkspaceRequirement requirement = workspace_requirement();
  const ProviderWorkspaceOwner workspace_owner =
      state_->provider_prepared.runtime.workspace_owner;
  if (workspace_owner == ProviderWorkspaceOwner::kProvider) {
    if (workspace.valid()) {
      return std::unexpected(Status::InvalidArgument(
          "provider-owned operation does not accept caller workspace"));
    }
  } else if (requirement.size != 0) {
    if (!workspace.valid() || workspace.size() < requirement.size ||
        workspace.offset() % requirement.alignment != 0) {
      return std::unexpected(Status::InvalidArgument(
          "operation workspace does not satisfy size/alignment"));
    }
    if (workspace.resource() != state_->workspace_resource) {
      return std::unexpected(Status::InvalidArgument(
          "operation workspace does not use its prepared memory resource"));
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
  detail::PreparedState &prepared = *state_->prepared;
  HYLE_ASSUME(prepared.provider_prepared.runtime.mode ==
              ProviderRuntimeMode::kSharedQueue);
  HYLE_ASSUME(prepared.provider_prepared.vtable->shared_submit != nullptr);
  detail::QueueState *queue_state = detail::ContextAccess::state(queue);
  if (queue_state == nullptr) {
    return std::unexpected(Status::FailedPrecondition("queue is invalid"));
  }
  if (queue_state->context != prepared.context) {
    return std::unexpected(Status::InvalidArgument(
        "queue and operation invocation belong to different contexts"));
  }
  Status binding_status = validate_bindings(prepared, inputs, outputs);
  if (!binding_status.ok())
    return std::unexpected(std::move(binding_status));

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
  detail::EnqueueResult enqueued =
      detail::OperationInvocationAccess::enqueue_prevalidated(
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
  Status binding_status = validate_bindings(prepared, inputs, outputs);
  if (!binding_status.ok())
    return EnqueueResult::rejected(std::move(binding_status));

  return prepared.provider_prepared.vtable->shared_submit(
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
  HYLE_ASSUME(prepared.provider_prepared.runtime.mode ==
              ProviderRuntimeMode::kSharedQueue);
  HYLE_ASSUME(prepared.provider_prepared.vtable->shared_submit != nullptr);
  return prepared.provider_prepared.vtable->shared_submit(
      prepared.provider_prepared.state, invocation.state_->provider_state,
      sequence.queue(), inputs, outputs);
}

detail::CompletionEndpoint
detail::OperationInvocationAccess::delegated_endpoint(
    const OperationInvocation &invocation) noexcept {
  if (!invocation.valid())
    return {};
  const detail::InvocationState &state = *invocation.state_;
  const detail::ProviderPrepared &provider =
      state.prepared->provider_prepared;
  if (provider.runtime.mode != ProviderRuntimeMode::kDelegated ||
      provider.vtable == nullptr || provider.vtable->endpoint == nullptr) {
    return {};
  }
  return provider.vtable->endpoint(provider.state, state.provider_state);
}

bool detail::OperationInvocationAccess::can_import_endpoint(
    const OperationInvocation &invocation,
    CompletionEndpoint source) noexcept {
  if (!source.valid() || !invocation.valid())
    return false;
  const detail::InvocationState &state = *invocation.state_;
  const detail::ProviderPrepared &provider =
      state.prepared->provider_prepared;
  return provider.runtime.mode == ProviderRuntimeMode::kDelegated &&
         provider.vtable != nullptr &&
         provider.vtable->can_import_endpoint != nullptr &&
         provider.vtable->can_import_endpoint(
             provider.state, state.provider_state, source);
}

Result<Completion>
detail::OperationInvocationAccess::submit_delegated_prevalidated(
    OperationInvocation &invocation,
    std::span<const CompletionDependency> dependencies,
    std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs) noexcept {
  if (!invocation.valid()) {
    return std::unexpected(
        Status::FailedPrecondition("operation invocation is invalid"));
  }
  detail::InvocationState &state = *invocation.state_;

  bool idle = false;
  if (!state.in_flight.compare_exchange_strong(
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
  } rollback{&state};

  auto completion = submit_delegated_borrowed_prevalidated(
      invocation, dependencies, inputs, outputs);
  if (!completion)
    return std::unexpected(std::move(completion).error());
  if (!completion->valid()) {
    return std::unexpected(
        Status::Internal("delegated provider returned an invalid completion"));
  }
  if (!detail::CompletionAccess::has_async_state(*completion)) {
    state.in_flight.store(false, std::memory_order_release);
    rollback.disarm();
    return completion;
  }

  detail::retain(&state);
  state.completion_lifetime = detail::CompletionLifetime{
      .state = &state,
      .terminal = &mark_operation_terminal,
      .release = &release_operation_lifetime,
  };
  Status attached = detail::CompletionAccess::attach_lifetime(
      *completion, state.completion_lifetime);
  if (!attached.ok()) {
    completion->wait().IgnoreError();
    return std::unexpected(std::move(attached));
  }
  rollback.disarm();
  return completion;
}

Result<Completion>
detail::OperationInvocationAccess::submit_delegated_borrowed_prevalidated(
    OperationInvocation &invocation,
    std::span<const CompletionDependency> dependencies,
    std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs) noexcept {
  if (!invocation.valid()) {
    return std::unexpected(
        Status::FailedPrecondition("operation invocation is invalid"));
  }
  detail::InvocationState &state = *invocation.state_;
  detail::PreparedState &prepared = *state.prepared;
  const detail::ProviderPrepared &provider = prepared.provider_prepared;
  if (provider.runtime.mode != ProviderRuntimeMode::kDelegated ||
      provider.vtable == nullptr ||
      provider.vtable->delegated_submit == nullptr) {
    return std::unexpected(Status::FailedPrecondition(
        "operation invocation is not delegated"));
  }
  if (inputs.size() != prepared.input_count ||
      outputs.size() != prepared.output_count) {
    return std::unexpected(
        Status::Internal("prevalidated delegated bindings are invalid"));
  }
  if (!prepared.context->admission.try_enter()) {
    return std::unexpected(Status::FailedPrecondition("context is closed"));
  }
  struct ContextAdmissionGuard final {
    detail::AdmissionGate *gate = nullptr;
    ~ContextAdmissionGuard() { gate->leave(); }
  } admission_guard{&prepared.context->admission};

  auto completion = provider.vtable->delegated_submit(
      provider.state, state.provider_state, dependencies, inputs, outputs);
  if (!completion)
    return std::unexpected(std::move(completion).error());
  if (!completion->valid()) {
    return std::unexpected(
        Status::Internal("delegated provider returned an invalid completion"));
  }
  return completion;
}

Result<Completion>
OperationInvocation::submit(Queue &queue,
                            std::span<const ConstTensorView> inputs,
                            const TensorView &output) noexcept {
  return submit(queue, inputs, std::span<const TensorView>(&output, 1));
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

Result<Completion> OperationInvocation::submit(
    Queue &queue, const ConstTensorView &first, const ConstTensorView &second,
    const ConstTensorView &third, const ConstTensorView &fourth,
    const TensorView &output) noexcept {
  const std::array inputs{first, second, third, fourth};
  return submit(queue, inputs, std::span<const TensorView>(&output, 1));
}

Result<Completion> OperationInvocation::submit(
    Queue &queue, const ConstTensorView &input,
    const TensorView &first_output,
    const TensorView &second_output) noexcept {
  const std::array outputs{first_output, second_output};
  return submit(queue, std::span<const ConstTensorView>(&input, 1), outputs);
}

} // namespace hyle::compute
