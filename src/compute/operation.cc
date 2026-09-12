#include "hyle/compute/prepared_operation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <new>
#include <span>
#include <string_view>
#include <utility>

#include "context_state.h"
#include "hyle/math/scalar.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/fat_kernel.h"
#include "hyle/simd/math.h"
#include "hyle/simd/ops.h"
#include "matmul_kernels.h"
#include "prepared_state.h"

namespace hyle::compute {
namespace {

using Parameters = detail::NativeParameters;
using Route = detail::NativeRoute;

void run_rms_norm(const float *input, const float *weight, float *output,
                  const Parameters &parameters, float epsilon) noexcept {
  const std::size_t rows = parameters[0];
  const std::size_t hidden = parameters[1];
  if (hidden == 0)
    return;
  for (std::size_t row = 0; row < rows; ++row) {
    const float *row_input = input + row * hidden;
    float *row_output = output + row * hidden;
    if (weight == nullptr) {
      simd::rms_norm_f32(row_input, row_output, hidden, epsilon);
    } else {
      simd::rms_norm_f32(row_input, weight, row_output, hidden, epsilon);
    }
  }
}

void run_silu_and_mul(const float *input, const float *, float *output,
                      const Parameters &parameters, float) noexcept {
  const std::size_t rows = parameters[0];
  const std::size_t hidden = parameters[1];
  if (hidden == 0)
    return;
  for (std::size_t row = 0; row < rows; ++row) {
    const float *gate = input + row * hidden * 2;
    const float *value = gate + hidden;
    simd::swiglu_f32_batch(gate, value, output + row * hidden, hidden);
  }
}

// B is a logical [K,N] view over physically contiguous [N,K] weights. Each
// output is therefore one contiguous SIMD dot, the normal inference layout.
void run_matmul_transposed_weight(const float *left, const float *right,
                                  float *output, const Parameters &parameters,
                                  float) noexcept {
  const std::size_t rows = parameters[0];
  const std::size_t inner = parameters[1];
  const std::size_t columns = parameters[2];
  if (rows == 0 || columns == 0)
    return;
  if (inner == 0) {
    std::fill_n(output, rows * columns, 0.0F);
    return;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    const float *left_row = left + row * inner;
    float *output_row = output + row * columns;
    for (std::size_t column = 0; column < columns; ++column) {
      output_row[column] = simd::dot(left_row, right + column * inner, inner);
    }
  }
}

Route select_row_major_matmul() noexcept {
#if HYLE_ARCH_X64
  return HYLE_SIMD_SELECT_FAT_KERNEL(
      &detail::run_matmul_row_major_scalar, &detail::run_matmul_row_major_sse4,
      &detail::run_matmul_row_major_avx2, &detail::run_matmul_row_major_avx512);
#elif HYLE_ARCH_ARM64
  return HYLE_SIMD_SELECT_FAT_KERNEL(&detail::run_matmul_row_major_scalar,
                                     &detail::run_matmul_row_major_neon,
                                     &detail::run_matmul_row_major_sve);
#elif HYLE_ARCH_RISCV64
  return HYLE_SIMD_SELECT_FAT_KERNEL(&detail::run_matmul_row_major_scalar,
                                     &detail::run_matmul_row_major_rvv);
#else
  return HYLE_SIMD_SELECT_FAT_KERNEL(&detail::run_matmul_row_major_scalar);
#endif
}

Status validate_logical_f32(const TensorDesc &description) noexcept {
  if (!description.valid()) {
    return Status::InvalidArgument("operation tensor description is invalid");
  }
  if (description.data_type() != DataType::kFloat32) {
    return Status::Unimplemented(
        "operation currently requires float32 logical tensors");
  }
  return Status::Ok();
}

Status validate_dense_f32(const TensorDesc &description) noexcept {
  Status status = validate_logical_f32(description);
  if (!status.ok())
    return status;
  if (!description.is_plain()) {
    return Status::Unimplemented(
        "HYLE-native operation requires plain tensor storage");
  }
  if (!description.is_dense()) {
    return Status::Unimplemented(
        "HYLE-native operation requires dense tensors");
  }
  return Status::Ok();
}

bool is_transposed_dense_matrix(const TensorDesc &description) noexcept {
  if (!description.valid() || !description.is_plain() ||
      description.rank() != 2)
    return false;
  const auto dimensions = description.dimensions();
  const auto strides = description.byte_strides();
  const auto row_bytes = math::try_multiply(
      dimensions[0], static_cast<std::int64_t>(sizeof(float)));
  return row_bytes.has_value() &&
         strides[0] == static_cast<std::int64_t>(sizeof(float)) &&
         strides[1] == *row_bytes;
}

bool is_supported_matmul_right(const TensorDesc &description) noexcept {
  return description.is_plain() &&
         (description.is_dense() || is_transposed_dense_matrix(description));
}

Result<std::size_t> outer_count(const TensorDesc &description) noexcept {
  std::size_t result = 1;
  const auto dimensions = description.dimensions();
  for (std::size_t axis = 0; axis + 1 < dimensions.size(); ++axis) {
    auto value = math::try_integer_cast<std::size_t>(dimensions[axis]);
    if (!value) {
      return std::unexpected(
          Status::OutOfRange("tensor dimension exceeds size_t"));
    }
    auto next = math::try_multiply(result, *value);
    if (!next) {
      return std::unexpected(
          Status::OutOfRange("tensor row count overflows size_t"));
    }
    result = *next;
  }
  return result;
}

Result<std::size_t> dimension(const TensorDesc &description,
                              std::size_t axis) noexcept {
  auto result =
      math::try_integer_cast<std::size_t>(description.dimensions()[axis]);
  if (!result) {
    return std::unexpected(
        Status::OutOfRange("tensor dimension exceeds size_t"));
  }
  return *result;
}

Status validate_native_operation(Operation operation,
                                 std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> outputs) noexcept {
  if (outputs.size() != 1)
    return Status::InvalidArgument("native operation requires one output");

  switch (operation.kind()) {
  case OperationKind::kRmsNorm: {
    Status status = validate_dense_f32(inputs[0]);
    if (!status.ok())
      return status;
    if (inputs.size() == 2) {
      status = validate_dense_f32(inputs[1]);
      if (!status.ok())
        return status;
    }
    return validate_dense_f32(outputs[0]);
  }
  case OperationKind::kMatMul: {
    if (inputs.size() != 2 || inputs[0].rank() != 2 || inputs[1].rank() != 2) {
      return Status::Unimplemented(
          "current native MatMul route requires rank-2 inputs");
    }
    Status status = validate_dense_f32(inputs[0]);
    if (!status.ok())
      return status;
    status = validate_logical_f32(inputs[1]);
    if (!status.ok())
      return status;
    if (!is_supported_matmul_right(inputs[1])) {
      return Status::Unimplemented(
          "HYLE-native MatMul right operand requires plain row-major or "
          "transposed-dense storage");
    }
    return validate_dense_f32(outputs[0]);
  }
  case OperationKind::kGatedActivation: {
    const GatedActivation *gated = operation.get_if<GatedActivation>();
    if (gated == nullptr || gated->activation != GatedActivationKind::kSilu ||
        gated->inputs != GatedInputKind::kPackedHalves || inputs.size() != 1) {
      return Status::Unimplemented(
          "current native gated route supports packed SiLU only");
    }
    Status status = validate_dense_f32(inputs[0]);
    if (!status.ok())
      return status;
    return validate_dense_f32(outputs[0]);
  }
  default:
    return Status::Unimplemented(
        "operation has no current native executable route");
  }
  return Status::Unimplemented("operation kind is unknown");
}

struct ByteRange final {
  mem::MemoryResourceRef resource{};
  void *handle = nullptr;
  std::size_t offset = 0;
  std::size_t size = 0;
};

ByteRange byte_range(mem::ConstBufferView view) noexcept {
  return ByteRange{view.resource(), view.native_handle(), view.offset(),
                   view.size()};
}

ByteRange byte_range(mem::BufferView view) noexcept {
  return ByteRange{view.resource(), view.native_handle(), view.offset(),
                   view.size()};
}

bool same_allocation(const ByteRange &left, const ByteRange &right) noexcept {
  return left.resource == right.resource && left.handle == right.handle;
}

bool equal_range(const ByteRange &left, const ByteRange &right) noexcept {
  return same_allocation(left, right) && left.offset == right.offset &&
         left.size == right.size;
}

bool overlaps(const ByteRange &left, const ByteRange &right) noexcept {
  if (!same_allocation(left, right) || left.size == 0 || right.size == 0) {
    return false;
  }
  if (left.offset <= right.offset) {
    return right.offset - left.offset < left.size;
  }
  return left.offset - right.offset < right.size;
}

bool aligned_for_storage(const void *address, std::size_t alignment) noexcept {
  return address == nullptr ||
         (alignment != 0 &&
          (reinterpret_cast<std::uintptr_t>(address) & (alignment - 1)) == 0);
}

void preserve_first_error(Status &terminal, Status candidate) noexcept {
  if (terminal.ok() && !candidate.ok())
    terminal = std::move(candidate);
}

Result<Completion> completed_failure(Status status) noexcept {
  return Completion::completed(std::move(status));
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

ProviderKind PreparedOperation::provider() const noexcept {
  return state_ != nullptr ? state_->provider : ProviderKind::kUnknown;
}

bool PreparedOperation::used_fallback() const noexcept {
  return state_ != nullptr && state_->fallback;
}

WorkspaceRequirement PreparedOperation::workspace_requirement() const noexcept {
  return state_ != nullptr ? state_->provider_prepared.workspace
                           : WorkspaceRequirement{};
}

std::size_t PreparedOperation::input_count() const noexcept {
  return state_ != nullptr ? state_->input_count : 0;
}

std::size_t PreparedOperation::output_count() const noexcept {
  return state_ != nullptr ? 1 : 0;
}

Result<const TensorDesc &>
PreparedOperation::input_description(std::size_t index) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("prepared operation is invalid"));
  }
  if (index >= state_->input_count) {
    return std::unexpected(
        Status::OutOfRange("prepared input index is out of range"));
  }
  return state_->inputs[index];
}

Result<const TensorDesc &>
PreparedOperation::output_description(std::size_t index) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("prepared operation is invalid"));
  }
  if (index != 0) {
    return std::unexpected(
        Status::OutOfRange("prepared output index is out of range"));
  }
  return state_->output;
}

Result<PreparedOperation>
prepare(const Context &context, Operation operation,
        std::span<const TensorDesc> inputs,
        std::span<const TensorDesc> outputs) noexcept {
  detail::ContextState *state = detail::ContextAccess::state(context);
  if (state == nullptr) {
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  }
  if (!state->open) {
    return std::unexpected(Status::FailedPrecondition("context is closed"));
  }
  if (outputs.size() != 1) {
    return std::unexpected(Status::InvalidArgument(
        "prepared operation requires exactly one output"));
  }

  TensorDesc inferred;
  Status inference = infer_output_descriptions(
      operation, inputs, std::span<TensorDesc>(&inferred, 1));
  if (!inference.ok())
    return std::unexpected(std::move(inference));
  if (outputs[0] != inferred) {
    return std::unexpected(Status::InvalidArgument(
        "operation output description does not match inferred shape"));
  }

  for (std::size_t index = 0; index < state->provider_count; ++index) {
    detail::ContextState::ProviderSlot &slot = state->providers[index];
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
        provider_prepared.vtable->create_invocation == nullptr ||
        provider_prepared.vtable->destroy_invocation == nullptr ||
        provider_prepared.vtable->execute == nullptr ||
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
    void *storage = state->allocator.allocate(sizeof(detail::PreparedState),
                                              alignof(detail::PreparedState));
    if (storage == nullptr) {
      provider_prepared.vtable->destroy_prepared(provider_prepared.state,
                                                 state->allocator);
      return std::unexpected(
          Status::ResourceExhausted("prepared operation allocation failed"));
    }
    auto *prepared = ::new (storage) detail::PreparedState(
        state, operation, inputs, outputs[0], table->kind, false);
    prepared->provider_prepared = provider_prepared;
    return PreparedOperation(prepared);
  }

  Status native_support = validate_native_operation(operation, inputs, outputs);
  if (!native_support.ok())
    return std::unexpected(std::move(native_support));

  Parameters parameters{};
  float scalar = 0.0F;
  Route route = nullptr;
  switch (operation.kind()) {
  case OperationKind::kRmsNorm: {
    auto rows = outer_count(inputs[0]);
    auto hidden = dimension(inputs[0], inputs[0].rank() - 1);
    if (!rows)
      return std::unexpected(std::move(rows).error());
    if (!hidden)
      return std::unexpected(std::move(hidden).error());
    parameters = {*rows, *hidden, 0};
    scalar = operation.get_if<RmsNorm>()->epsilon;
    route = &run_rms_norm;
    break;
  }
  case OperationKind::kMatMul: {
    auto rows = dimension(inputs[0], 0);
    auto inner = dimension(inputs[0], 1);
    auto columns = dimension(inputs[1], 1);
    if (!rows)
      return std::unexpected(std::move(rows).error());
    if (!inner)
      return std::unexpected(std::move(inner).error());
    if (!columns)
      return std::unexpected(std::move(columns).error());
    parameters = {*rows, *inner, *columns};
    route = inputs[1].is_dense() ? select_row_major_matmul()
                                 : &run_matmul_transposed_weight;
    break;
  }
  case OperationKind::kGatedActivation: {
    auto rows = outer_count(inputs[0]);
    auto packed = dimension(inputs[0], inputs[0].rank() - 1);
    if (!rows)
      return std::unexpected(std::move(rows).error());
    if (!packed)
      return std::unexpected(std::move(packed).error());
    parameters = {*rows, *packed / 2, 0};
    route = &run_silu_and_mul;
    break;
  }
  default:
    return std::unexpected(
        Status::Unimplemented("operation has no current native route"));
  }

  void *storage = state->allocator.allocate(sizeof(detail::PreparedState),
                                            alignof(detail::PreparedState));
  if (storage == nullptr) {
    return std::unexpected(
        Status::ResourceExhausted("prepared operation allocation failed"));
  }
  auto *prepared = ::new (storage) detail::PreparedState(
      state, operation, inputs, outputs[0], ProviderKind::kHyleNative,
      state->provider_count != 0);
  prepared->parameters = parameters;
  prepared->scalar = scalar;
  prepared->native_route = route;
  return PreparedOperation(prepared);
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
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("prepared operation is invalid"));
  }
  const WorkspaceRequirement requirement = workspace_requirement();
  if (requirement.size != 0) {
    if (!workspace.valid() || workspace.size() < requirement.size) {
      return std::unexpected(Status::InvalidArgument(
          "operation workspace is absent or smaller than required"));
    }
    const mem::MemoryProperties properties = workspace.resource().properties();
    if (!properties.has(mem::MemoryProperty::kHostAccessible)) {
      return std::unexpected(Status::FailedPrecondition(
          "CPU operation workspace must be host-accessible"));
    }
  }

  void *provider_state = nullptr;
  const detail::ProviderPrepared &provider = state_->provider_prepared;
  if (provider.vtable != nullptr) {
    Status status = provider.vtable->create_invocation(
        provider.state, state_->allocator, workspace, &provider_state);
    if (!status.ok())
      return std::unexpected(std::move(status));
  }
  return OperationInvocation(state_, provider_state);
}

OperationInvocation::OperationInvocation(detail::PreparedState *state,
                                         void *provider_state) noexcept
    : state_(state), provider_state_(provider_state) {
  detail::retain(state_);
}

OperationInvocation::OperationInvocation(OperationInvocation &&other) noexcept
    : state_(other.state_), provider_state_(other.provider_state_) {
  other.state_ = nullptr;
  other.provider_state_ = nullptr;
}

OperationInvocation &
OperationInvocation::operator=(OperationInvocation &&other) noexcept {
  if (this != &other) {
    if (state_ != nullptr && state_->provider_prepared.vtable != nullptr) {
      state_->provider_prepared.vtable->destroy_invocation(provider_state_,
                                                           state_->allocator);
    }
    detail::release(state_);
    state_ = other.state_;
    provider_state_ = other.provider_state_;
    other.state_ = nullptr;
    other.provider_state_ = nullptr;
  }
  return *this;
}

OperationInvocation::~OperationInvocation() {
  if (state_ != nullptr && state_->provider_prepared.vtable != nullptr) {
    state_->provider_prepared.vtable->destroy_invocation(provider_state_,
                                                         state_->allocator);
  }
  detail::release(state_);
}

ProviderKind OperationInvocation::provider() const noexcept {
  return state_ != nullptr ? state_->provider : ProviderKind::kUnknown;
}

Result<Completion>
OperationInvocation::submit(Queue &queue,
                            std::span<const ConstTensorView> inputs,
                            std::span<const TensorView> outputs) noexcept {
  return submit_impl(queue, inputs.empty() ? nullptr : &inputs.front(),
                     inputs.size() > 1 ? &inputs[1] : nullptr, inputs.size(),
                     outputs.empty() ? nullptr : &outputs.front(),
                     outputs.size());
}

Result<Completion> OperationInvocation::submit_impl(
    Queue &queue, const ConstTensorView *first, const ConstTensorView *second,
    std::size_t provided_input_count, const TensorView *output,
    std::size_t provided_output_count) noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("prepared operation is invalid"));
  }
  if (!queue.valid()) {
    return std::unexpected(Status::FailedPrecondition("queue is invalid"));
  }
  if (!queue.open_ || !queue.state_->open) {
    return std::unexpected(Status::FailedPrecondition("queue is closed"));
  }
  if (queue.state_ != state_->context) {
    return std::unexpected(Status::InvalidArgument(
        "queue and operation invocation belong to different contexts"));
  }
  if (provided_input_count != state_->input_count ||
      provided_output_count != 1) {
    return std::unexpected(Status::InvalidArgument(
        "binding count does not match the prepared operation"));
  }
  if (first == nullptr || !first->valid() ||
      first->description() != state_->inputs[0]) {
    return std::unexpected(Status::InvalidArgument(
        "first input binding does not match its prepared description"));
  }
  if (state_->input_count == 2 &&
      (second == nullptr || !second->valid() ||
       second->description() != state_->inputs[1])) {
    return std::unexpected(Status::InvalidArgument(
        "second input binding does not match its prepared description"));
  }
  if (output == nullptr || !output->valid() ||
      output->description() != state_->output) {
    return std::unexpected(Status::InvalidArgument(
        "output binding does not match its prepared description"));
  }

  const mem::ConstBufferView first_storage = first->storage();
  const mem::BufferView output_storage = output->storage();
  const ByteRange first_range = byte_range(first_storage);
  const ByteRange output_range = byte_range(output_storage);
  const bool in_place = state_->operation.kind() == OperationKind::kRmsNorm &&
                        equal_range(first_range, output_range);
  if (overlaps(first_range, output_range) && !in_place) {
    return std::unexpected(Status::InvalidArgument(
        "operation input and output have an unsafe overlap"));
  }

  mem::ConstBufferView second_storage;
  ByteRange second_range{};
  bool shared_inputs = false;
  if (state_->input_count == 2) {
    second_storage = second->storage();
    second_range = byte_range(second_storage);
    if (overlaps(second_range, output_range)) {
      return std::unexpected(Status::InvalidArgument(
          "operation second input and output have an unsafe overlap"));
    }
    shared_inputs = equal_range(first_range, second_range);
  }

  const mem::MemoryProperties first_properties =
      first_storage.resource().properties();
  const mem::MemoryProperties output_properties =
      output_storage.resource().properties();
  mem::MemoryProperties second_properties{};
  if (state_->input_count == 2) {
    second_properties = second_storage.resource().properties();
  }
  if (!first_properties.has(mem::MemoryProperty::kHostAccessible) ||
      !output_properties.has(mem::MemoryProperty::kHostAccessible) ||
      (state_->input_count == 2 &&
       !second_properties.has(mem::MemoryProperty::kHostAccessible))) {
    return std::unexpected(Status::FailedPrecondition(
        "CPU operation requires host-accessible tensor storage"));
  }

  // Admission ends here. Mapping/coherence failures are terminal Completion
  // values, preserving the same boundary future asynchronous providers use.
  if (!first_properties.has(mem::MemoryProperty::kHostDeviceCoherent)) {
    Status status = first_storage.invalidate();
    if (!status.ok())
      return completed_failure(std::move(status));
  }
  if (state_->input_count == 2 && !shared_inputs &&
      !second_properties.has(mem::MemoryProperty::kHostDeviceCoherent)) {
    Status status = second_storage.invalidate();
    if (!status.ok())
      return completed_failure(std::move(status));
  }

  mem::HostMapping first_mapping;
  mem::HostMapping second_mapping;
  mem::HostMapping output_mapping;
  const float *first_data = nullptr;
  const float *second_data = nullptr;
  float *output_data = nullptr;

  if (in_place) {
    auto mapping = output_storage.map_host(mem::HostAccess::kReadWrite);
    if (!mapping)
      return completed_failure(std::move(mapping).error());
    output_mapping = std::move(*mapping);
    auto writable = output_mapping.writable_bytes();
    if (!writable)
      return completed_failure(std::move(writable).error());
    output_data = reinterpret_cast<float *>(writable->data());
    first_data = output_data;
  } else {
    auto mapping = first_storage.map_host();
    if (!mapping)
      return completed_failure(std::move(mapping).error());
    first_mapping = std::move(*mapping);
    first_data = reinterpret_cast<const float *>(first_mapping.bytes().data());
  }

  if (state_->input_count == 2) {
    if (shared_inputs && !in_place) {
      second_data = first_data;
    } else {
      auto mapping = second_storage.map_host();
      if (!mapping) {
        first_mapping.close().IgnoreError();
        output_mapping.close().IgnoreError();
        return completed_failure(std::move(mapping).error());
      }
      second_mapping = std::move(*mapping);
      second_data =
          reinterpret_cast<const float *>(second_mapping.bytes().data());
    }
  }

  if (!in_place) {
    auto mapping = output_storage.map_host(mem::HostAccess::kWrite);
    if (!mapping) {
      second_mapping.close().IgnoreError();
      first_mapping.close().IgnoreError();
      return completed_failure(std::move(mapping).error());
    }
    output_mapping = std::move(*mapping);
    auto writable = output_mapping.writable_bytes();
    if (!writable) {
      output_mapping.close().IgnoreError();
      second_mapping.close().IgnoreError();
      first_mapping.close().IgnoreError();
      return completed_failure(std::move(writable).error());
    }
    output_data = reinterpret_cast<float *>(writable->data());
  }

  const bool first_aligned =
      aligned_for_storage(first_data, state_->inputs[0].required_alignment());
  const bool second_aligned =
      state_->input_count != 2 ||
      aligned_for_storage(second_data, state_->inputs[1].required_alignment());
  const bool output_aligned =
      aligned_for_storage(output_data, state_->output.required_alignment());
  if (!first_aligned || !second_aligned || !output_aligned) {
    output_mapping.close().IgnoreError();
    second_mapping.close().IgnoreError();
    first_mapping.close().IgnoreError();
    return completed_failure(Status::FailedPrecondition(
        "CPU operation tensor mapping does not satisfy its format alignment"));
  }

  Status terminal = Status::Ok();
  if (state_->provider_prepared.vtable != nullptr) {
    const std::array<const void *, 2> provider_inputs{first_data, second_data};
    const std::array<void *, 1> provider_outputs{output_data};
    terminal = state_->provider_prepared.vtable->execute(
        state_->provider_prepared.state, provider_state_,
        std::span<const void *const>(provider_inputs.data(),
                                     state_->input_count),
        provider_outputs);
  } else {
    state_->native_route(first_data, second_data, output_data,
                         state_->parameters, state_->scalar);
  }

  if (terminal.ok() &&
      !output_properties.has(mem::MemoryProperty::kHostDeviceCoherent)) {
    preserve_first_error(terminal, output_mapping.flush());
  }
  preserve_first_error(terminal, output_mapping.close());
  preserve_first_error(terminal, second_mapping.close());
  preserve_first_error(terminal, first_mapping.close());
  return Completion::completed(std::move(terminal));
}

Result<Completion>
OperationInvocation::submit(Queue &queue, const ConstTensorView &input,
                            const TensorView &output) noexcept {
  return submit_impl(queue, &input, nullptr, 1, &output, 1);
}

Result<Completion>
OperationInvocation::submit(Queue &queue, const ConstTensorView &first,
                            const ConstTensorView &second,
                            const TensorView &output) noexcept {
  return submit_impl(queue, &first, &second, 2, &output, 1);
}

} // namespace hyle::compute
