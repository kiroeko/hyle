#include "hyle/compute/operation.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

#include "context_state.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/math.h"

namespace hyle::compute {
namespace {

void run_rms_norm(const float* input, const float*, float* output,
                  std::size_t count, float epsilon) noexcept {
  simd::rms_norm_f32(input, output, count, epsilon);
}

void run_weighted_rms_norm(const float* input, const float* weight,
                           float* output, std::size_t count,
                           float epsilon) noexcept {
  simd::rms_norm_f32(input, weight, output, count, epsilon);
}

Status validate_rms_tensor(const TensorDesc& description,
                           std::string_view role) noexcept {
  if (!description.valid()) {
    return Status::InvalidArgument(role == "output"
                                       ? "RMSNorm output is invalid"
                                       : "RMSNorm input is invalid");
  }
  if (description.data_type() != DataType::kFloat32) {
    return Status::Unimplemented("RMSNorm v1 requires float32 tensors");
  }
  if (description.rank() != 1) {
    return Status::Unimplemented("RMSNorm v1 requires one-dimensional rows");
  }
  if (!description.is_dense()) {
    return Status::Unimplemented("RMSNorm v1 requires dense tensors");
  }
  return Status::Ok();
}

struct ByteRange final {
  mem::MemoryResourceRef resource{};
  void* handle = nullptr;
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

bool same_allocation(const ByteRange& left, const ByteRange& right) noexcept {
  return left.resource == right.resource && left.handle == right.handle;
}

bool equal_range(const ByteRange& left, const ByteRange& right) noexcept {
  return same_allocation(left, right) && left.offset == right.offset &&
         left.size == right.size;
}

bool overlaps(const ByteRange& left, const ByteRange& right) noexcept {
  if (!same_allocation(left, right) || left.size == 0 || right.size == 0) {
    return false;
  }
  if (left.offset <= right.offset) {
    return right.offset - left.offset < left.size;
  }
  return left.offset - right.offset < right.size;
}

bool host_accessible(mem::ConstBufferView view) noexcept {
  return view.resource().properties().has(mem::MemoryProperty::kHostAccessible);
}

bool coherent(mem::ConstBufferView view) noexcept {
  return view.resource().properties().has(
      mem::MemoryProperty::kHostDeviceCoherent);
}

bool aligned_for_float(const void* address) noexcept {
  return address == nullptr ||
         reinterpret_cast<std::uintptr_t>(address) % alignof(float) == 0;
}

void preserve_first_error(Status& terminal, Status candidate) noexcept {
  if (terminal.ok() && !candidate.ok()) terminal = std::move(candidate);
}

Result<Completion> completed_failure(Status status) noexcept {
  return Completion::completed(std::move(status));
}

}  // namespace

std::string_view provider_name(ProviderKind provider) noexcept {
  switch (provider) {
    case ProviderKind::kHyleNative:
      return "hyle-native";
  }
  return "unknown";
}

std::string_view operation_name(OperationKind operation) noexcept {
  switch (operation) {
    case OperationKind::kRmsNorm:
      return "rms-norm";
  }
  return "unknown";
}

PreparedOperation::PreparedOperation(
    detail::ContextState* state, Operation operation,
    std::span<const TensorDesc> inputs, TensorDesc output) noexcept
    : state_(state), operation_(operation), output_(output),
      element_count_(static_cast<std::size_t>(
          inputs.front().dimensions().front())),
      input_count_(static_cast<std::uint8_t>(inputs.size())),
      route_(inputs.size() == 1 ? &run_rms_norm : &run_weighted_rms_norm) {
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    inputs_[index] = inputs[index];
  }
  detail::retain(state_);
}

PreparedOperation::PreparedOperation(PreparedOperation&& other) noexcept
    : state_(other.state_), operation_(other.operation_), inputs_(other.inputs_),
      output_(other.output_), element_count_(other.element_count_),
      input_count_(other.input_count_),
      route_(other.route_) {
  other.state_ = nullptr;
  other.element_count_ = 0;
  other.input_count_ = 0;
  other.route_ = nullptr;
}

PreparedOperation& PreparedOperation::operator=(
    PreparedOperation&& other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    operation_ = other.operation_;
    inputs_ = other.inputs_;
    output_ = other.output_;
    element_count_ = other.element_count_;
    input_count_ = other.input_count_;
    route_ = other.route_;
    other.state_ = nullptr;
    other.element_count_ = 0;
    other.input_count_ = 0;
    other.route_ = nullptr;
  }
  return *this;
}

PreparedOperation::~PreparedOperation() { detail::release(state_); }

Result<const TensorDesc&> PreparedOperation::input_description(
    std::size_t index) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("prepared operation is invalid"));
  }
  if (index >= input_count_) {
    return std::unexpected(
        Status::OutOfRange("prepared input index is out of range"));
  }
  return inputs_[index];
}

Result<const TensorDesc&> PreparedOperation::output_description(
    std::size_t index) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("prepared operation is invalid"));
  }
  if (index != 0) {
    return std::unexpected(
        Status::OutOfRange("prepared output index is out of range"));
  }
  return output_;
}

Result<PreparedOperation> prepare(
    const Context& context, Operation operation,
    std::span<const TensorDesc> inputs,
    std::span<const TensorDesc> outputs) noexcept {
  detail::ContextState* state = detail::ContextAccess::state(context);
  if (state == nullptr) {
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  }
  if (!state->open) {
    return std::unexpected(Status::FailedPrecondition("context is closed"));
  }
  if (inputs.size() != 1 && inputs.size() != 2) {
    return std::unexpected(
        Status::InvalidArgument("RMSNorm requires one input and optional weight"));
  }
  if (outputs.size() != 1) {
    return std::unexpected(
        Status::InvalidArgument("RMSNorm requires exactly one output"));
  }
  if (!std::isfinite(operation.rms_norm().epsilon) ||
      operation.rms_norm().epsilon < 0.0F) {
    return std::unexpected(Status::InvalidArgument(
        "RMSNorm epsilon must be finite and non-negative"));
  }

  for (const TensorDesc& input : inputs) {
    Status status = validate_rms_tensor(input, "input");
    if (!status.ok()) return std::unexpected(std::move(status));
  }
  Status output_status = validate_rms_tensor(outputs.front(), "output");
  if (!output_status.ok()) {
    return std::unexpected(std::move(output_status));
  }
  if (inputs.front() != outputs.front()) {
    return std::unexpected(Status::InvalidArgument(
        "RMSNorm input and output descriptions must match"));
  }
  if (inputs.size() == 2 && inputs[1] != inputs.front()) {
    return std::unexpected(Status::InvalidArgument(
        "RMSNorm weight description must match the input"));
  }
  return PreparedOperation(state, operation, inputs, outputs.front());
}

Result<PreparedOperation> prepare(const Context& context, RmsNorm operation,
                                  TensorDesc input,
                                  TensorDesc output) noexcept {
  const std::array inputs{input};
  const std::array outputs{output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<PreparedOperation> prepare(const Context& context, RmsNorm operation,
                                  TensorDesc input, TensorDesc weight,
                                  TensorDesc output) noexcept {
  const std::array inputs{input, weight};
  const std::array outputs{output};
  return prepare(context, Operation(operation), inputs, outputs);
}

Result<Completion> PreparedOperation::submit(
    Queue& queue, std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs) const noexcept {
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
  if (queue.state_ != state_) {
    return std::unexpected(Status::InvalidArgument(
        "queue and prepared operation belong to different contexts"));
  }
  if (inputs.size() != input_count_ || outputs.size() != 1) {
    return std::unexpected(Status::InvalidArgument(
        "RMSNorm binding count does not match the prepared operation"));
  }
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    if (!inputs[index].valid() ||
        inputs[index].description() != inputs_[index]) {
      return std::unexpected(Status::InvalidArgument(
          "RMSNorm input binding does not match its prepared description"));
    }
  }
  if (!outputs.front().valid() ||
      outputs.front().description() != output_) {
    return std::unexpected(Status::InvalidArgument(
        "RMSNorm output binding does not match its prepared description"));
  }

  const mem::ConstBufferView input_storage = inputs.front().storage();
  const mem::BufferView output_storage = outputs.front().storage();
  const ByteRange input_range = byte_range(input_storage);
  const ByteRange output_range = byte_range(output_storage);
  const bool in_place = equal_range(input_range, output_range);
  if (overlaps(input_range, output_range) && !in_place) {
    return std::unexpected(Status::InvalidArgument(
        "RMSNorm input and output may only overlap by exact in-place alias"));
  }

  mem::ConstBufferView weight_storage;
  if (input_count_ == 2) {
    weight_storage = inputs[1].storage();
    if (overlaps(byte_range(weight_storage), output_range)) {
      return std::unexpected(Status::InvalidArgument(
          "RMSNorm weight and output storage must not overlap"));
    }
  }
  if (!host_accessible(input_storage) || !host_accessible(output_storage) ||
      (input_count_ == 2 && !host_accessible(weight_storage))) {
    return std::unexpected(Status::FailedPrecondition(
        "CPU RMSNorm requires host-accessible tensor storage"));
  }

  // The submission is admitted after every cheap contract check. Provider
  // map/coherence failures below are terminal Completion results, matching the
  // same lifecycle boundary future asynchronous backends use.
  if (!coherent(input_storage)) {
    Status status = input_storage.invalidate();
    if (!status.ok()) return completed_failure(std::move(status));
  }
  if (input_count_ == 2 && !coherent(weight_storage)) {
    Status status = weight_storage.invalidate();
    if (!status.ok()) return completed_failure(std::move(status));
  }

  mem::HostMapping input_mapping;
  mem::HostMapping weight_mapping;
  mem::HostMapping output_mapping;
  const float* input_data = nullptr;
  const float* weight_data = nullptr;
  float* output_data = nullptr;

  if (in_place) {
    auto mapping = output_storage.map_host(mem::HostAccess::kReadWrite);
    if (!mapping) return completed_failure(std::move(mapping).error());
    output_mapping = std::move(*mapping);
    auto writable = output_mapping.writable_bytes();
    if (!writable) return completed_failure(std::move(writable).error());
    output_data = reinterpret_cast<float*>(writable->data());
    input_data = output_data;
  } else {
    auto mapping = input_storage.map_host();
    if (!mapping) return completed_failure(std::move(mapping).error());
    input_mapping = std::move(*mapping);
    input_data = reinterpret_cast<const float*>(input_mapping.bytes().data());
  }

  if (input_count_ == 2) {
    auto mapping = weight_storage.map_host();
    if (!mapping) {
      input_mapping.close().IgnoreError();
      output_mapping.close().IgnoreError();
      return completed_failure(std::move(mapping).error());
    }
    weight_mapping = std::move(*mapping);
    weight_data =
        reinterpret_cast<const float*>(weight_mapping.bytes().data());
  }

  if (!in_place) {
    auto mapping = output_storage.map_host(mem::HostAccess::kWrite);
    if (!mapping) {
      weight_mapping.close().IgnoreError();
      input_mapping.close().IgnoreError();
      return completed_failure(std::move(mapping).error());
    }
    output_mapping = std::move(*mapping);
    auto writable = output_mapping.writable_bytes();
    if (!writable) {
      output_mapping.close().IgnoreError();
      weight_mapping.close().IgnoreError();
      input_mapping.close().IgnoreError();
      return completed_failure(std::move(writable).error());
    }
    output_data = reinterpret_cast<float*>(writable->data());
  }

  if (!aligned_for_float(input_data) || !aligned_for_float(weight_data) ||
      !aligned_for_float(output_data)) {
    output_mapping.close().IgnoreError();
    weight_mapping.close().IgnoreError();
    input_mapping.close().IgnoreError();
    return completed_failure(Status::FailedPrecondition(
        "CPU RMSNorm tensor mapping is not aligned for float32"));
  }

  route_(input_data, weight_data, output_data, element_count_,
         operation_.rms_norm().epsilon);

  Status terminal = Status::Ok();
  if (!coherent(output_storage)) {
    preserve_first_error(terminal, output_mapping.flush());
  }
  preserve_first_error(terminal, output_mapping.close());
  preserve_first_error(terminal, weight_mapping.close());
  preserve_first_error(terminal, input_mapping.close());
  return Completion::completed(std::move(terminal));
}

Result<Completion> PreparedOperation::submit(Queue& queue,
                                              ConstTensorView input,
                                              TensorView output) const noexcept {
  const std::array inputs{input};
  const std::array outputs{output};
  return submit(queue, inputs, outputs);
}

Result<Completion> PreparedOperation::submit(
    Queue& queue, ConstTensorView input, ConstTensorView weight,
    TensorView output) const noexcept {
  const std::array inputs{input, weight};
  const std::array outputs{output};
  return submit(queue, inputs, outputs);
}

}  // namespace hyle::compute
