#include "native_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <utility>

#include "hyle/mem/resource.h"
#include "hyle/simd/math.h"
#include "hyle/simd/ops.h"

namespace hyle::compute::native::internal {
namespace {

constexpr float kSqrt2Inv = 0.70710678118654752440F;
constexpr float kSqrt2OverPi = 0.79788456080286535588F;
constexpr float kGeluCoefficient = 0.044715F;

float sigmoid(float value) noexcept {
  if (std::isnan(value))
    return value;
  const float exponential = std::exp(-std::abs(value));
  return value < 0.0F ? exponential / (1.0F + exponential)
                      : 1.0F / (1.0F + exponential);
}

template <UnaryKind Kind> float evaluate_unary(float value) noexcept {
  if constexpr (Kind == UnaryKind::kAbs) {
    return std::fabs(value);
  } else if constexpr (Kind == UnaryKind::kNeg) {
    return -value;
  } else if constexpr (Kind == UnaryKind::kExp) {
    return std::exp(value);
  } else if constexpr (Kind == UnaryKind::kLog) {
    return std::log(value);
  } else if constexpr (Kind == UnaryKind::kSqrt) {
    return std::sqrt(value);
  } else if constexpr (Kind == UnaryKind::kRsqrt) {
    return 1.0F / std::sqrt(value);
  } else if constexpr (Kind == UnaryKind::kTanh) {
    return std::tanh(value);
  } else if constexpr (Kind == UnaryKind::kSigmoid) {
    return sigmoid(value);
  } else if constexpr (Kind == UnaryKind::kRelu) {
    if (std::isnan(value))
      return value;
    return value > 0.0F ? value : 0.0F;
  } else if constexpr (Kind == UnaryKind::kGelu) {
    if (value == -std::numeric_limits<float>::infinity())
      return 0.0F;
    return 0.5F * value * (1.0F + std::erf(value * kSqrt2Inv));
  } else if constexpr (Kind == UnaryKind::kGeluTanh) {
    if (value == -std::numeric_limits<float>::infinity())
      return 0.0F;
    const float inner =
        kSqrt2OverPi * value * std::fma(kGeluCoefficient, value * value, 1.0F);
    return 0.5F * value * (1.0F + std::tanh(inner));
  } else {
    static_assert(Kind == UnaryKind::kSilu);
    if (value == -std::numeric_limits<float>::infinity())
      return 0.0F;
    return value * sigmoid(value);
  }
}

template <UnaryKind Kind>
void unary_linear(const void *first, const void *, const void *, void *output,
                  std::size_t count) noexcept {
  const auto *source = static_cast<const float *>(first);
  auto *destination = static_cast<float *>(output);
  if constexpr (Kind == UnaryKind::kAbs) {
    simd::abs(source, destination, count);
  } else if constexpr (Kind == UnaryKind::kNeg) {
    simd::negate(source, destination, count);
  } else if constexpr (Kind == UnaryKind::kExp) {
    simd::exp_f32_batch(source, destination, count);
  } else if constexpr (Kind == UnaryKind::kLog) {
    simd::log_f32_batch(source, destination, count);
  } else if constexpr (Kind == UnaryKind::kSqrt) {
    simd::sqrt(source, destination, count);
  } else if constexpr (Kind == UnaryKind::kTanh) {
    simd::tanh_f32_batch(source, destination, count);
  } else if constexpr (Kind == UnaryKind::kSigmoid) {
    simd::sigmoid_f32_batch(source, destination, count);
  } else if constexpr (Kind == UnaryKind::kRelu) {
    simd::relu_f32_batch(source, destination, count);
  } else if constexpr (Kind == UnaryKind::kGelu) {
    simd::gelu_f32_batch(source, destination, count);
  } else if constexpr (Kind == UnaryKind::kGeluTanh) {
    simd::gelu_tanh_f32_batch(source, destination, count);
  } else if constexpr (Kind == UnaryKind::kSilu) {
    simd::silu_f32_batch(source, destination, count);
  } else {
    static_assert(Kind == UnaryKind::kRsqrt);
    for (std::size_t index = 0; index < count; ++index)
      destination[index] = evaluate_unary<Kind>(source[index]);
  }
}

template <UnaryKind Kind>
void unary_scalar(const std::byte *first, const std::byte *, const std::byte *,
                  std::byte *output) noexcept {
  float source = 0.0F;
  std::memcpy(&source, first, sizeof(source));
  const float destination = evaluate_unary<Kind>(source);
  std::memcpy(output, &destination, sizeof(destination));
}

template <BinaryKind Kind>
float evaluate_binary(float left, float right) noexcept {
  if constexpr (Kind == BinaryKind::kAdd)
    return left + right;
  else if constexpr (Kind == BinaryKind::kSubtract)
    return left - right;
  else if constexpr (Kind == BinaryKind::kMultiply)
    return left * right;
  else if constexpr (Kind == BinaryKind::kDivide)
    return left / right;
  else if constexpr (Kind == BinaryKind::kMinimum)
    return std::fmin(left, right);
  else {
    static_assert(Kind == BinaryKind::kMaximum);
    return std::fmax(left, right);
  }
}

template <BinaryKind Kind>
void binary_linear(const void *first, const void *second, const void *,
                   void *output, std::size_t count) noexcept {
  const auto *left = static_cast<const float *>(first);
  const auto *right = static_cast<const float *>(second);
  auto *destination = static_cast<float *>(output);
  if constexpr (Kind == BinaryKind::kAdd) {
    simd::add(left, right, destination, count);
  } else if constexpr (Kind == BinaryKind::kSubtract) {
    simd::subtract(left, right, destination, count);
  } else if constexpr (Kind == BinaryKind::kMultiply) {
    simd::multiply(left, right, destination, count);
  } else if constexpr (Kind == BinaryKind::kDivide) {
    simd::divide(left, right, destination, count);
  } else {
    for (std::size_t index = 0; index < count; ++index)
      destination[index] = evaluate_binary<Kind>(left[index], right[index]);
  }
}

template <BinaryKind Kind>
void binary_scalar(const std::byte *first, const std::byte *second,
                   const std::byte *, std::byte *output) noexcept {
  float left = 0.0F;
  float right = 0.0F;
  std::memcpy(&left, first, sizeof(left));
  std::memcpy(&right, second, sizeof(right));
  const float destination = evaluate_binary<Kind>(left, right);
  std::memcpy(output, &destination, sizeof(destination));
}

template <std::size_t ElementSize>
void select_linear(const void *first, const void *second, const void *third,
                   void *output, std::size_t count) noexcept {
  const auto *condition = static_cast<const std::byte *>(first);
  const auto *when_true = static_cast<const std::byte *>(second);
  const auto *when_false = static_cast<const std::byte *>(third);
  auto *destination = static_cast<std::byte *>(output);
  for (std::size_t index = 0; index < count; ++index) {
    const std::byte *source = condition[index] != std::byte{0}
                                  ? when_true + index * ElementSize
                                  : when_false + index * ElementSize;
    std::byte *target = destination + index * ElementSize;
    if (source != target)
      std::memcpy(target, source, ElementSize);
  }
}

template <std::size_t ElementSize>
void select_scalar(const std::byte *first, const std::byte *second,
                   const std::byte *third, std::byte *output) noexcept {
  const std::byte *source = *first != std::byte{0} ? second : third;
  if (source != output)
    std::memcpy(output, source, ElementSize);
}

template <class Left, class Right>
bool same_storage_range(const Left &left, const Right &right) noexcept {
  return left.resource() == right.resource() &&
         left.native_handle() == right.native_handle() &&
         left.offset() == right.offset() && left.size() == right.size();
}

void close_ignoring_errors(std::array<mem::HostMapping, 3> &mappings,
                           mem::HostMapping &output) noexcept {
  output.close().IgnoreError();
  for (mem::HostMapping &mapping : mappings)
    mapping.close().IgnoreError();
}

Result<Completion>
submit_elementwise(const NativePlan &plan,
                   std::span<const ConstTensorView> inputs,
                   std::span<const TensorView> outputs) noexcept {
  if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1 ||
      plan.linear_elementwise_route == nullptr ||
      plan.scalar_elementwise_route == nullptr) {
    return completed(Status::Internal("native elementwise plan is invalid"));
  }
  const std::size_t elements = plan.parameters[0];
  if (elements == 0)
    return completed(Status::Ok());

  const mem::BufferView output_storage = outputs[0].storage();
  bool aliases_output = false;
  for (const ConstTensorView &input : inputs)
    aliases_output =
        aliases_output || same_storage_range(input.storage(), output_storage);
  if (aliases_output && plan.parameters[1] == 0) {
    return completed(Status::InvalidArgument(
        "broadcast or strided elementwise input cannot alias its output"));
  }

  for (std::size_t index = 0; index < inputs.size(); ++index) {
    bool already_invalidated = false;
    for (std::size_t prior = 0; prior < index; ++prior) {
      already_invalidated =
          already_invalidated ||
          same_storage_range(inputs[index].storage(), inputs[prior].storage());
    }
    const mem::ConstBufferView storage = inputs[index].storage();
    if (!already_invalidated && !storage.resource().properties().has(
                                    mem::MemoryProperty::kHostDeviceCoherent)) {
      Status status = storage.invalidate();
      if (!status.ok())
        return completed(std::move(status));
    }
  }

  std::array<mem::HostMapping, 3> input_mappings{};
  std::array<const std::byte *, 3> source_data{};
  mem::HostMapping output_mapping;
  std::byte *output_data = nullptr;

  if (aliases_output) {
    auto mapping = output_storage.map_host(mem::HostAccess::kReadWrite);
    if (!mapping)
      return completed(std::move(mapping).error());
    output_mapping = std::move(*mapping);
    auto writable = output_mapping.writable_bytes();
    if (!writable) {
      output_mapping.close().IgnoreError();
      return completed(std::move(writable).error());
    }
    output_data = writable->data();
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      if (same_storage_range(inputs[index].storage(), output_storage))
        source_data[index] = output_data;
    }
  }

  for (std::size_t index = 0; index < inputs.size(); ++index) {
    if (source_data[index] != nullptr)
      continue;
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (same_storage_range(inputs[index].storage(),
                             inputs[prior].storage())) {
        source_data[index] = source_data[prior];
        break;
      }
    }
    if (source_data[index] != nullptr)
      continue;
    auto mapping = inputs[index].storage().map_host();
    if (!mapping) {
      close_ignoring_errors(input_mappings, output_mapping);
      return completed(std::move(mapping).error());
    }
    input_mappings[index] = std::move(*mapping);
    source_data[index] = input_mappings[index].bytes().data();
  }

  if (!aliases_output) {
    auto mapping = output_storage.map_host(mem::HostAccess::kWrite);
    if (!mapping) {
      close_ignoring_errors(input_mappings, output_mapping);
      return completed(std::move(mapping).error());
    }
    output_mapping = std::move(*mapping);
    auto writable = output_mapping.writable_bytes();
    if (!writable) {
      close_ignoring_errors(input_mappings, output_mapping);
      return completed(std::move(writable).error());
    }
    output_data = writable->data();
  }

  bool aligned = aligned_for_storage(
      output_data, outputs[0].description().required_alignment());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    aligned = aligned && aligned_for_storage(
                             source_data[index],
                             inputs[index].description().required_alignment());
  }
  if (!aligned) {
    close_ignoring_errors(input_mappings, output_mapping);
    return completed(Status::FailedPrecondition(
        "CPU elementwise mapping does not satisfy tensor alignment"));
  }

  const void *second = inputs.size() > 1 ? source_data[1] : nullptr;
  const void *third = inputs.size() > 2 ? source_data[2] : nullptr;
  if (plan.parameters[1] != 0) {
    plan.linear_elementwise_route(source_data[0], second, third, output_data,
                                  elements);
  } else {
    const auto output_dimensions = outputs[0].description().dimensions();
    const std::size_t output_element_size = plan.parameters[2];
    for (std::size_t linear = 0; linear < elements; ++linear) {
      std::array<const std::byte *, 3> logical{};
      for (std::size_t index = 0; index < inputs.size(); ++index) {
        logical[index] = source_data[index] + logical_broadcast_byte_offset(
                                                  inputs[index].description(),
                                                  output_dimensions, linear);
      }
      plan.scalar_elementwise_route(logical[0],
                                    inputs.size() > 1 ? logical[1] : nullptr,
                                    inputs.size() > 2 ? logical[2] : nullptr,
                                    output_data + linear * output_element_size);
    }
  }

  Status terminal = Status::Ok();
  if (!output_storage.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    preserve_first_error(terminal, output_mapping.flush());
  }
  preserve_first_error(terminal, output_mapping.close());
  for (mem::HostMapping &mapping : input_mappings)
    preserve_first_error(terminal, mapping.close());
  return completed(std::move(terminal));
}

bool select_unary(UnaryKind kind, NativePlan &plan) noexcept {
#define HYLE_NATIVE_SELECT_UNARY(name)                                         \
  case UnaryKind::name:                                                        \
    plan.linear_elementwise_route = &unary_linear<UnaryKind::name>;            \
    plan.scalar_elementwise_route = &unary_scalar<UnaryKind::name>;            \
    return true

  switch (kind) {
    HYLE_NATIVE_SELECT_UNARY(kAbs);
    HYLE_NATIVE_SELECT_UNARY(kNeg);
    HYLE_NATIVE_SELECT_UNARY(kExp);
    HYLE_NATIVE_SELECT_UNARY(kLog);
    HYLE_NATIVE_SELECT_UNARY(kSqrt);
    HYLE_NATIVE_SELECT_UNARY(kRsqrt);
    HYLE_NATIVE_SELECT_UNARY(kTanh);
    HYLE_NATIVE_SELECT_UNARY(kSigmoid);
    HYLE_NATIVE_SELECT_UNARY(kRelu);
    HYLE_NATIVE_SELECT_UNARY(kGelu);
    HYLE_NATIVE_SELECT_UNARY(kGeluTanh);
    HYLE_NATIVE_SELECT_UNARY(kSilu);
  }
  return false;

#undef HYLE_NATIVE_SELECT_UNARY
}

bool select_binary(BinaryKind kind, NativePlan &plan) noexcept {
#define HYLE_NATIVE_SELECT_BINARY(name)                                        \
  case BinaryKind::name:                                                       \
    plan.linear_elementwise_route = &binary_linear<BinaryKind::name>;          \
    plan.scalar_elementwise_route = &binary_scalar<BinaryKind::name>;          \
    return true

  switch (kind) {
    HYLE_NATIVE_SELECT_BINARY(kAdd);
    HYLE_NATIVE_SELECT_BINARY(kSubtract);
    HYLE_NATIVE_SELECT_BINARY(kMultiply);
    HYLE_NATIVE_SELECT_BINARY(kDivide);
    HYLE_NATIVE_SELECT_BINARY(kMinimum);
    HYLE_NATIVE_SELECT_BINARY(kMaximum);
  }
  return false;

#undef HYLE_NATIVE_SELECT_BINARY
}

bool select_select(std::size_t element_size, NativePlan &plan) noexcept {
#define HYLE_NATIVE_SELECT_WIDTH(width)                                        \
  case width:                                                                  \
    plan.linear_elementwise_route = &select_linear<width>;                     \
    plan.scalar_elementwise_route = &select_scalar<width>;                     \
    return true

  switch (element_size) {
    HYLE_NATIVE_SELECT_WIDTH(1);
    HYLE_NATIVE_SELECT_WIDTH(2);
    HYLE_NATIVE_SELECT_WIDTH(4);
    HYLE_NATIVE_SELECT_WIDTH(8);
  default:
    return false;
  }

#undef HYLE_NATIVE_SELECT_WIDTH
}

bool dense_same_shape(const TensorDesc &input,
                      const TensorDesc &output) noexcept {
  return input.is_dense() &&
         std::ranges::equal(input.dimensions(), output.dimensions());
}

Status validate_plain_f32(const TensorDesc &description) noexcept {
  if (description.data_type() != DataType::kFloat32 ||
      !description.is_plain()) {
    return Status::Unimplemented(
        "native elementwise arithmetic currently requires plain float32");
  }
  return Status::Ok();
}

} // namespace

Status prepare_elementwise(Operation operation,
                           std::span<const TensorDesc> inputs,
                           std::span<const TensorDesc> outputs,
                           NativePlan &plan) noexcept {
  if (outputs.size() != 1 || !outputs[0].is_plain() || !outputs[0].is_dense()) {
    return Status::Unimplemented(
        "native elementwise output must be dense plain storage");
  }

  bool linear = false;
  switch (operation.kind()) {
  case OperationKind::kUnary: {
    if (inputs.size() != 1)
      return Status::Internal("native Unary arity is invalid");
    Status status = validate_plain_f32(inputs[0]);
    if (!status.ok())
      return status;
    status = validate_plain_f32(outputs[0]);
    if (!status.ok())
      return status;
    const Unary *unary = operation.get_if<Unary>();
    if (unary == nullptr || !select_unary(unary->kind, plan))
      return Status::InvalidArgument("native Unary kind is invalid");
    linear = dense_same_shape(inputs[0], outputs[0]);
    break;
  }
  case OperationKind::kBinary: {
    if (inputs.size() != 2)
      return Status::Internal("native Binary arity is invalid");
    for (const TensorDesc &input : inputs) {
      Status status = validate_plain_f32(input);
      if (!status.ok())
        return status;
    }
    Status status = validate_plain_f32(outputs[0]);
    if (!status.ok())
      return status;
    const Binary *binary = operation.get_if<Binary>();
    if (binary == nullptr || !select_binary(binary->kind, plan))
      return Status::InvalidArgument("native Binary kind is invalid");
    linear = dense_same_shape(inputs[0], outputs[0]) &&
             dense_same_shape(inputs[1], outputs[0]);
    break;
  }
  case OperationKind::kSelect: {
    if (inputs.size() != 3)
      return Status::Internal("native Select arity is invalid");
    if (inputs[0].data_type() != DataType::kBool || !inputs[0].is_plain() ||
        !inputs[1].is_plain() || !inputs[2].is_plain()) {
      return Status::Unimplemented(
          "native Select requires plain Bool condition and branch storage");
    }
    const std::size_t element_size = data_type_size(outputs[0].data_type());
    if (!select_select(element_size, plan)) {
      return Status::Unimplemented(
          "native Select branch scalar width is unsupported");
    }
    linear = dense_same_shape(inputs[0], outputs[0]) &&
             dense_same_shape(inputs[1], outputs[0]) &&
             dense_same_shape(inputs[2], outputs[0]);
    break;
  }
  default:
    return Status::Unimplemented("operation is not native elementwise");
  }

  auto elements = outputs[0].element_count();
  if (!elements)
    return std::move(elements).error();
  plan.parameters[0] = *elements;
  plan.parameters[1] = linear ? 1 : 0;
  plan.parameters[2] = data_type_size(outputs[0].data_type());
  plan.submit_route = &submit_elementwise;
  return Status::Ok();
}

} // namespace hyle::compute::native::internal
