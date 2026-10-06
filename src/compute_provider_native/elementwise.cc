#include "native_plan.h"
#include "numeric.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/math/scalar.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/math.h"
#include "hyle/simd/ops.h"

namespace hyle::compute::native::internal {
namespace {

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
  } else if constexpr (Kind == UnaryKind::kSoftplus) {
    return std::fmax(value, 0.0F) + std::log1p(std::exp(-std::abs(value)));
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
  } else if constexpr (Kind == UnaryKind::kSilu) {
    simd::silu_f32_batch(source, destination, count);
  } else if constexpr (Kind == UnaryKind::kSoftplus) {
    // Reuse the qualified math kernels in bounded stack tiles. Capturing the
    // exponential before writing the result also preserves exact input alias.
    std::array<float, 128> scratch;
    for (std::size_t begin = 0; begin < count; begin += scratch.size()) {
      const std::size_t size = (std::min)(scratch.size(), count - begin);
      simd::abs(source + begin, scratch.data(), size);
      simd::negate(scratch.data(), scratch.data(), size);
      simd::exp_f32_batch(scratch.data(), scratch.data(), size);
      simd::log1p_f32_batch(scratch.data(), scratch.data(), size);
      for (std::size_t index = 0; index < size; ++index)
        destination[begin + index] =
            std::fmax(source[begin + index], 0.0F) + scratch[index];
    }
  } else {
    static_assert(Kind == UnaryKind::kRsqrt);
    simd::reciprocal_sqrt(source, destination, count);
  }
}

template <UnaryKind Kind, class Value>
void unary_narrow_linear(const void *first, const void *, const void *,
                         void *output, std::size_t count) noexcept {
  std::array<float, 128> values;
  for (std::size_t begin = 0; begin < count; begin += values.size()) {
    const std::size_t size = (std::min)(values.size(), count - begin);
    numeric::load_tile<Value>(first, begin, values.data(), size);
    unary_linear<Kind>(values.data(), nullptr, nullptr, values.data(), size);
    numeric::store_tile<Value>(values.data(), output, begin, size);
  }
}

template <UnaryKind Kind, class Value>
void unary_scalar(const std::byte *first, const std::byte *, const std::byte *,
                  std::byte *output) noexcept {
  numeric::store<Value>(output,
                        evaluate_unary<Kind>(numeric::load<Value>(first)));
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
  } else if constexpr (Kind == BinaryKind::kMinimum) {
    simd::fmin(left, right, destination, count);
  } else {
    static_assert(Kind == BinaryKind::kMaximum);
    simd::fmax(left, right, destination, count);
  }
}

template <BinaryKind Kind, class Value>
void binary_narrow_linear(const void *first, const void *second, const void *,
                          void *output, std::size_t count) noexcept {
  std::array<float, 128> left;
  std::array<float, 128> right;
  for (std::size_t begin = 0; begin < count; begin += left.size()) {
    const std::size_t size = (std::min)(left.size(), count - begin);
    numeric::load_tile<Value>(first, begin, left.data(), size);
    numeric::load_tile<Value>(second, begin, right.data(), size);
    binary_linear<Kind>(left.data(), right.data(), nullptr, left.data(), size);
    numeric::store_tile<Value>(left.data(), output, begin, size);
  }
}

template <BinaryKind Kind, class Value>
void binary_scalar(const std::byte *first, const std::byte *second,
                   const std::byte *, std::byte *output) noexcept {
  numeric::store<Value>(output,
                        evaluate_binary<Kind>(numeric::load<Value>(first),
                                              numeric::load<Value>(second)));
}

template <class Value>
void gated_narrow_linear(const void *first, const void *second, const void *,
                         void *output, std::size_t count) noexcept {
  std::array<float, 128> gate;
  std::array<float, 128> value;
  for (std::size_t begin = 0; begin < count; begin += gate.size()) {
    const std::size_t size = (std::min)(gate.size(), count - begin);
    numeric::load_tile<Value>(first, begin, gate.data(), size);
    numeric::load_tile<Value>(second, begin, value.data(), size);
    simd::swiglu_f32_batch(gate.data(), value.data(), gate.data(), size);
    numeric::store_tile<Value>(gate.data(), output, begin, size);
  }
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

void select_f32_linear(const void *first, const void *second, const void *third,
                       void *output, std::size_t count) noexcept {
  simd::select_values(static_cast<const std::uint8_t *>(first),
                      static_cast<const float *>(second),
                      static_cast<const float *>(third),
                      static_cast<float *>(output), count);
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

struct GatedEpisode final {
  const NativePlan *plan = nullptr;
  const std::byte *first = nullptr;
  const std::byte *second = nullptr;
  std::byte *output = nullptr;
};

void run_gated_range(void *context, std::size_t, std::size_t begin,
                     std::size_t end) noexcept {
  const auto &episode = *static_cast<const GatedEpisode *>(context);
  const NativePlan &plan = *episode.plan;
  const std::size_t hidden = plan.parameters[1];
  const std::size_t element_size = plan.parameters[2];
  for (std::size_t row = begin; row < end; ++row) {
    const std::byte *gate =
        episode.first + row * plan.row_first_stride * element_size;
    const std::byte *value = episode.second != nullptr
                                 ? episode.second + row * hidden * element_size
                                 : gate + hidden * element_size;
    plan.linear_elementwise_route(gate, value, nullptr,
                                  episode.output + row * hidden * element_size,
                                  hidden);
  }
}

template <bool Gated = false>
detail::EnqueueResult
submit_elementwise(const NativePlan &plan,
                   std::span<const ConstTensorView> inputs,
                   std::span<const TensorView> outputs) noexcept {
  if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1 ||
      plan.linear_elementwise_route == nullptr ||
      (!Gated && plan.scalar_elementwise_route == nullptr)) {
    return completed(Status::Internal("native elementwise plan is invalid"));
  }
  const std::size_t elements =
      plan.parameters[0] * (Gated ? plan.parameters[1] : std::size_t{1});
  if (elements == 0)
    return completed(Status::Ok());

  const mem::BufferView output_storage = outputs[0].storage();
  bool aliases_output = false;
  for (const ConstTensorView &input : inputs)
    aliases_output =
        aliases_output || same_storage_range(input.storage(), output_storage);
  if (aliases_output) {
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      if (same_storage_range(inputs[index].storage(), output_storage) &&
          (plan.alias_input_mask & (std::uint8_t{1} << index)) == 0) {
        return completed(Status::InvalidArgument(
            "broadcast or strided elementwise input cannot alias its output"));
      }
    }
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

  if constexpr (Gated) {
    GatedEpisode episode{.plan = &plan,
                         .first = source_data[0],
                         .second = inputs.size() > 1 ? source_data[1] : nullptr,
                         .output = output_data};
    parallel_for_ranges(plan, plan.parameters[0], &run_gated_range, &episode);
  } else {
    const std::size_t linear_chunk = plan.parameters[1];
    if (linear_chunk != 0) {
      const auto output_dimensions = outputs[0].description().dimensions();
      for (std::size_t base = 0; base < elements; base += linear_chunk) {
        std::array<const std::byte *, 3> logical{};
        for (std::size_t index = 0; index < inputs.size(); ++index) {
          logical[index] = source_data[index] + logical_broadcast_byte_offset(
                                                    inputs[index].description(),
                                                    output_dimensions, base);
        }
        plan.linear_elementwise_route(
            logical[0], inputs.size() > 1 ? logical[1] : nullptr,
            inputs.size() > 2 ? logical[2] : nullptr,
            output_data + base * plan.parameters[2], linear_chunk);
      }
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
        plan.scalar_elementwise_route(
            logical[0], inputs.size() > 1 ? logical[1] : nullptr,
            inputs.size() > 2 ? logical[2] : nullptr,
            output_data + linear * output_element_size);
      }
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

template <class Value>
bool select_unary(UnaryKind kind, NativePlan &plan) noexcept {
  if constexpr (std::is_same_v<Value, float>)
    if (kind == UnaryKind::kSilu)
      return false;
#define HYLE_NATIVE_SELECT_UNARY(name)                                         \
  case UnaryKind::name:                                                        \
    if constexpr (std::is_same_v<Value, float>)                                \
      plan.linear_elementwise_route = &unary_linear<UnaryKind::name>;          \
    else                                                                       \
      plan.linear_elementwise_route =                                          \
          &unary_narrow_linear<UnaryKind::name, Value>;                        \
    plan.scalar_elementwise_route = &unary_scalar<UnaryKind::name, Value>;     \
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
    HYLE_NATIVE_SELECT_UNARY(kSilu);
    HYLE_NATIVE_SELECT_UNARY(kSoftplus);
  case UnaryKind::kGelu:
  case UnaryKind::kGeluTanh:
    return false;
  }
  return false;

#undef HYLE_NATIVE_SELECT_UNARY
}

template <class Value>
bool select_binary(BinaryKind kind, NativePlan &plan) noexcept {
#define HYLE_NATIVE_SELECT_BINARY(name)                                        \
  case BinaryKind::name:                                                       \
    if constexpr (std::is_same_v<Value, float>)                                \
      plan.linear_elementwise_route = &binary_linear<BinaryKind::name>;        \
    else                                                                       \
      plan.linear_elementwise_route =                                          \
          &binary_narrow_linear<BinaryKind::name, Value>;                      \
    plan.scalar_elementwise_route = &binary_scalar<BinaryKind::name, Value>;   \
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

bool select_unary(UnaryKind kind, DataType type, NativePlan &plan) noexcept {
  switch (type) {
  case DataType::kFloat32:
    return select_unary<float>(kind, plan);
  case DataType::kFloat16:
    return select_unary<simd::float16_t>(kind, plan);
  case DataType::kBFloat16:
    return select_unary<simd::bfloat16_t>(kind, plan);
  default:
    return false;
  }
}

bool select_binary(BinaryKind kind, DataType type, NativePlan &plan) noexcept {
  switch (type) {
  case DataType::kFloat32:
    return select_binary<float>(kind, plan);
  case DataType::kFloat16:
    return select_binary<simd::float16_t>(kind, plan);
  case DataType::kBFloat16:
    return select_binary<simd::bfloat16_t>(kind, plan);
  default:
    return false;
  }
}

bool select_select(DataType data_type, std::size_t element_size,
                   NativePlan &plan) noexcept {
  if (data_type == DataType::kFloat32) {
    plan.linear_elementwise_route = &select_f32_linear;
    plan.scalar_elementwise_route = &select_scalar<sizeof(float)>;
    return true;
  }
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

std::size_t dense_broadcast_suffix(std::span<const TensorDesc> inputs,
                                   const TensorDesc &output) noexcept {
  if (!output.is_dense())
    return 0;
  for (const TensorDesc &input : inputs) {
    if (!input.is_dense())
      return 0;
  }

  const auto output_dimensions = output.dimensions();
  std::size_t chunk = 1;
  for (std::size_t reverse = 0; reverse < output_dimensions.size(); ++reverse) {
    const std::size_t output_axis = output_dimensions.size() - 1 - reverse;
    const std::int64_t output_extent = output_dimensions[output_axis];
    bool all_match = true;
    for (const TensorDesc &input : inputs) {
      if (reverse >= input.rank() ||
          input.dimensions()[input.rank() - 1 - reverse] != output_extent) {
        all_match = false;
        break;
      }
    }
    if (!all_match)
      break;
    const auto next =
        math::try_multiply(chunk, static_cast<std::size_t>(output_extent));
    if (!next)
      return 0;
    chunk = *next;
  }
  return chunk > 1 ? chunk : 0;
}

Status validate_plain_arithmetic(const TensorDesc &description) noexcept {
  const DataType type = description.data_type();
  if ((type != DataType::kFloat32 && type != DataType::kFloat16 &&
       type != DataType::kBFloat16) ||
      !description.is_plain()) {
    return Status::Unimplemented("native elementwise arithmetic requires plain "
                                 "float32/float16/bfloat16");
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
  std::size_t broadcast_chunk = 0;
  switch (operation.kind()) {
  case OperationKind::kUnary: {
    if (inputs.size() != 1)
      return Status::Internal("native Unary arity is invalid");
    Status status = validate_plain_arithmetic(inputs[0]);
    if (!status.ok())
      return status;
    status = validate_plain_arithmetic(outputs[0]);
    if (!status.ok())
      return status;
    const Unary *unary = operation.get_if<Unary>();
    if (unary == nullptr ||
        !select_unary(unary->kind, inputs[0].data_type(), plan))
      return Status::Unimplemented("native Unary kind/dtype is not retained");
    linear = dense_same_shape(inputs[0], outputs[0]);
    break;
  }
  case OperationKind::kBinary: {
    if (inputs.size() != 2)
      return Status::Internal("native Binary arity is invalid");
    for (const TensorDesc &input : inputs) {
      Status status = validate_plain_arithmetic(input);
      if (!status.ok())
        return status;
    }
    Status status = validate_plain_arithmetic(outputs[0]);
    if (!status.ok())
      return status;
    const Binary *binary = operation.get_if<Binary>();
    if (binary == nullptr ||
        !select_binary(binary->kind, inputs[0].data_type(), plan))
      return Status::InvalidArgument("native Binary kind is invalid");
    linear = dense_same_shape(inputs[0], outputs[0]) &&
             dense_same_shape(inputs[1], outputs[0]);
    if (!linear)
      broadcast_chunk = dense_broadcast_suffix(inputs, outputs[0]);
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
    if (!select_select(outputs[0].data_type(), element_size, plan)) {
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
  plan.parameters[1] = linear ? *elements : broadcast_chunk;
  plan.parameters[2] = data_type_size(outputs[0].data_type());
  if (linear) {
    plan.alias_input_mask = static_cast<std::uint8_t>(
        (std::uint8_t{1} << static_cast<std::uint8_t>(inputs.size())) - 1);
  } else if (broadcast_chunk != 0) {
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      if (dense_same_shape(inputs[index], outputs[0])) {
        plan.alias_input_mask = static_cast<std::uint8_t>(
            plan.alias_input_mask | (std::uint8_t{1} << index));
      }
    }
  }
  plan.submit_route = &submit_elementwise<>;
  return Status::Ok();
}

Status prepare_half_gated(Operation operation,
                          std::span<const TensorDesc> inputs,
                          std::span<const TensorDesc> outputs,
                          NativePlan &plan) noexcept {
  const GatedActivation *gated = operation.get_if<GatedActivation>();
  if (gated == nullptr || outputs.size() != 1 || inputs.empty() ||
      inputs.size() != (gated->inputs == GatedInputKind::kSeparate ? 2U : 1U) ||
      outputs[0].rank() == 0)
    return Status::Internal("native half gated activation arity is invalid");
  const DataType type = inputs[0].data_type();
  if (type != DataType::kFloat16 ||
      gated->activation != GatedActivationKind::kSilu)
    return Status::Unimplemented(
        "native gated activation retains only qualified float16 SiLU");
  for (const TensorDesc &input : inputs)
    if (!input.is_plain() || !input.is_dense())
      return Status::Unimplemented(
          "native gated activation requires dense plain input");
  if (!outputs[0].is_plain() || !outputs[0].is_dense())
    return Status::Unimplemented(
        "native gated activation requires dense plain output");
  const auto dimensions = outputs[0].dimensions();
  std::size_t rows = 1;
  for (std::size_t axis = 0; axis + 1 < dimensions.size(); ++axis) {
    const auto extent = math::try_integer_cast<std::size_t>(dimensions[axis]);
    if (!extent)
      return Status::OutOfRange("gated row extent exceeds size_t");
    const auto next = math::try_multiply(rows, *extent);
    if (!next)
      return Status::OutOfRange("gated row count exceeds size_t");
    rows = *next;
  }
  const auto width =
      math::try_integer_cast<std::size_t>(inputs[0].dimensions().back());
  const auto hidden = math::try_integer_cast<std::size_t>(dimensions.back());
  if (!width || !hidden)
    return Status::OutOfRange("gated hidden width exceeds size_t");
  plan.parameters = {rows, *hidden, data_type_size(type)};
  plan.row_first_stride = *width;
  plan.row_output_stride = *hidden;
  plan.linear_elementwise_route = &gated_narrow_linear<simd::float16_t>;
  plan.submit_route = &submit_elementwise<true>;
  return Status::Ok();
}

} // namespace hyle::compute::native::internal
