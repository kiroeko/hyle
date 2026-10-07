#include "native_plan.h"
#include "numeric.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/math/scalar.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/math.h"
#include "hyle/simd/ops.h"

namespace hyle::compute::native::internal {
namespace {

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

struct NormEpisode final {
  const NativePlan *plan = nullptr;
  const void *input = nullptr;
  const void *scale = nullptr;
  void *output = nullptr;
};

template <class Value, class Scale>
void run_narrow_norm_range(void *context, std::size_t, std::size_t begin,
                           std::size_t end) noexcept {
  auto &episode = *static_cast<NormEpisode *>(context);
  const std::size_t hidden = episode.plan->parameters[1];
  std::array<float, 128> values;
  std::array<float, 128> parameter;
  for (std::size_t row = begin; row < end; ++row) {
    const std::size_t offset = row * hidden;

    float squares = 0.0F;
    for (std::size_t column = 0; column < hidden; column += values.size()) {
      const std::size_t size = (std::min)(values.size(), hidden - column);
      numeric::load_tile<Value>(episode.input, offset + column, values.data(),
                                size);
      squares += simd::dot(values.data(), values.data(), size);
    }
    const float inverse =
        1.0F /
        std::sqrt(squares / static_cast<float>(hidden) + episode.plan->scalar);

    // Reductions finish before the first write, including exact input alias.
    // Parameters stay in their declared storage and output is rounded once.
    for (std::size_t column = 0; column < hidden; column += values.size()) {
      const std::size_t size = (std::min)(values.size(), hidden - column);
      numeric::load_tile<Value>(episode.input, offset + column, values.data(),
                                size);
      for (std::size_t index = 0; index < size; ++index)
        values[index] = (values[index] - 0.0F) * inverse;
      if (episode.scale != nullptr) {
        numeric::load_tile<Scale>(episode.scale, column, parameter.data(),
                                  size);
        simd::multiply(values.data(), parameter.data(), values.data(), size);
      }

      numeric::store_tile<Value>(values.data(), episode.output, offset + column,
                                 size);
    }
  }
}

template <class Value, class Scale>
detail::EnqueueResult
submit_normalization(const NativePlan &plan,
                     std::span<const ConstTensorView> inputs,
                     std::span<const TensorView> outputs) noexcept {
  if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1)
    return completed(Status::Internal("native normalization plan is invalid"));
  const std::size_t rows = plan.parameters[0];
  const std::size_t hidden = plan.parameters[1];
  if (rows == 0 || hidden == 0)
    return completed(Status::Ok());

  const mem::BufferView output_storage = outputs[0].storage();
  const bool in_place = same_storage_range(inputs[0].storage(), output_storage);

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
  std::array<const void *, 3> source_data{};
  mem::HostMapping output_mapping;
  void *output_data = nullptr;

  if (in_place) {
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
    source_data[0] = output_data;
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

  if (!in_place) {
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
        "CPU normalization mapping does not satisfy tensor alignment"));
  }

  NormEpisode episode{
      .plan = &plan,
      .input = source_data[0],
      .scale = inputs.size() > 1 ? source_data[1] : nullptr,
      .output = output_data,
  };
  parallel_for_ranges(plan, rows, &run_narrow_norm_range<Value, Scale>, &episode);

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

Result<std::size_t> outer_count(const TensorDesc &description) noexcept {
  std::size_t result = 1;
  const auto dimensions = description.dimensions();
  for (std::size_t axis = 0; axis + 1 < dimensions.size(); ++axis) {
    auto value = math::try_integer_cast<std::size_t>(dimensions[axis]);
    if (!value)
      return std::unexpected(
          Status::OutOfRange("normalization dimension exceeds size_t"));
    auto next = math::try_multiply(result, *value);
    if (!next)
      return std::unexpected(
          Status::OutOfRange("normalization row count overflows size_t"));
    result = *next;
  }
  return result;
}

Status validate_dense_arithmetic(const TensorDesc &description) noexcept {
  const DataType type = description.data_type();
  if ((type != DataType::kFloat32 && type != DataType::kFloat16 &&
       type != DataType::kBFloat16) ||
      !description.is_plain() || !description.is_dense()) {
    return Status::Unimplemented(
        "native normalization requires dense plain float32/float16/bfloat16");
  }
  return Status::Ok();
}

template <class Value>
SubmitRoute select_narrow_norm(std::span<const TensorDesc> inputs) noexcept {
  const bool scale_f32 =
      inputs.size() > 1 && inputs[1].data_type() == DataType::kFloat32;

  return scale_f32 ? &submit_normalization<Value, float>
                   : &submit_normalization<Value, Value>;

}

} // namespace

Status prepare_normalization(Operation operation,
                             std::span<const TensorDesc> inputs,
                             std::span<const TensorDesc> outputs,
                             NativePlan &plan) noexcept {
  if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1)
    return Status::Internal("native normalization arity is invalid");
  const RmsNorm *rms = operation.get_if<RmsNorm>();
  if (rms == nullptr)
    return Status::Internal("native normalization descriptor is missing");
  const DataType accumulation = rms->accumulation_type;
  if (accumulation != DataType::kFloat32) {
    return Status::Unimplemented(
        "native normalization supports float32 accumulation");
  }
  for (const TensorDesc &input : inputs) {
    Status status = validate_dense_arithmetic(input);
    if (!status.ok())
      return status;
  }
  Status output_status = validate_dense_arithmetic(outputs[0]);
  if (!output_status.ok())
    return output_status;

  auto rows = outer_count(inputs[0]);
  if (!rows)
    return std::move(rows).error();
  auto hidden = math::try_integer_cast<std::size_t>(
      inputs[0].dimensions()[inputs[0].rank() - 1]);
  if (!hidden)
    return Status::OutOfRange("normalization hidden dimension exceeds size_t");
  plan.parameters = {*rows, *hidden, 0};
  plan.scalar = rms->epsilon;
  plan.alias_input_mask = 1;
  switch (inputs[0].data_type()) {
  case DataType::kFloat32:
    return Status::Internal("float32 RMSNorm uses its native float route");
  case DataType::kBFloat16:
    plan.submit_route = select_narrow_norm<simd::bfloat16_t>(inputs);
    break;
  default:
    return Status::Unimplemented("unsupported native normalization data type");
  }
  return Status::Ok();
}

} // namespace hyle::compute::native::internal
