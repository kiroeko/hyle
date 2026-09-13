#include "native_plan.h"

#include <array>
#include <cstddef>
#include <span>
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

detail::EnqueueResult
submit_layer_norm(const NativePlan &plan,
                  std::span<const ConstTensorView> inputs,
                  std::span<const TensorView> outputs) noexcept {
  if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1)
    return completed(Status::Internal("native LayerNorm plan is invalid"));
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
  std::array<const float *, 3> source_data{};
  mem::HostMapping output_mapping;
  float *output_data = nullptr;

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
    output_data = reinterpret_cast<float *>(writable->data());
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
    source_data[index] =
        reinterpret_cast<const float *>(input_mappings[index].bytes().data());
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
    output_data = reinterpret_cast<float *>(writable->data());
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
        "CPU LayerNorm mapping does not satisfy tensor alignment"));
  }

  const float *scale = inputs.size() > 1 ? source_data[1] : nullptr;
  const float *bias = inputs.size() > 2 ? source_data[2] : nullptr;
  for (std::size_t row = 0; row < rows; ++row) {
    const float *row_input = source_data[0] + row * hidden;
    float *row_output = output_data + row * hidden;
    if (bias != nullptr) {
      simd::layer_norm_f32(row_input, scale, bias, row_output, hidden,
                           plan.scalar);
    } else {
      simd::layer_norm_f32(row_input, row_output, hidden, plan.scalar);
      if (scale != nullptr)
        simd::multiply(row_output, scale, row_output, hidden);
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

Result<std::size_t> outer_count(const TensorDesc &description) noexcept {
  std::size_t result = 1;
  const auto dimensions = description.dimensions();
  for (std::size_t axis = 0; axis + 1 < dimensions.size(); ++axis) {
    auto value = math::try_integer_cast<std::size_t>(dimensions[axis]);
    if (!value)
      return std::unexpected(
          Status::OutOfRange("LayerNorm dimension exceeds size_t"));
    auto next = math::try_multiply(result, *value);
    if (!next)
      return std::unexpected(
          Status::OutOfRange("LayerNorm row count overflows size_t"));
    result = *next;
  }
  return result;
}

Status validate_dense_f32(const TensorDesc &description) noexcept {
  if (description.data_type() != DataType::kFloat32 ||
      !description.is_plain() || !description.is_dense()) {
    return Status::Unimplemented(
        "native LayerNorm currently requires dense plain float32 tensors");
  }
  return Status::Ok();
}

} // namespace

Status prepare_layer_norm(Operation operation,
                          std::span<const TensorDesc> inputs,
                          std::span<const TensorDesc> outputs,
                          NativePlan &plan) noexcept {
  if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1)
    return Status::Internal("native LayerNorm arity is invalid");
  const LayerNorm *normalization = operation.get_if<LayerNorm>();
  if (normalization == nullptr)
    return Status::Internal("native LayerNorm descriptor is missing");
  if (normalization->accumulation_type != DataType::kFloat32) {
    return Status::Unimplemented(
        "native LayerNorm currently supports float32 accumulation");
  }
  for (const TensorDesc &input : inputs) {
    Status status = validate_dense_f32(input);
    if (!status.ok())
      return status;
  }
  Status output_status = validate_dense_f32(outputs[0]);
  if (!output_status.ok())
    return output_status;

  auto rows = outer_count(inputs[0]);
  if (!rows)
    return std::move(rows).error();
  auto hidden = math::try_integer_cast<std::size_t>(
      inputs[0].dimensions()[inputs[0].rank() - 1]);
  if (!hidden)
    return Status::OutOfRange("LayerNorm hidden dimension exceeds size_t");
  plan.parameters = {*rows, *hidden, 0};
  plan.scalar = normalization->epsilon;
  plan.alias_input_mask = 1;
  plan.submit_route = &submit_layer_norm;
  return Status::Ok();
}

} // namespace hyle::compute::native::internal
