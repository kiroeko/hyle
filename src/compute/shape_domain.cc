#include "shape_domain.h"

#include <array>
#include <utility>

namespace hyle::compute::detail {
namespace {

Status validate_tensors(ProviderShapeDomain domain,
                        std::span<const TensorDesc> tensors,
                        std::span<const ProviderTensorAxes> maps) noexcept {
  if (tensors.size() != maps.size()) {
    return Status::InvalidArgument(
        "provider shape axis maps do not match tensor arity");
  }
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const TensorDesc &description = tensors[index];
    if (!description.valid())
      return Status::InvalidArgument("provider shape tensor is invalid");
    for (std::size_t axis = description.rank(); axis < kMaxTensorRank; ++axis) {
      if (maps[index].values[axis] != kFixedShapeAxis) {
        return Status::InvalidArgument(
            "provider shape axis map exceeds tensor rank");
      }
    }
    bool dynamic = false;
    for (std::size_t axis = 0; axis < description.rank(); ++axis) {
      const std::uint32_t symbol = maps[index].values[axis];
      if (symbol == kFixedShapeAxis)
        continue;
      if (symbol >= domain.dimensions.size()) {
        return Status::InvalidArgument(
            "provider shape axis refers to an unknown value");
      }
      const ProviderShapeRange range = domain.dimensions[symbol].bounds;
      if (description.dimensions()[axis] != range.last()) {
        return Status::InvalidArgument(
            "provider shape tensor does not describe its envelope");
      }
      dynamic = dynamic || !range.exact();
      if (!range.exact() && description.block_axis() == axis &&
          range.multiple_of %
                  description.storage_format().elements_per_block() !=
              0) {
        return Status::InvalidArgument(
            "provider shape domain contains incomplete storage blocks");
      }
    }
    if (dynamic &&
        (!description.is_dense() ||
         description.storage_format().kind() == StorageFormatKind::kOpaque)) {
      return Status::Unimplemented(
          "provider shape domain requires dense dynamic tensor layouts");
    }
  }
  return Status::Ok();
}

Status bind_tensors(std::span<const TensorDesc> upper,
                    std::span<const ProviderTensorAxes> maps,
                    std::span<const std::int64_t> point,
                    std::span<TensorDesc> destination) noexcept {
  for (std::size_t index = 0; index < upper.size(); ++index) {
    const TensorDesc &description = upper[index];
    std::array<std::int64_t, kMaxTensorRank> extents{};
    bool changed = false;
    for (std::size_t axis = 0; axis < description.rank(); ++axis) {
      const std::uint32_t symbol = maps[index].values[axis];
      extents[axis] = symbol == kFixedShapeAxis ? description.dimensions()[axis]
                                                : point[symbol];
      changed = changed || extents[axis] != description.dimensions()[axis];
    }
    if (!changed) {
      destination[index] = description;
      continue;
    }
    const auto dimensions =
        std::span<const std::int64_t>(extents).first(description.rank());
    auto concrete =
        description.is_plain()
            ? TensorDesc::dense(description.data_type(), dimensions)
            : TensorDesc::blocked_dense(description.data_type(), dimensions,
                                        description.storage_format());
    if (!concrete)
      return std::move(concrete).error();
    destination[index] = *concrete;
  }
  return Status::Ok();
}

} // namespace

Status
validate_shape_domain(ProviderShapeDomain domain,
                      std::span<const TensorDesc> upper_inputs,
                      std::span<const TensorDesc> upper_outputs) noexcept {
  if (domain.empty())
    return Status::Ok();
  Status program_status = validate_shape_program(domain.program());
  if (!program_status.ok())
    return program_status;
  Status status = validate_tensors(domain, upper_inputs, domain.inputs);
  return status.ok() ? validate_tensors(domain, upper_outputs, domain.outputs)
                     : status;
}

Status bind_shape_domain(ProviderShapeDomain domain,
                         std::span<const TensorDesc> upper_inputs,
                         std::span<const TensorDesc> upper_outputs,
                         std::span<std::int64_t> values,
                         std::span<TensorDesc> inputs,
                         std::span<TensorDesc> outputs) noexcept {
  if (inputs.size() != upper_inputs.size() ||
      outputs.size() != upper_outputs.size() ||
      values.size() != domain.dimensions.size()) {
    return Status::InvalidArgument(
        "provider shape binding counts do not match the domain");
  }
  Status status = validate_shape_domain(domain, upper_inputs, upper_outputs);
  if (!status.ok())
    return status;
  status = evaluate_shape_program(domain.program(), values);
  if (!status.ok())
    return status;
  if (domain.empty()) {
    for (std::size_t index = 0; index < inputs.size(); ++index)
      inputs[index] = upper_inputs[index];
    for (std::size_t index = 0; index < outputs.size(); ++index)
      outputs[index] = upper_outputs[index];
    return Status::Ok();
  }
  status = bind_tensors(upper_inputs, domain.inputs, values, inputs);
  return status.ok()
             ? bind_tensors(upper_outputs, domain.outputs, values, outputs)
             : status;
}

} // namespace hyle::compute::detail
