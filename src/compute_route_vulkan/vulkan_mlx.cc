#include "vulkan_mlx.h"
#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <limits>
#include <new>

#include "hyle/compute/model_ops.h"

namespace hyle::compute::vulkan::detail {
namespace iv = iree_vulkan;
namespace {
iv::ArtifactScalar scalar(DataType type) noexcept {
  return type == DataType::kFloat16 ? iv::ArtifactScalar::Float16
         : type == DataType::kBFloat16 ? iv::ArtifactScalar::BFloat16
                                     : iv::ArtifactScalar::Float32;
}

bool fits(const Device &device, const iv::ProgramSpec &spec,
          bool embedding) noexcept {
  const auto &limits = device.properties.limits;
  if (limits.maxComputeWorkGroupInvocations < 64 ||
      limits.maxComputeWorkGroupSize[0] < 64 ||
      limits.maxComputeWorkGroupCount[1] < 1 ||
      limits.maxComputeWorkGroupCount[2] < 1)
    return false;
  constexpr auto max_elements = std::uint64_t{INT32_MAX} / 4;
  const auto r = static_cast<std::uint64_t>(spec.rows);
  const auto n = static_cast<std::uint64_t>(spec.columns);
  const auto k = static_cast<std::uint64_t>(spec.depth);
  const auto e = static_cast<std::uint64_t>(spec.experts);
  const auto input = static_cast<std::uint64_t>(iv::mlx_input_depth(spec));
  const auto output = static_cast<std::uint64_t>(iv::mlx_output_columns(spec));
  const auto tile = iv::mlx_as_stored(spec.kind) ? (std::min)(n, std::uint64_t{64})
                                               : std::uint64_t{spec.group_size};
  const auto words = (k / 32) * spec.bits;
  const auto product = [](std::uint64_t a, std::uint64_t b) {
    return a > 0 && b > 0 && a <= max_elements / b;
  };
  if (!product(r, output) ||
      (iv::mlx_grouped(spec.kind) ? iv::mlx_activation_elements(spec) == 0 : !product(r, input)) ||
      (embedding ? !product(e, words)
                 : (!product(e, n) || !product(e * n, words) ||
                    (iv::mlx_as_stored(spec.kind)
                         ? (!product(r, spec.group_size) || !product(r * spec.group_size, tile))
                         : !product(r * n, spec.group_size)))))
    return false;
  // Original compiler owns one selected K group and the loop accumulators.
  // Include alignment and both source-typed and F32 intermediates, without
  // ever charging/allocating a full float expert bank.
  const auto work = embedding ? r * n : iv::mlx_as_stored(spec.kind)
      ? r * spec.group_size * tile : r * n * spec.group_size;
  const auto bytes = embedding ? 16 * r * n + 512
                               : 4 * work + 4 * r * tile + 40 * r * output + 512;
  return bytes <= limits.maxStorageBufferRange &&
      (std::min)((work + 63) / 64, std::uint64_t{65535}) <=
          limits.maxComputeWorkGroupCount[0];
}

std::uint32_t append(ProgramState &program, const TensorDesc &description) {
  const auto id = static_cast<std::uint32_t>(program.values.size());
  program.values.push_back({description, {}});
  program.aliases.push_back({id, 0});
  return id;
}
} // namespace

Status lower_mlx(ProgramState &program, const PlanStep &step) noexcept {
  try {
    const auto op = *step.operation.custom_as<model_ops::MlxQuantized>();
    const bool embedding = op.kind == model_ops::MlxQuantizedKind::kEmbedding;
    const bool grouped = op.kind == model_ops::MlxQuantizedKind::kGrouped;
    const bool affine = op.format == model_ops::MlxWeightFormat::kAffine;
    const bool as_stored = op.orientation == model_ops::MlxMatmulOrientation::kWeightAsStored;
    const auto output = program.values[step.outputs[0]].description;
    if (*output.element_count() == 0)
      return Status::Ok(); // Only an empty, schema-valid embedding admits this.
    const auto &weight = program.values[step.inputs[embedding ? 0 : 1]].description;
    iv::ProgramSpec spec;
    spec.bits = op.bits;
    spec.group_size = op.group_size;
    spec.output_type = scalar(output.data_type());
    spec.activation_type = embedding ? spec.output_type
        : scalar(program.values[step.inputs[0]].description.data_type());
    spec.scale_type = affine
        ? scalar(program.values[step.inputs[embedding ? 1 : 2]].description.data_type())
        : spec.output_type;
    spec.experts = (embedding || grouped) ? weight.dimensions()[0] : 1;
    if (embedding) {
      spec.kind = iv::ProgramKind::MlxAffineEmbedding;
      spec.rows = output.dimensions()[0];
      spec.columns = output.dimensions()[1];
      spec.depth = spec.columns;
    } else {
      spec.kind = as_stored
                  ? (grouped ? iv::ProgramKind::MlxAffineGroupedAsStored : iv::ProgramKind::MlxAffineLinearAsStored)
                  : affine ? (grouped ? iv::ProgramKind::MlxAffineGrouped : iv::ProgramKind::MlxAffineLinear)
                  : op.format == model_ops::MlxWeightFormat::kMxfp4
                      ? (grouped ? iv::ProgramKind::MlxMx4Grouped : iv::ProgramKind::MlxMx4Linear)
                      : (grouped ? iv::ProgramKind::MlxMx8Grouped : iv::ProgramKind::MlxMx8Linear);
      spec.depth = program.values[step.inputs[2]].description.dimensions().back() * op.group_size;
      spec.columns = weight.dimensions()[weight.rank() - 2];
      spec.rows = static_cast<std::int64_t>(*output.element_count()) / iv::mlx_output_columns(spec);
    }
    const auto activation_geometry = [&](std::uint32_t value) {
      if (!grouped) return;
      const auto &a = program.values[value].description;
      const auto width = static_cast<std::int64_t>(data_type_size(a.data_type()));
      spec.selection_slots = output.dimensions()[1];
      spec.activation_slots = a.dimensions()[1] == 1 || a.byte_strides()[1] == 0
                                  ? 1 : a.dimensions()[1];
      spec.activation_token_stride = a.dimensions()[0] > 1
          ? a.byte_strides()[0] / width : spec.activation_slots * iv::mlx_input_depth(spec);
    };
    activation_geometry(step.inputs[0]);
    const auto &device = *program.runtime->device;
    const auto &features = device.provider.native();
    if (!features.shader_int16 || !features.shader_float16 ||
        !features.storage_buffer16 || !features.preserve_float32_specials ||
        !fits(device, spec, embedding))
      return Status::Unimplemented("Vulkan MLX original numeric features, group workspace or index domain");
    HYLE_RETURN_IF_ERROR(ensure_iree_device(program));
    auto inputs = step.inputs;
    if (grouped) {
      const auto &a = program.values[inputs[0]].description;
      if (a.dimensions()[1] != 1 && a.byte_strides()[1] == 0) {
        std::array<std::int64_t, 3> dimensions{a.dimensions()[0], 1, a.dimensions()[2]};
        HYLE_ASSIGN_OR_RETURN(auto collapsed,
            TensorDesc::strided_bytes(a.data_type(), dimensions, a.byte_strides()));
        const auto source = inputs[0];
        inputs[0] = append(program, collapsed);
        program.aliases[inputs[0]] = program.aliases[source];
      }
    }
    // Typed programs widen actual selected scalars, preserving the original
    // activation pitch and scale/bias banks instead of casting whole inputs.
    activation_geometry(inputs[0]);
    auto destination = step.outputs[0];
    if (embedding || grouped) {
      if (!program.status_value) {
        HYLE_ASSIGN_OR_RETURN(auto status, TensorDesc::dense(DataType::kInt32, {}));
        program.status_value = append(program, status);
      }
      const auto ids = program.values[inputs.back()].description;
      const auto safe = append(program, ids);
      HYLE_RETURN_IF_ERROR(lower_index_check(program, inputs.back(), safe,
          static_cast<std::uint64_t>(spec.experts), true));
      inputs.back() = safe;
      destination = append(program, output);
    }
    Dispatch dispatch;
    dispatch.iree = spec;
    dispatch.values = std::move(inputs);
    dispatch.values.push_back(destination);
    program.dispatches.push_back(std::move(dispatch));
    return (embedding || grouped)
               ? lower_checked_copy(program, destination, step.outputs[0])
               : Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan original MLX preparation metadata");
  }
}
} // namespace hyle::compute::vulkan::detail
