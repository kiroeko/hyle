#include "vulkan_mlx.h"
#include "vulkan_tensor.h"

#include <algorithm>
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
  const bool mx8 = spec.kind == iv::ProgramKind::MlxMx8Linear ||
                   spec.kind == iv::ProgramKind::MlxMx8Grouped;
  const auto pack = mx8 ? 4U : 8U;
  const auto product = [](std::uint64_t a, std::uint64_t b) {
    return a > 0 && b > 0 && a <= max_elements / b;
  };
  if (!product(r, n) || !product(r, k) ||
      (embedding ? !product(e, k / 8)
                 : (!product(e, n) || !product(e * n, k / pack) ||
                    !product(r * n, spec.group_size))))
    return false;
  // Original compiler owns one selected K group and the loop accumulators.
  // Include alignment and both source-typed and F32 intermediates, without
  // ever charging/allocating a full float expert bank.
  const auto work = embedding ? r * n : r * n * spec.group_size;
  const auto bytes = embedding ? 16 * r * n + 512
                               : 4 * work + 4 * r * spec.group_size + 40 * r * n + 512;
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
    const bool affine = op.format == model_ops::MlxWeightFormat::kAffine4;
    const auto output = program.values[step.outputs[0]].description;
    if (*output.element_count() == 0)
      return Status::Ok(); // Only an empty, schema-valid embedding admits this.
    const auto &weight = program.values[step.inputs[embedding ? 0 : 1]].description;
    iv::ProgramSpec spec;
    spec.group_size = op.group_size;
    spec.output_type = scalar(output.data_type());
    spec.scale_type = spec.output_type;
    spec.experts = (embedding || grouped) ? weight.dimensions()[0] : 1;
    if (embedding) {
      spec.kind = iv::ProgramKind::MlxAffineEmbedding;
      spec.rows = output.dimensions()[0];
      spec.columns = output.dimensions()[1];
      spec.depth = spec.columns;
    } else {
      spec.kind = affine ? (grouped ? iv::ProgramKind::MlxAffineGrouped : iv::ProgramKind::MlxAffineLinear)
                  : op.format == model_ops::MlxWeightFormat::kMxfp4
                      ? (grouped ? iv::ProgramKind::MlxMx4Grouped : iv::ProgramKind::MlxMx4Linear)
                      : (grouped ? iv::ProgramKind::MlxMx8Grouped : iv::ProgramKind::MlxMx8Linear);
      spec.depth = program.values[step.inputs[0]].description.dimensions().back();
      spec.columns = weight.dimensions()[weight.rank() - 2];
      spec.rows = static_cast<std::int64_t>(*output.element_count()) / spec.columns;
    }
    const auto &device = *program.runtime->device;
    const auto &features = device.provider.native();
    if (!features.shader_int16 || !features.shader_float16 ||
        !features.storage_buffer16 || !features.preserve_float32_specials ||
        !fits(device, spec, embedding))
      return Status::Unimplemented("Vulkan MLX original numeric features, group workspace or index domain");
    HYLE_RETURN_IF_ERROR(ensure_iree_device(program));
    auto inputs = step.inputs;
    if (!embedding && affine) {
      // Schema-promoted mixed affine ports use existing qualified casts. Do
      // this cold once; the corresponding GPU work remains charged on replay.
      for (std::size_t port : {std::size_t{0}, std::size_t{2}, std::size_t{3}}) {
        const auto description = program.values[inputs[port]].description;
        if (description.data_type() == output.data_type()) continue;
        HYLE_ASSIGN_OR_RETURN(auto promoted,
            TensorDesc::dense(output.data_type(), description.dimensions()));
        const auto value = append(program, promoted);
        HYLE_RETURN_IF_ERROR(lower_native_cast(program, inputs[port], value));
        inputs[port] = value;
      }
    }
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
