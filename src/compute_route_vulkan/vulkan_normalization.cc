#include "vulkan_program_internal.h"
#include "vulkan_tensor.h"

#include <array>
#include <new>

namespace hyle::compute::vulkan::detail {

bool layer_norm_limits(const ProgramState &program,
                       const TensorDesc &input) noexcept {
  const auto elements = input.element_count();
  if (!elements || *elements == 0 || input.rank() == 0)
    return false;
  const auto width = static_cast<std::uint64_t>(input.dimensions().back());
  const auto &limits = program.runtime->device->properties.limits;
  return width != 0 && *elements <= INT32_MAX &&
         *elements <= limits.maxStorageBufferRange / sizeof(float) &&
         limits.maxComputeWorkGroupInvocations >= 512 &&
         limits.maxComputeWorkGroupSize[0] >= 512 &&
         limits.maxComputeSharedMemorySize >= 512 * sizeof(float) &&
         *elements / width <= limits.maxComputeWorkGroupCount[0];
}

Status lower_layer_norm(ProgramState &program, const PlanStep &step) noexcept {
  try {
    const auto input = program.values[step.inputs[0]].description;
    const auto output = program.values[step.outputs[0]].description;
    const auto operation = *step.operation.get_if<LayerNorm>();
    if (operation.accumulation_type != DataType::kFloat32)
      return Status::Unimplemented(
          "Vulkan normalization requires F32 accumulation");
    if (*output.element_count() == 0)
      return Status::Ok();
    if (!layer_norm_limits(program, input))
      return lower_tensor(program, step);

    while (program.aliases.size() < program.values.size()) {
      const auto id = static_cast<std::uint32_t>(program.aliases.size());
      program.aliases.push_back({id, 0});
    }
    const auto append = [&](const TensorDesc &description) {
      const auto id = static_cast<std::uint32_t>(program.values.size());
      program.values.push_back({description, {}});
      program.aliases.push_back({id, 0});
      return id;
    };
    const auto alias = [&](const TensorDesc &description,
                           std::uint32_t source) {
      const auto owner = program.aliases[source];
      const auto id = append(description);
      program.aliases[id] = owner;
      return id;
    };
    const auto wide = [&](std::uint32_t source) -> Result<std::uint32_t> {
      const auto description = program.values[source].description;
      if (description.is_dense() &&
          description.data_type() == DataType::kFloat32)
        return source;
      HYLE_ASSIGN_OR_RETURN(
          auto dense,
          TensorDesc::dense(DataType::kFloat32, description.dimensions()));
      const auto converted = append(dense);
      HYLE_RETURN_IF_ERROR(
          lower_tensor(program, {Operation(Cast{DataType::kFloat32}),
                                 {source},
                                 {converted},
                                 {},
                                 {1, 1, 1},
                                 {}}));
      return converted;
    };

    const auto width = input.dimensions().back();
    const auto rows = static_cast<std::int64_t>(*input.element_count()) / width;
    HYLE_ASSIGN_OR_RETURN(
        auto matrix,
        TensorDesc::dense(DataType::kFloat32,
                          std::array<std::int64_t, 2>{rows, width}));
    HYLE_ASSIGN_OR_RETURN(auto source, wide(step.inputs[0]));
    std::vector<std::uint32_t> inputs{alias(matrix, source)};
    for (std::size_t port = 1; port < step.inputs.size(); ++port) {
      HYLE_ASSIGN_OR_RETURN(auto parameter, wide(step.inputs[port]));
      inputs.push_back(parameter);
    }

    auto destination = step.outputs[0];
    if (output.data_type() != DataType::kFloat32) {
      HYLE_ASSIGN_OR_RETURN(auto dense, TensorDesc::dense(DataType::kFloat32,
                                                          output.dimensions()));
      destination = append(dense);
    }
    const auto result = alias(matrix, destination);
    Plan region;
    region.steps.push_back(
        {step.operation, std::move(inputs), {result}, {}, {1, 1, 1}, {}});
    region.outputs = {result};
    const std::vector<std::size_t> last_uses(program.values.size(), 0);
    HYLE_RETURN_IF_ERROR(lower_region(program, region, 0, 1, last_uses));
    if (destination != step.outputs[0])
      HYLE_RETURN_IF_ERROR(
          lower_tensor(program, {Operation(Cast{output.data_type()}),
                                 {destination},
                                 step.outputs,
                                 {},
                                 {1, 1, 1},
                                 {}}));
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan LayerNorm adapter metadata");
  }
}

} // namespace hyle::compute::vulkan::detail
