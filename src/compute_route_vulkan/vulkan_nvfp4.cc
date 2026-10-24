#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <limits>
#include <new>

#include "hyle/compute/model_ops.h"
#include "vulkan_shaders.h"

namespace hyle::compute::vulkan::detail {
Status lower_nvfp4(ProgramState &program, const PlanStep &step) noexcept try {
  const auto mx = step.operation.custom_as<model_ops::Mxfp4Linear>();
  const auto grouped_mx = step.operation.custom_as<model_ops::GroupedMxfp4Linear>();
  const bool mxfp4 = mx || grouped_mx;
  const bool weight_only = mxfp4 &&
      (grouped_mx ? grouped_mx->activation_mode : mx->activation_mode) ==
          model_ops::Mxfp4ActivationMode::kBFloat16;
  const auto local = step.operation.custom_as<model_ops::LocalNvfp4Linear>();
  const auto grouped_local = step.operation.custom_as<model_ops::GroupedLocalNvfp4Linear>();
  const bool local_math = local || grouped_local;
  const bool grouped = step.operation.custom_as<model_ops::GroupedMxfp4Linear>() || grouped_local ||
      step.operation.custom_as<model_ops::GroupedNvfp4Linear>().has_value();
  const bool round_input = local_math &&
      (grouped_local ? grouped_local->rounding : local->rounding) ==
          model_ops::Fp4InputRounding::kBFloat16;
  std::array<TensorDesc, 6> descriptions;
  const auto id_port = mxfp4 ? 3U : 5U;
  const std::span inputs(descriptions.data(), mxfp4 ? (grouped ? 4U : 3U) : (grouped ? 6U : 5U));
  if (step.inputs.size() != inputs.size() || step.outputs.size() != 1)
    return Status::InvalidArgument("Vulkan NVFP4 port count differs");
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = program.values[step.inputs[i]].description;
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(step.operation, inputs, inferred));
  if (inferred[0] != program.values[step.outputs[0]].description)
    return Status::InvalidArgument("Vulkan NVFP4 output differs from schema");
  const auto &device = *program.runtime->device;
  const auto &limits = device.properties.limits;
  const auto columns = std::uint64_t(inputs[1].dimensions()[grouped ? 1 : 0]);
  const auto depth = std::uint64_t(inputs[0].dimensions().back());
  const auto rows = std::uint64_t(*inferred[0].element_count()) / columns;
  const auto slots = grouped ? std::uint64_t(inputs[id_port].dimensions()[1]) : 1;
  const auto experts = grouped ? std::uint64_t(inputs[1].dimensions()[0]) : 1;
  // Unused strides have no addressing effect, including a padded T=1 view.
  const auto token_stride = grouped && inputs[0].dimensions()[0] > 1
      ? std::uint64_t(inputs[0].byte_strides()[0]) : 0;
  const auto slot_stride = grouped && inputs[0].dimensions()[1] != 1
      ? std::uint64_t(inputs[0].byte_strides()[1]) : 0;
  // MXFP4 has no expert-dependent input global. Shared source rows are QDQd
  // once per token; ModelOpt/Local still quantize every selected slot.
  const auto quant_slots = mxfp4 && slot_stride == 0 ? 1 : slots;
  const auto quant_rows = mxfp4 ? rows / slots * quant_slots : rows;
  const auto group_size = mxfp4 ? 32U : 16U;
  constexpr auto maximum = std::uint64_t{UINT32_MAX};
  if (rows > maximum || columns > maximum || depth > maximum ||
      slots > maximum || experts > maximum || token_stride > maximum ||
      slot_stride > maximum ||
      limits.maxComputeWorkGroupInvocations < 64 ||
      limits.maxComputeWorkGroupSize[0] < 64 || limits.maxComputeWorkGroupCount[0] == 0 ||
      limits.maxComputeWorkGroupCount[1] == 0)
    return Status::Unimplemented("Vulkan NVFP4 byte indexing or dispatch limit");
  // Check the original carriers, not the hypothetical dense weight bank.
  // The general plan also bounds each actual element count to INT32_MAX;
  // that keeps the shader's group/output products within its U32 index domain.
  const auto check_bytes = [&](const TensorDesc &description) -> Status {
    HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
    HYLE_ASSIGN_OR_RETURN(auto capacity, buffer_capacity(bytes));
    if (capacity > maximum || capacity > limits.maxStorageBufferRange)
      return Status::Unimplemented("Vulkan NVFP4 storage exceeds byte indexing or device limits");
    return Status::Ok();
  };
  for (const auto &description : inputs) HYLE_RETURN_IF_ERROR(check_bytes(description));
  HYLE_RETURN_IF_ERROR(check_bytes(inferred[0]));
  const auto append = [&](const TensorDesc &description) {
    const auto id = static_cast<std::uint32_t>(program.values.size());
    program.values.push_back({description, {}});
    program.aliases.push_back({id, 0});
    return id;
  };
  // A16 reads the original strided BF16 rows directly. Only A4 materializes
  // activation QDQ; neither mode expands a decoded weight bank.
  auto activation = step.inputs[0];
  if (!weight_only) {
    HYLE_ASSIGN_OR_RETURN(auto activation_description, TensorDesc::dense(
        DataType::kBFloat16, std::array{std::int64_t(quant_rows), std::int64_t(depth)}));
    HYLE_RETURN_IF_ERROR(check_bytes(activation_description));
    activation = append(activation_description);
  }
  if (!program.status_value) {
    HYLE_ASSIGN_OR_RETURN(auto status, TensorDesc::dense(DataType::kInt32, {}));
    program.status_value = append(status);
  }
  const auto add_dispatch = [&](std::uint32_t phase,
      std::span<const std::uint32_t> values, std::uint64_t work) -> Status {
    std::array<SpecializationConstant, 9> constants{{
        {0, phase}, {1, std::uint32_t(rows)}, {2, std::uint32_t(columns)},
        {3, std::uint32_t(depth)}, {4, std::uint32_t(slots)},
        {5, std::uint32_t(experts)}, {6, std::uint32_t(token_stride)},
        {7, std::uint32_t(slot_stride)}}};
    std::size_t constant_count = grouped ? 8U : 4U;
    if (local_math) constants[constant_count++] = {8, std::uint32_t(round_input)};
    if (mxfp4 && grouped) constants[constant_count++] = {8, std::uint32_t(quant_slots)};
    Dispatch dispatch;
    const auto bindings = mxfp4 ? (grouped ? 6U : 5U) : (grouped ? 7U : 6U);
    const auto push_bytes = (bindings - 1) * sizeof(std::uint32_t);
    HYLE_ASSIGN_OR_RETURN(dispatch.pipeline, create_pipeline(program.runtime->device,
        {mxfp4 ? (weight_only
                    ? (grouped ? shaders::mxfp4_grouped_weight_only() : shaders::mxfp4_weight_only())
                    : (grouped ? shaders::mxfp4_grouped() : shaders::mxfp4()))
                   : local_math ? (grouped ? shaders::nvfp4_grouped_local() : shaders::nvfp4_local())
                    : (grouped ? shaders::nvfp4_grouped() : shaders::nvfp4()), bindings,
         std::uint32_t(push_bytes), {constants.data(), constant_count}}));
    dispatch.values.assign(values.begin(), values.end());
    dispatch.tensor_byte_origins = true;
    dispatch.checked_status = true;
    dispatch.push.resize(push_bytes);
    const auto groups = (work + 63) / 64;
    const auto x = (std::min)(groups, std::uint64_t(limits.maxComputeWorkGroupCount[0]));
    const auto y = (groups + x - 1) / x;
    if (y > limits.maxComputeWorkGroupCount[1])
      return Status::Unimplemented("Vulkan NVFP4 dispatch grid exceeds native limits");
    dispatch.groups = {std::uint32_t(x), std::uint32_t(y), 1};
    program.dispatches.push_back(std::move(dispatch));
    return Status::Ok();
  };
  const auto status = *program.status_value;
  if (mxfp4) {
    // Validation uses the final output descriptor without writing it in A16.
    // The next dispatch sees its checked status before any dot work.
    const auto validation_output = weight_only ? step.outputs[0] : activation;
    if (grouped) {
      const std::array quantize{step.inputs[0], step.inputs[1], status,
          step.inputs[2], step.inputs[id_port], validation_output};
      HYLE_RETURN_IF_ERROR(add_dispatch(2, quantize, rows));
      HYLE_RETURN_IF_ERROR(add_dispatch(3, quantize, quant_rows * (depth / group_size)));
      const std::array dot{activation, step.inputs[1], status, step.inputs[2],
          step.inputs[id_port], step.outputs[0]};
      return add_dispatch(4, dot, (rows * columns + 1) / 2);
    }
    const std::array quantize{step.inputs[0], step.inputs[1], status,
        step.inputs[2], validation_output};
    HYLE_RETURN_IF_ERROR(add_dispatch(0, quantize,
        (std::max)(quant_rows * (depth / group_size), columns * (depth / group_size))));
    const std::array dot{activation, step.inputs[1], status, step.inputs[2], step.outputs[0]};
    return add_dispatch(1, dot, (rows * columns + 1) / 2);
  }
  if (grouped) {
    const std::array quantize{step.inputs[0], step.inputs[4], status,
        step.inputs[3], step.inputs[2], step.inputs[5], activation};
    // The first dispatch validates all IDs before any selected-expert access.
    HYLE_RETURN_IF_ERROR(add_dispatch(2, quantize, rows));
    HYLE_RETURN_IF_ERROR(add_dispatch(3, quantize, rows * (depth / 16)));
    const std::array dot{activation, step.inputs[1], status, step.inputs[2],
        step.inputs[3], step.inputs[5], step.outputs[0]};
    return add_dispatch(4, dot, (rows * columns + 1) / 2);
  }
  const std::array quantize{step.inputs[0], step.inputs[4], status,
      step.inputs[3], step.inputs[2], activation};
  HYLE_RETURN_IF_ERROR(add_dispatch(0,
      quantize,
      (std::max)(rows * (depth / 16), columns * (depth / 16))));
  const std::array dot{activation, step.inputs[1], status, step.inputs[2],
      step.inputs[3], step.outputs[0]};
  return add_dispatch(1, dot, (rows * columns + 1) / 2);
} catch (const std::bad_alloc &) {
  return Status::ResourceExhausted("Vulkan NVFP4 preparation metadata");
}
} // namespace hyle::compute::vulkan::detail
