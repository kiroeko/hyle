#include "vulkan_program_internal.h"
#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <new>

namespace hyle::compute::vulkan::detail {
namespace {
bool same_shape(const TensorDesc &left, const TensorDesc &right) noexcept {
  return left.rank() == right.rank() &&
         std::ranges::equal(left.dimensions(), right.dimensions());
}
} // namespace

bool managed_elementwise_limits(const ProgramState &program,
                                const PlanStep &step) noexcept {
  if (const auto *unary = step.operation.get_if<Unary>()) {
    switch (unary->kind) {
    case UnaryKind::kAbs:
    case UnaryKind::kNeg:
    case UnaryKind::kExp:
    case UnaryKind::kLog:
    case UnaryKind::kSqrt:
    case UnaryKind::kRsqrt:
    case UnaryKind::kTanh:
    case UnaryKind::kSigmoid:
    case UnaryKind::kSilu:
    case UnaryKind::kSoftplus:
      break;
    default:
      return false;
    }
  } else if (const auto *binary = step.operation.get_if<Binary>()) {
    if (binary->kind != BinaryKind::kAdd &&
        binary->kind != BinaryKind::kSubtract &&
        binary->kind != BinaryKind::kMultiply &&
        binary->kind != BinaryKind::kDivide)
      return false;
  } else if (const auto *gate = step.operation.get_if<GatedActivation>()) {
    if (gate->activation != GatedActivationKind::kSilu)
      return false;
  } else {
    return false;
  }
  const auto *unary = step.operation.get_if<Unary>();
  const bool fill = unary && unary->kind == UnaryKind::kRsqrt;
  const auto &limits = program.runtime->device->properties.limits;
  const auto fits = [&](std::uint32_t value) {
    const auto &description = program.values[value].description;
    const auto type = description.data_type();
    const auto count = description.element_count();
    return (type == DataType::kFloat32 || type == DataType::kFloat16 ||
            type == DataType::kBFloat16) &&
           count && *count > 0 && *count <= INT32_MAX &&
           *count <= limits.maxStorageBufferRange / sizeof(float) &&
           (std::min)((*count + 511) / 512, std::size_t{512}) <=
               limits.maxComputeWorkGroupCount[1] &&
           (*count + 262143) / 262144 <= limits.maxComputeWorkGroupCount[2] &&
           (!fill || (*count + 511) / 512 <=
                         std::uint64_t{limits.maxComputeWorkGroupCount[0]} *
                             limits.maxComputeWorkGroupCount[1]);
  };
  // Original Unary/Log/GLU and the Fill used by Rsqrt use 512 lanes. Flattened
  // dispatches split the domain into 512 x 512 element tiles.
  return limits.maxComputeWorkGroupInvocations >= 512 &&
         limits.maxComputeWorkGroupSize[0] >= 512 &&
         limits.maxComputeWorkGroupCount[0] != 0 &&
         std::ranges::all_of(step.inputs, fits) &&
         std::ranges::all_of(step.outputs, fits);
}

Status lower_managed_elementwise(ProgramState &program,
                                 const PlanStep &step) noexcept {
  try {
    const auto output = program.values[step.outputs[0]].description;
    const auto *gate = step.operation.get_if<GatedActivation>();
    const bool packed = gate && gate->inputs == GatedInputKind::kPackedHalves;
    const bool flatten = output.rank() > 4;
    const auto count = static_cast<std::int64_t>(*output.element_count());
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
    const auto cast = [&](std::uint32_t source, std::uint32_t destination) {
      return lower_tensor(
          program,
          {Operation(Cast{program.values[destination].description.data_type()}),
           {source},
           {destination},
           {},
           {1, 1, 1},
           {}});
    };
    auto normalized = output;
    if (flatten) {
      // Packed GLU halves are split within each row, never across the flattened
      // tensor. Retain that row boundary without materializing two halves.
      const auto width = packed ? output.dimensions().back() : count;
      HYLE_ASSIGN_OR_RETURN(
          normalized, TensorDesc::dense(DataType::kFloat32,
                                        std::array{count / width, width}));
    } else {
      HYLE_ASSIGN_OR_RETURN(normalized, TensorDesc::dense(DataType::kFloat32,
                                                          output.dimensions()));
    }
    std::vector<std::uint32_t> inputs;
    for (auto source : step.inputs) {
      auto description = program.values[source].description;
      bool broadcast = false;
      if (step.operation.get_if<Binary>()) {
        // ggml broadcasts only RHS and needs one operand to have the result
        // shape. Preserve low-rank vendor broadcasting whenever possible.
        broadcast =
            flatten ? (*description.element_count() != 1 &&
                       !same_shape(description, output))
                    : std::ranges::none_of(
                          step.inputs,
                          [&](auto value) {
                            return same_shape(program.values[value].description,
                                              output);
                          }) &&
                          inputs.empty();
      }
      if (broadcast) {
        HYLE_ASSIGN_OR_RETURN(auto shape,
                              TensorShape::create(output.dimensions()));
        HYLE_ASSIGN_OR_RETURN(auto view, View::broadcast_to(shape));
        HYLE_ASSIGN_OR_RETURN(
            description,
            infer_single_output(Operation(view), std::span(&description, 1)));
        source = alias(description, source);
      }
      if (!description.is_dense() ||
          description.data_type() != DataType::kFloat32) {
        HYLE_ASSIGN_OR_RETURN(
            auto dense,
            TensorDesc::dense(DataType::kFloat32, description.dimensions()));
        const auto converted = append(dense);
        HYLE_RETURN_IF_ERROR(cast(source, converted));
        source = converted;
        description = dense;
      }
      if (flatten) {
        auto shape = normalized;
        if (packed) {
          HYLE_ASSIGN_OR_RETURN(
              shape,
              TensorDesc::dense(DataType::kFloat32,
                                std::array{normalized.dimensions()[0],
                                           normalized.dimensions()[1] * 2}));
        } else if (step.operation.get_if<Binary>() &&
                   *description.element_count() == 1 &&
                   !same_shape(description, output)) {
          HYLE_ASSIGN_OR_RETURN(shape,
                                TensorDesc::dense(DataType::kFloat32, {}));
        }
        source = alias(shape, source);
      }
      inputs.push_back(source);
    }
    const bool direct = output.data_type() == DataType::kFloat32;
    const auto result =
        direct ? alias(normalized, step.outputs[0]) : append(normalized);
    Plan local;
    local.outputs = {result};
    local.steps.push_back(
        {step.operation, std::move(inputs), {result}, {}, {1, 1, 1}, {}});
    const std::vector<std::size_t> last_uses(program.values.size(), 0);
    HYLE_RETURN_IF_ERROR(lower_region(program, local, 0, 1, last_uses));
    if (!direct) {
      HYLE_ASSIGN_OR_RETURN(
          auto restored,
          TensorDesc::dense(DataType::kFloat32, output.dimensions()));
      HYLE_RETURN_IF_ERROR(cast(alias(restored, result), step.outputs[0]));
    }
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan elementwise adapter metadata");
  }
}
} // namespace hyle::compute::vulkan::detail
