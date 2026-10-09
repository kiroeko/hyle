#include "vulkan_official.h"

#include <algorithm>
#include <new>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "vulkan_mlx.h"
#include "vulkan_tensor.h"

namespace hyle::compute::vulkan::detail {
namespace iv = iree_vulkan;
namespace {
bool encoded(const TensorDesc &a, const TensorDesc &w) noexcept {
  if (a.rank() != 2 || w.rank() != 2) return false;
  const auto mode = Float8ScaleMode::kFloat32Block128;
  const auto expected_a = float8_activation_description(a.dimensions()[0], a.dimensions()[1], mode);
  const auto expected_w = float8_weight_description(w.dimensions()[0], w.dimensions()[1], mode);
  return expected_a && expected_w && a == *expected_a && w == *expected_w;
}

std::uint32_t append(ProgramState &program, const TensorDesc &description) {
  const auto id = static_cast<std::uint32_t>(program.values.size());
  program.values.push_back({description, {}});program.aliases.push_back({id, 0});return id;
}

bool fits(const Device &device, const iv::ProgramSpec &spec) noexcept {
  const auto &limits = device.properties.limits;
  if (limits.maxComputeWorkGroupInvocations < 64 || limits.maxComputeWorkGroupSize[0] < 64 ||
      limits.maxComputeWorkGroupCount[1] < 1 || limits.maxComputeWorkGroupCount[2] < 1) return false;
  constexpr auto maximum = std::uint64_t{INT32_MAX} / 8;
  const auto r = static_cast<std::uint64_t>(spec.rows), n = static_cast<std::uint64_t>(spec.columns),
      k = static_cast<std::uint64_t>(spec.depth), e = static_cast<std::uint64_t>(spec.experts);
  const auto product = [](std::uint64_t a, std::uint64_t b) {
    return a > 0 && b > 0 && a <= maximum / b;
  };
  if (!product(r, n)) return false;
  const bool decode = spec.kind == iv::ProgramKind::OfficialWeightDecodeF32 ||
                       spec.kind == iv::ProgramKind::OfficialWeightDecodeBf16;
  const bool roundtrip = iv::official_roundtrip(spec.kind);
  std::uint64_t work = r * n, bytes = 32 * work + 512;
  if (!decode && !roundtrip) {
    const auto group = spec.kind == iv::ProgramKind::OfficialGrouped4Bf16 ? 32U : 128U;
    if (!product(r, k) || !product(e, n) || !product(e * n, k) ||
        !product(r * n, group)) return false;
    work = r * n * group;
    // Quantized A, source wide/scale, selected current group, dot/accumulators,
    // typed private output and flag: no full float weight/expert bank.
    bytes = 32 * r * k + 4 * work + 4 * r * group + 48 * r * n + 1024;
  }
  return bytes <= limits.maxStorageBufferRange &&
      (std::min)((work + 63) / 64, std::uint64_t{65535}) <= limits.maxComputeWorkGroupCount[0];
}
} // namespace

bool official_float_operation(const ProgramState &program, const PlanStep &step) noexcept {
  if (step.custom || step.outputs.size() != 1) return false;
  if (step.operation.custom_as<model_ops::Float8WeightDecode>() ||
      step.operation.custom_as<model_ops::Float8BlockLinear>() ||
      step.operation.custom_as<model_ops::GroupedFloat8Linear>() ||
      step.operation.custom_as<model_ops::GroupedFloat4Linear>() ||
      step.operation.custom_as<model_ops::BlockQuantizationRoundtrip>()) return true;
  return step.operation.get_if<MatMul>() && step.inputs.size() == 2 &&
      encoded(program.values[step.inputs[0]].description, program.values[step.inputs[1]].description);
}

Status lower_official_float(ProgramState &program, const PlanStep &step) noexcept {
  try {
    const auto output = program.values[step.outputs[0]].description;
    const auto &first = program.values[step.inputs[0]].description;
    iv::ProgramSpec spec;
    const bool bf16 = output.data_type() == DataType::kBFloat16;
    spec.output_type = bf16 ? iv::ArtifactScalar::BFloat16 : iv::ArtifactScalar::Float32;
    const auto roundtrip = step.operation.custom_as<model_ops::BlockQuantizationRoundtrip>();
    const auto decode = step.operation.custom_as<model_ops::Float8WeightDecode>();
    const auto linear = step.operation.custom_as<model_ops::Float8BlockLinear>();
    const auto grouped8 = step.operation.custom_as<model_ops::GroupedFloat8Linear>();
    const bool grouped4 = step.operation.custom_as<model_ops::GroupedFloat4Linear>().has_value();
    if (roundtrip) {
      using Format = model_ops::BlockQuantizationFormat;
      spec.kind = roundtrip->format == Format::kFloat4E2M1Block32 ? iv::ProgramKind::OfficialRoundtrip4Group32
          : roundtrip->format == Format::kFloat8E4M3Block64 ? iv::ProgramKind::OfficialRoundtrip8Group64
                                                         : iv::ProgramKind::OfficialRoundtrip8Group128;
      spec.rows = first.dimensions()[0];spec.columns = first.dimensions()[1];spec.depth = spec.columns;
    } else if (decode) {
      spec.kind = bf16 ? iv::ProgramKind::OfficialWeightDecodeBf16 : iv::ProgramKind::OfficialWeightDecodeF32;
      spec.rows = output.dimensions()[0];spec.columns = output.dimensions()[1];spec.depth = spec.columns;
    } else if (grouped8 || grouped4) {
      const auto &a = program.values[step.inputs[1]].description;
      const auto &ids = program.values[step.inputs[2]].description;
      spec.kind = grouped4 ? iv::ProgramKind::OfficialGrouped4Bf16
          : bf16 ? iv::ProgramKind::OfficialGrouped8Bf16 : iv::ProgramKind::OfficialGrouped8F32;
      spec.experts = first.dimensions()[0];spec.columns = first.dimensions()[1];spec.depth = first.dimensions()[2];
      spec.selection_slots = ids.dimensions()[1];spec.activation_slots = a.dimensions()[1];
      spec.rows = a.dimensions()[0] * spec.selection_slots;
      spec.absmax_floor = grouped4 ? 1.0e-4F : grouped8->absmax_floor;
    } else {
      const auto &w = program.values[step.inputs[1]].description;
      spec.rows = first.dimensions()[0];spec.depth = first.dimensions()[1];spec.columns = w.dimensions()[1];
      if (linear) {
        const bool power = linear->scale_rounding == model_ops::Float8ScaleRounding::kPowerOfTwo;
        spec.kind = power ? (bf16 ? iv::ProgramKind::OfficialLinearPowerBf16 : iv::ProgramKind::OfficialLinearPowerF32)
                          : (bf16 ? iv::ProgramKind::OfficialLinearBf16 : iv::ProgramKind::OfficialLinearF32);
        spec.absmax_floor = linear->absmax_floor;
      } else {
        if (step.operation.get_if<MatMul>()->accumulation_type != DataType::kFloat32)
          return Status::Unimplemented("Vulkan encoded block128 MatMul requires F32 accumulation");
        spec.kind = iv::ProgramKind::OfficialEncodedBlock128;
      }
    }
    const auto &features = program.runtime->device->provider.native();
    if (!features.shader_int16 || !features.shader_float16 || !features.storage_buffer16 ||
        !features.preserve_float32_specials || !fits(*program.runtime->device, spec))
      return Status::Unimplemented("Vulkan official float numeric features, workspace or index domain");
    HYLE_RETURN_IF_ERROR(ensure_iree_device(program));
    auto inputs = step.inputs;auto destination = step.outputs[0];
    if (grouped8 || grouped4) {
      if (!program.status_value) {
        HYLE_ASSIGN_OR_RETURN(auto status, TensorDesc::dense(DataType::kInt32, {}));
        program.status_value = append(program, status);
      }
      const auto ids = program.values[inputs.back()].description;
      const auto safe = append(program, ids);
      HYLE_RETURN_IF_ERROR(lower_index_check(program, inputs.back(), safe, static_cast<std::uint64_t>(spec.experts), true));
      inputs.back() = safe;destination = append(program, output);
    }
    Dispatch dispatch;dispatch.iree = spec;dispatch.values = std::move(inputs);
    dispatch.values.push_back(destination);program.dispatches.push_back(std::move(dispatch));
    return (grouped8 || grouped4) ? lower_checked_copy(program, destination, step.outputs[0]) : Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan official float preparation metadata");
  }
}
} // namespace hyle::compute::vulkan::detail
