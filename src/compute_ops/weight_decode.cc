#include "hyle/compute/model_ops.h"

#include <algorithm>
#include <limits>

namespace hyle::compute::model_ops {
namespace {
bool floating(DataType type) noexcept {
  return type == DataType::kFloat32 || type == DataType::kFloat16 ||
         type == DataType::kBFloat16;
}
bool shaped(const TensorDesc &desc, DataType type,
            std::initializer_list<std::int64_t> dimensions) noexcept {
  return desc.data_type() == type && desc.rank() == dimensions.size() &&
         std::equal(dimensions.begin(), dimensions.end(), desc.dimensions().begin());
}
} // namespace

Status ArtifactWeightDecode::infer_output_descriptions(
    std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept {
  if (format != WeightArtifactFormat::kNf4 &&
      format != WeightArtifactFormat::kNf4DoubleQuant &&
      format != WeightArtifactFormat::kAwqGemm4 &&
      format != WeightArtifactFormat::kCompressedTensorsSymmetric4)
    return Status::InvalidArgument("unknown original weight artifact format");
  if (inputs.size() != operation_traits().inputs.minimum || outputs.size() != 1 ||
      group_size == 0 || (output_type != DataType::kUnknown && !floating(output_type)))
    return Status::InvalidArgument("artifact decode ports, group or output dtype are invalid");
  for (const auto &input : inputs)
    if (!input.valid() || !input.is_plain() || !input.is_dense())
      return Status::InvalidArgument("artifact decode requires plain dense typed payloads");
  if (inputs[0].rank() != 2 || inputs[0].dimensions()[0] <= 0 ||
      inputs[0].dimensions()[1] <= 0)
    return Status::InvalidArgument("artifact decode requires a positive packed matrix");
  const bool nf4 = format == WeightArtifactFormat::kNf4 ||
                   format == WeightArtifactFormat::kNf4DoubleQuant;
  const bool awq = format == WeightArtifactFormat::kAwqGemm4;
  const auto packed = inputs[0].dimensions();
  const auto multiplier = nf4 ? 2 : 8;
  if (packed[1] > std::numeric_limits<std::int64_t>::max() / multiplier)
    return Status::OutOfRange("artifact packed width overflows");
  const auto n = awq ? packed[1] * multiplier : packed[0];
  const auto k = awq ? packed[0] : packed[1] * multiplier;
  if (k % group_size != 0)
    return Status::InvalidArgument("artifact K does not contain complete scale groups");
  auto output = TensorDesc::dense(DataType::kFloat32, std::array{n, k});
  if (!output) return std::move(output).error();
  const auto elements = output->element_count();
  if (!elements || *elements > static_cast<std::uint64_t>(INT64_MAX))
    return Status::OutOfRange("artifact logical capacity exceeds int64");
  const auto groups = static_cast<std::int64_t>(*elements / group_size);
  DataType dtype = output_type;
  if (nf4) {
    if (inputs[0].data_type() != DataType::kUInt8)
      return Status::InvalidArgument("NF4 codes require original U8 high-first packing");
    const bool nested = format == WeightArtifactFormat::kNf4DoubleQuant;
    if (nested) {
      if (!shaped(inputs[1], DataType::kUInt8, {groups}) ||
          !shaped(inputs[2], DataType::kFloat32, {groups / 256 + (groups % 256 != 0)}) ||
          !shaped(inputs[3], DataType::kFloat32, {256}) ||
          inputs[4].data_type() != DataType::kFloat32 ||
          inputs[4].element_count().value() != 1 ||
          !shaped(inputs[5], DataType::kFloat32, {16}))
        return Status::InvalidArgument("nested NF4 absmax/codebook/offset shapes differ");
    } else if (!shaped(inputs[1], DataType::kFloat32, {groups}) ||
               !shaped(inputs[2], DataType::kFloat32, {16})) {
      return Status::InvalidArgument("NF4 absmax/codebook shapes differ");
    }
    if (dtype == DataType::kUnknown) dtype = DataType::kBFloat16;
  } else {
    const auto &scale = inputs[awq ? 2 : 1];
    if (inputs[0].data_type() != DataType::kInt32 || !floating(scale.data_type()) ||
        scale.rank() != 2 || scale.dimensions()[0] != (awq ? k / group_size : n) ||
        scale.dimensions()[1] != (awq ? n : k / group_size))
      return Status::InvalidArgument("original int4 word/scale layout differs");
    if (awq && !shaped(inputs[1], DataType::kInt32, {k / group_size, n / 8}))
      return Status::InvalidArgument("AWQ packed zero-point layout differs");
    if (dtype == DataType::kUnknown) dtype = scale.data_type();
  }
  output = TensorDesc::dense(dtype, std::array{n, k});
  if (!output) return std::move(output).error();
  outputs[0] = *output;
  return Status::Ok();
}
} // namespace hyle::compute::model_ops
