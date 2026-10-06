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

Status MlxQuantized::infer_output_descriptions(std::span<const TensorDesc> inputs,
    std::span<TensorDesc> outputs) const noexcept {
  if ((format != MlxWeightFormat::kAffine4 && format != MlxWeightFormat::kMxfp4 &&
       format != MlxWeightFormat::kMxfp8) ||
      (kind != MlxQuantizedKind::kLinear && kind != MlxQuantizedKind::kGrouped &&
       kind != MlxQuantizedKind::kEmbedding) || group_size == 0 ||
      inputs.size() != operation_traits().inputs.minimum || outputs.size() != 1)
    return Status::InvalidArgument("MLX packed operation format/ports/group are invalid");
  const bool affine = format == MlxWeightFormat::kAffine4;
  const bool embedding = kind == MlxQuantizedKind::kEmbedding;
  const bool grouped = kind == MlxQuantizedKind::kGrouped;
  if ((!affine && group_size != 32) ||
      (affine && group_size != 32 && group_size != 64 && group_size != 128) ||
      (embedding && !affine))
    return Status::InvalidArgument("MLX format requires its original scale grouping");
  for (const auto &input : inputs)
    if (!input.valid() || !input.is_plain() || !input.is_dense())
      return Status::InvalidArgument("MLX packed ports require plain dense storage");
  const auto &w = inputs[embedding ? 0 : 1];
  const auto &scale = inputs[embedding ? 1 : 2];
  const std::size_t rank = grouped ? 3 : 2;
  if (w.rank() != rank || scale.rank() != rank || w.data_type() != DataType::kUInt32 ||
      (affine ? !floating(scale.data_type()) : scale.data_type() != DataType::kUInt8))
    return Status::InvalidArgument("MLX original word/scale dtype or rank differs");
  for (const auto d : w.dimensions())
    if (d <= 0) return Status::InvalidArgument("MLX packed weight dimensions must be positive");
  const std::int64_t pack = format == MlxWeightFormat::kMxfp8 ? 4 : 8;
  if (w.dimensions().back() > INT64_MAX / pack)
    return Status::OutOfRange("MLX unpacked K overflows");
  const auto k = w.dimensions().back() * pack;
  const auto n = w.dimensions()[rank - 2];
  if (k % group_size != 0 || scale.dimensions().back() != k / group_size)
    return Status::InvalidArgument("MLX codes and complete scale groups differ");
  for (std::size_t axis = 0; axis + 1 < rank; ++axis)
    if (scale.dimensions()[axis] != w.dimensions()[axis])
      return Status::InvalidArgument("MLX weight/scale leading axes differ");
  if (affine && inputs[embedding ? 2 : 3] != scale)
    return Status::InvalidArgument("MLX affine biases must match scale shape and dtype");
  DataType dtype = scale.data_type();
  if (embedding) {
    const auto &ids = inputs.back();
    if (ids.rank() != 1 || ids.data_type() != DataType::kInt32)
      return Status::InvalidArgument("MLX embedding requires checked I32 row ids");
    auto out = TensorDesc::dense(dtype, std::array{ids.dimensions()[0], k});
    if (!out) return std::move(out).error();
    outputs[0] = *out;
    return Status::Ok();
  }
  const auto &a = inputs[0];
  if (a.rank() != rank || !floating(a.data_type()) || a.dimensions().back() != k)
    return Status::InvalidArgument("MLX activation rank, floating dtype or K differs");
  for (const auto d : a.dimensions())
    if (d <= 0) return Status::InvalidArgument("MLX activation dimensions must be positive");
  dtype = a.data_type();
  if (affine && scale.data_type() != dtype) dtype = DataType::kFloat32;
  if (grouped) {
    if (!shaped(inputs.back(), DataType::kInt32,
                 {a.dimensions()[0], a.dimensions()[1]}))
      return Status::InvalidArgument("MLX selected expert ids must match [T,S]");
    auto out = TensorDesc::dense(dtype, std::array{a.dimensions()[0], a.dimensions()[1], n});
    if (!out) return std::move(out).error();
    outputs[0] = *out;
  } else {
    auto out = TensorDesc::dense(dtype, std::array{a.dimensions()[0], n});
    if (!out) return std::move(out).error();
    outputs[0] = *out;
  }
  return Status::Ok();
}

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
