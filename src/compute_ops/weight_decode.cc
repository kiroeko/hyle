#include "hyle/compute/model_ops.h"

#include <algorithm>
#include <limits>

namespace hyle::compute::model_ops {
namespace {
bool floating(DataType type) noexcept {
  return type == DataType::kFloat32 || type == DataType::kFloat16 ||
         type == DataType::kBFloat16;
}
bool shared_grouped_activation(const TensorDesc &desc) noexcept {
  if (!desc.valid() || !desc.is_plain() || desc.rank() != 3 ||
      !floating(desc.data_type()))
    return false;
  const auto shape = desc.dimensions(), stride = desc.byte_strides();
  const auto width = static_cast<std::int64_t>(data_type_size(desc.data_type()));
  return shape[0] > 0 && shape[1] > 0 && shape[2] > 0 &&
      stride[2] == width && stride[1] == 0 && stride[0] % width == 0 &&
      (shape[0] == 1 || stride[0] / width >= shape[2]);
}
bool shaped(const TensorDesc &desc, DataType type,
            std::initializer_list<std::int64_t> dimensions) noexcept {
  return desc.data_type() == type && desc.rank() == dimensions.size() &&
         std::equal(dimensions.begin(), dimensions.end(), desc.dimensions().begin());
}
Status infer_fp4_dense(std::span<const TensorDesc> inputs,
    std::span<TensorDesc> outputs, bool microscaled = false) noexcept {
  if (inputs.size() != (microscaled ? 3u : 5u) || outputs.size() != 1)
    return Status::InvalidArgument("FP4 reference linear input/output count differs");
  const auto scale_type = microscaled ? DataType::kUInt8 : DataType::kFloat8E4M3FN;
  const std::int64_t group = microscaled ? 32 : 16;
  for (const auto &input : inputs) {
    if (!input.valid() || !input.is_plain() || !input.is_dense())
      return Status::InvalidArgument("NVFP4 reference linear needs plain dense typed storage");
    auto bytes = input.required_bytes();
    if (!bytes) return std::move(bytes).error();
  }
  const auto &a = inputs[0], &w = inputs[1], &scale = inputs[2];
  if (a.rank() != 2 || w.rank() != 2 || scale.rank() != 2 ||
      a.data_type() != DataType::kBFloat16 ||
      w.data_type() != DataType::kUInt8 || scale.data_type() != scale_type)
    return Status::InvalidArgument("FP4 reference linear activation/weight/scale types differ");
  const auto m = a.dimensions()[0], k = a.dimensions()[1], n = w.dimensions()[0];
  if (m <= 0 || n <= 0 || k <= 0 || k % group != 0 ||
      w.dimensions()[1] != k / 2 || scale.dimensions()[0] != n ||
      scale.dimensions()[1] != k / group)
    return Status::InvalidArgument("FP4 reference linear requires matching positive M/N and complete K groups");
  for (std::size_t port = 3; port < inputs.size(); ++port) {
    const auto &global = inputs[port];
    if (global.data_type() != DataType::kFloat32 || global.rank() > 1 ||
        global.element_count().value() != 1)
      return Status::InvalidArgument("NVFP4 reference linear needs independent F32 weight/input scalars");
  }
  HYLE_ASSIGN_OR_RETURN(auto output, TensorDesc::dense(DataType::kBFloat16,
                                                       std::array{m,n}));
  auto bytes = output.required_bytes();
  if (!bytes) return std::move(bytes).error();
  outputs[0] = std::move(output);
  return Status::Ok();
}

Status infer_fp4_grouped(
    std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs,
    bool microscaled = false) noexcept {
  if (inputs.size() != (microscaled ? 4u : 6u) || outputs.size() != 1)
    return Status::InvalidArgument("grouped FP4 reference input/output count differs");
  const auto scale_type = microscaled ? DataType::kUInt8 : DataType::kFloat8E4M3FN;
  const std::int64_t group = microscaled ? 32 : 16;
  for (std::size_t port = 0; port < inputs.size(); ++port) {
    const auto &input = inputs[port];
    if (!input.valid() || !input.is_plain() ||
        (!input.is_dense() && !(port == 0 && shared_grouped_activation(input))))
      return Status::InvalidArgument("grouped NVFP4 needs dense ports or a shared activation view");
    auto bytes = input.required_bytes();
    if (!bytes) return std::move(bytes).error();
  }
  const auto &a = inputs[0], &w = inputs[1], &scale = inputs[2], &ids = inputs.back();
  if (a.data_type() != DataType::kBFloat16 || a.rank() != 3 ||
      w.data_type() != DataType::kUInt8 || w.rank() != 3 ||
      scale.data_type() != scale_type || scale.rank() != 3 ||
      ids.data_type() != DataType::kInt32 || ids.rank() != 2)
    return Status::InvalidArgument("grouped NVFP4 typed activation/weight/scale/ID ranks differ");
  const auto t = a.dimensions()[0], k = a.dimensions()[2];
  const auto e = w.dimensions()[0], n = w.dimensions()[1], s = ids.dimensions()[1];
  if (t <= 0 || s <= 0 || e <= 0 || n <= 0 || k <= 0 || k % group != 0 ||
      ids.dimensions()[0] != t ||
      (a.dimensions()[1] != 1 && a.dimensions()[1] != s) ||
      w.dimensions()[2] != k / 2 ||
      !shaped(scale, scale_type, {e,n,k/group}) ||
      (!microscaled && (!shaped(inputs[3], DataType::kFloat32, {e}) ||
                       !shaped(inputs[4], DataType::kFloat32, {e}))))
    return Status::InvalidArgument("grouped FP4 requires matching positive T/S/E/N and complete K groups");
  HYLE_ASSIGN_OR_RETURN(auto output,
      TensorDesc::dense(DataType::kBFloat16, std::array{t,s,n}));
  auto bytes = output.required_bytes();
  if (!bytes) return std::move(bytes).error();
  outputs[0] = std::move(output);
  return Status::Ok();
}

bool local_fp4_rounding(Fp4InputRounding rounding) noexcept {
  return rounding == Fp4InputRounding::kFloat32 ||
         rounding == Fp4InputRounding::kBFloat16;
}
} // namespace

Status Nvfp4Linear::infer_output_descriptions(
    std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept {
  return infer_fp4_dense(inputs, outputs);
}
Status GroupedNvfp4Linear::infer_output_descriptions(
    std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept {
  return infer_fp4_grouped(inputs, outputs);
}
Status LocalNvfp4Linear::infer_output_descriptions(
    std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept {
  if (!local_fp4_rounding(rounding))
    return Status::InvalidArgument("local NVFP4 input rounding is invalid");
  return infer_fp4_dense(inputs, outputs);
}
Status GroupedLocalNvfp4Linear::infer_output_descriptions(
    std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept {
  if (!local_fp4_rounding(rounding))
    return Status::InvalidArgument("local NVFP4 input rounding is invalid");
  return infer_fp4_grouped(inputs, outputs);
}

Status Mxfp4Linear::infer_output_descriptions(
    std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept {
  if (activation_mode != Mxfp4ActivationMode::kFloat4Group32 &&
      activation_mode != Mxfp4ActivationMode::kBFloat16)
    return Status::InvalidArgument("MXFP4 activation mode is invalid");
  return infer_fp4_dense(inputs, outputs, true);
}
Status GroupedMxfp4Linear::infer_output_descriptions(
    std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept {
  if (activation_mode != Mxfp4ActivationMode::kFloat4Group32 &&
      activation_mode != Mxfp4ActivationMode::kBFloat16)
    return Status::InvalidArgument("MXFP4 activation mode is invalid");
  return infer_fp4_grouped(inputs, outputs, true);
}

Status MlxQuantized::infer_output_descriptions(std::span<const TensorDesc> inputs,
    std::span<TensorDesc> outputs) const noexcept {
  if ((format != MlxWeightFormat::kAffine && format != MlxWeightFormat::kMxfp4 &&
       format != MlxWeightFormat::kMxfp8) ||
      (kind != MlxQuantizedKind::kLinear && kind != MlxQuantizedKind::kGrouped &&
       kind != MlxQuantizedKind::kEmbedding) || group_size == 0 ||
      (orientation != MlxMatmulOrientation::kTransposeWeight &&
       orientation != MlxMatmulOrientation::kWeightAsStored) ||
      inputs.size() != operation_traits().inputs.minimum || outputs.size() != 1)
    return Status::InvalidArgument("MLX packed operation format/ports/group are invalid");
  const bool affine = format == MlxWeightFormat::kAffine;
  const bool embedding = kind == MlxQuantizedKind::kEmbedding;
  const bool grouped = kind == MlxQuantizedKind::kGrouped;
  const bool transpose_weight = orientation == MlxMatmulOrientation::kTransposeWeight;
  if (!transpose_weight && (!affine || embedding))
    return Status::InvalidArgument("MLX as-stored multiplication requires affine linear or grouped weights");
  if ((affine && bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
      (!affine && bits != (format == MlxWeightFormat::kMxfp8 ? 8u : 4u)) ||
      (!affine && group_size != 32) ||
      (affine && group_size != 32 && group_size != 64 && group_size != 128) ||
      (embedding && !affine))
    return Status::InvalidArgument("MLX format requires its declared bits and scale grouping");
  for (std::size_t port = 0; port < inputs.size(); ++port) {
    const auto &input = inputs[port];
    if (!input.valid() || !input.is_plain() ||
        (!input.is_dense() &&
         !(grouped && port == 0 && shared_grouped_activation(input))))
      return Status::InvalidArgument(
          "MLX packed ports require dense storage or a shared grouped activation view");
  }
  const auto &w = inputs[embedding ? 0 : 1];
  const auto &scale = inputs[embedding ? 1 : 2];
  const std::size_t rank = grouped ? 3 : 2;
  if (w.rank() != rank || scale.rank() != rank || w.data_type() != DataType::kUInt32 ||
      (affine ? !floating(scale.data_type()) : scale.data_type() != DataType::kUInt8))
    return Status::InvalidArgument("MLX original word/scale dtype or rank differs");
  for (const auto d : w.dimensions())
    if (d <= 0) return Status::InvalidArgument("MLX packed weight dimensions must be positive");
  const auto groups = scale.dimensions().back();
  if (groups <= 0)
    return Status::InvalidArgument("MLX scales require complete positive groups");
  if (groups > INT64_MAX / group_size)
    return Status::OutOfRange("MLX scale groups overflow K");
  const auto k = groups * group_size;
  const auto n = w.dimensions()[rank - 2];
  // All admitted groups contain whole32-code blocks; this multiplication
  // cannot overflow after checked K, even for8bit rows.
  if (w.dimensions().back() != (k / 32) * bits)
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
  const auto reduction = transpose_weight ? k : n;
  const auto output_width = transpose_weight ? n : k;
  if (a.rank() != rank || !floating(a.data_type()) || a.dimensions().back() != reduction)
    return Status::InvalidArgument("MLX activation rank, floating dtype or reduction extent differs");
  for (const auto d : a.dimensions())
    if (d <= 0) return Status::InvalidArgument("MLX activation dimensions must be positive");
  dtype = a.data_type();
  if (affine && scale.data_type() != dtype) dtype = DataType::kFloat32;
  if (grouped) {
    const auto &ids = inputs.back();
    if (ids.data_type() != DataType::kInt32 || ids.rank() != 2 ||
        ids.dimensions()[0] != a.dimensions()[0] || ids.dimensions()[1] <= 0 ||
        (a.dimensions()[1] != 1 && a.dimensions()[1] != ids.dimensions()[1]))
      return Status::InvalidArgument("MLX grouped activation must share or match the [T,S] expert ids");
    auto out = TensorDesc::dense(dtype, std::array{a.dimensions()[0], ids.dimensions()[1], output_width});
    if (!out) return std::move(out).error();
    outputs[0] = *out;
  } else {
    auto out = TensorDesc::dense(dtype, std::array{a.dimensions()[0], output_width});
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
      format != WeightArtifactFormat::kCompressedTensorsSymmetric4 &&
      format != WeightArtifactFormat::kCompressedTensorsAsymmetric4 &&
      format != WeightArtifactFormat::kGptqV1Int4)
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
  if (format == WeightArtifactFormat::kGptqV1Int4) {
    const auto packed = inputs[0].dimensions();
    if (packed[0] > std::numeric_limits<std::int64_t>::max() / 8)
      return Status::OutOfRange("GPTQ packed K overflows");
    const auto k = packed[0] * 8, n = packed[1];
    if (k % group_size != 0 || n % 8 != 0 ||
        inputs[0].data_type() != DataType::kInt32 ||
        !shaped(inputs[1], DataType::kInt32, {k / group_size, n / 8}) ||
        !shaped(inputs[2], DataType::kFloat16, {k / group_size, n}) ||
        !shaped(inputs[3], DataType::kInt32, {k}))
      return Status::InvalidArgument("GPTQ-v1 needs complete groups and original Q/Z/F16 scale/index layouts");
    for (const auto &input : inputs) {
      auto bytes = input.required_bytes();
      if (!bytes) return std::move(bytes).error();
    }
    HYLE_ASSIGN_OR_RETURN(auto output, TensorDesc::dense(
        output_type == DataType::kUnknown ? DataType::kFloat16 : output_type,
        std::array{n,k}));
    auto bytes = output.required_bytes();
    if (!bytes) return std::move(bytes).error();
    outputs[0] = std::move(output);
    return Status::Ok();
  }
  const bool nf4 = format == WeightArtifactFormat::kNf4 ||
                   format == WeightArtifactFormat::kNf4DoubleQuant;
  const bool awq = format == WeightArtifactFormat::kAwqGemm4;
  const bool asymmetric = format == WeightArtifactFormat::kCompressedTensorsAsymmetric4;
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
    const auto &scale = inputs[awq || asymmetric ? 2 : 1];
    if (inputs[0].data_type() != DataType::kInt32 || !floating(scale.data_type()) ||
        scale.rank() != 2 || scale.dimensions()[0] != (awq ? k / group_size : n) ||
        scale.dimensions()[1] != (awq ? n : k / group_size))
      return Status::InvalidArgument("original int4 word/scale layout differs");
    if (awq && !shaped(inputs[1], DataType::kInt32, {k / group_size, n / 8}))
      return Status::InvalidArgument("AWQ packed zero-point layout differs");
    if (asymmetric && !shaped(inputs[1], DataType::kInt32,
                              {n / 8 + (n % 8 != 0), k / group_size}))
      return Status::InvalidArgument("CT packed zero-point output-axis layout differs");
    if (dtype == DataType::kUnknown) dtype = scale.data_type();
  }
  output = TensorDesc::dense(dtype, std::array{n, k});
  if (!output) return std::move(output).error();
  outputs[0] = *output;
  return Status::Ok();
}
} // namespace hyle::compute::model_ops
