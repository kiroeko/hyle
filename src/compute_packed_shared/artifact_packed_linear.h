#pragma once

#include "hyle/compute/model_ops.h"

#include <array>

namespace hyle::compute::packed::internal {

// Same-build contraction of the actual public Decode[/Cast]/View/MatMul or
// GroupedMatMul chain. The two types record real rounding boundaries, not a
// precision policy. A grouped bank is the original flat decoder carrier;
// expert_count only gives the exact public reshape, not a fragmented layout.
struct ArtifactPackedLinear final {
  model_ops::WeightArtifactFormat format =
      model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4;
  std::uint32_t group_size = 32;
  DataType decode_type = DataType::kBFloat16;
  DataType operand_type = DataType::kBFloat16;
  std::uint64_t expert_count = 0; // Zero is ordinary MatMul; otherwise grouped.
  static constexpr OperationId operation_id{0x48594c4543554441ULL,
                                             0x43545041434b0001ULL};
  static constexpr std::string_view operation_name =
      "private-artifact-packed-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    const auto count = std::uint16_t((
        format == model_ops::WeightArtifactFormat::kGptqV1Int4 ? 5 :
        format == model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4
            ? 4 : 3) + (expert_count != 0));
    return {{count, count}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> in,
                                   std::span<TensorDesc> out) const noexcept {
    const auto floating = [](DataType type) {
      return type == DataType::kFloat16 || type == DataType::kBFloat16 ||
             type == DataType::kFloat32;
    };
    if ((format != model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4 &&
         format != model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4 &&
         format != model_ops::WeightArtifactFormat::kGptqV1Int4) ||
        !floating(decode_type) || !floating(operand_type) ||
        in.size() != operation_traits().inputs.minimum || out.size() != 1)
      return Status::InvalidArgument("private packed artifact linear ports/types differ");
    if (expert_count && format == model_ops::WeightArtifactFormat::kGptqV1Int4)
      return Status::Unimplemented("private grouped artifact requires the CT bank layout");
    // An empty MatMul still follows a checked public GPTQ Decode. Keep that
    // decomposition instead of dropping its g_idx failure behind an empty dot.
    if (format == model_ops::WeightArtifactFormat::kGptqV1Int4 &&
        in[0].rank() == 2 && in[0].dimensions()[0] == 0)
      return Status::Unimplemented("checked packed artifact linear requires nonempty rows");
    std::array<TensorDesc, 1> decoded;
    HYLE_RETURN_IF_ERROR((model_ops::ArtifactWeightDecode{
        format, group_size, decode_type}.infer_output_descriptions(
            in.subspan(1, in.size() - 1 - (expert_count != 0)), decoded)));
    if (expert_count) {
      const auto rows = static_cast<std::uint64_t>(decoded[0].dimensions()[0]);
      if (expert_count > rows || rows % expert_count != 0)
        return Status::InvalidArgument("private grouped artifact bank reshape differs");
      HYLE_ASSIGN_OR_RETURN(auto weight, TensorDesc::dense(operand_type,
          std::array{static_cast<std::int64_t>(expert_count),
              static_cast<std::int64_t>(rows / expert_count), decoded[0].dimensions()[1]}));
      const std::array operands{weight, in[0], in.back()};
      return model_ops::GroupedMatMul{}.infer_output_descriptions(operands, out);
    }
    HYLE_ASSIGN_OR_RETURN(auto weight, TensorDesc::dense(operand_type,
        std::array{decoded[0].dimensions()[1], decoded[0].dimensions()[0]}));
    const std::array operands{in[0], weight};
    return hyle::compute::infer_output_descriptions(Operation(MatMul{}), operands, out);
  }
};
static_assert(detail::InlineCustomOperationDescriptor<ArtifactPackedLinear>);

} // namespace hyle::compute::packed::internal
