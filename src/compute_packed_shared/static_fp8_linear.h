#pragma once

#include "hyle/compute/model_ops.h"

#include <array>

namespace hyle::compute::packed::internal {

// Same-build contraction of a public, stored-scale software graph:
// F32(A) * F32(1/SA) -> SATFINITE E4M3 RNE -> F32 code dot W codes,
// then separate ordered F32 scale products and final Cast. Never operand BF16 QDQ.
// Scale bytes are current inputs, including IEEE exceptional values. SA is
// scalar or [M,1]; SW is scalar, [N] or [1,N]. These are explicit outer axes,
// never a flattened scale-count heuristic.
enum class StaticFloat8ScaleOrder : std::uint32_t {
  kActivationThenWeight,
  kWeightThenActivation,
};
struct StaticFloat8Linear final {
  DataType output_type = DataType::kBFloat16;
  StaticFloat8ScaleOrder scale_order = StaticFloat8ScaleOrder::kActivationThenWeight;
  static constexpr OperationId operation_id{0x48594c4553465038ULL, 2};
  static constexpr std::string_view operation_name = "private-static-float8-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{4, 4}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> in,
                                   std::span<TensorDesc> out) const noexcept {
    const auto floating = [](DataType t) {
      return t == DataType::kFloat32 || t == DataType::kFloat16 || t == DataType::kBFloat16;
    };
    if (in.size() != 4 || out.size() != 1 || !floating(output_type) ||
        (scale_order != StaticFloat8ScaleOrder::kActivationThenWeight &&
         scale_order != StaticFloat8ScaleOrder::kWeightThenActivation) ||
        !floating(in[0].data_type()) || in[1].data_type() != DataType::kFloat8E4M3FN)
      return Status::InvalidArgument("static FP8 linear ports or types differ");
    for (const auto &d : in)
      if (!d.is_plain() || !d.is_dense())
        return Status::Unimplemented("static FP8 linear requires plain dense ports");
    if (in[0].rank() != 2 || in[1].rank() != 2 ||
        in[0].dimensions()[1] != in[1].dimensions()[1])
      return Status::InvalidArgument("static FP8 linear matrix dimensions differ");
    if (in[0].dimensions()[0] <= 0 || in[1].dimensions()[0] <= 0 || in[0].dimensions()[1] <= 0)
      return Status::Unimplemented("static FP8 linear leaves empty matrices to the ordinary graph");
    const auto scalar = [](const TensorDesc &d) {
      return d.rank() == 0 || (d.rank() == 1 && d.dimensions()[0] == 1);
    };
    const auto m = in[0].dimensions()[0], n = in[1].dimensions()[0];
    if (in[2].data_type() != DataType::kFloat32 ||
        !(scalar(in[2]) || (in[2].rank() == 2 &&
            in[2].dimensions()[0] == m && in[2].dimensions()[1] == 1)))
      return Status::InvalidArgument("static FP8 SA requires an F32 scalar or [M,1]");
    if (in[3].data_type() != DataType::kFloat32 ||
        !(scalar(in[3]) || (in[3].rank() == 1 && in[3].dimensions()[0] == n) ||
            (in[3].rank() == 2 && in[3].dimensions()[0] == 1 && in[3].dimensions()[1] == n)))
      return Status::InvalidArgument("static FP8 SW requires an F32 scalar, [N] or [1,N]");
    HYLE_ASSIGN_OR_RETURN(out[0], TensorDesc::dense(output_type,
        std::array{in[0].dimensions()[0], in[1].dimensions()[0]}));
    return Status::Ok();
  }
};
static_assert(sizeof(StaticFloat8Linear) == 8);
static_assert(detail::InlineCustomOperationDescriptor<StaticFloat8Linear>);

} // namespace hyle::compute::packed::internal
