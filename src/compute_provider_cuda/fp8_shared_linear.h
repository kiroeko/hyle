#pragma once

#include <array>

#include "hyle/compute/model_ops.h"

namespace hyle::compute::cuda_native::internal {

// Same-build composite for two original linears sharing the exact logical A
// and quantization contract. Weights and outputs remain separate real ports.
struct Float8BlockLinearPair final {
  model_ops::Float8ScalePolicy scale_policy =
      model_ops::Float8ScalePolicy::kDivide;
  float absmax_floor = 1.0e-10F;
  static constexpr OperationId operation_id{0x48594c4543554441ULL,
                                             0x4650385041490002ULL};
  static constexpr std::string_view operation_name =
      "private-cuda-float8-block-linear-pair";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{3, 3}, 2, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 3 || outputs.size() != 2)
      return Status::InvalidArgument("FP8 linear pair port counts differ");
    const model_ops::Float8BlockLinear linear{scale_policy, absmax_floor};
    std::array<TensorDesc, 2> inferred;
    for (std::size_t branch = 0; branch < 2; ++branch) {
      const std::array ports{inputs[0], inputs[branch + 1]};
      HYLE_RETURN_IF_ERROR(linear.infer_output_descriptions(
          ports, std::span(inferred).subspan(branch, 1)));
    }
    outputs[0] = inferred[0];
    outputs[1] = inferred[1];
    return Status::Ok();
  }
};
static_assert(sizeof(Float8BlockLinearPair) == 8);
static_assert(detail::InlineCustomOperationDescriptor<Float8BlockLinearPair>);

} // namespace hyle::compute::cuda_native::internal
