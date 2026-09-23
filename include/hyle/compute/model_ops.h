#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

#include "hyle/compute/operation.h"

namespace hyle::compute::model_ops {

// Depthwise causal-window convolution used by recurrent LLM blocks. The
// caller supplies the K-1 history prefix explicitly, so replay is stateless:
//   samples [B, T + K - 1, C], weights [C, K] -> output [B, T, C].
// Providers keep their native layout and may fuse following bias/activation
// when the graph boundary and exact semantics permit it.
struct CausalConv1d final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x434f4e5631440001ULL};
  static constexpr std::string_view operation_name = "causal-conv1d";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{2, 2}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 2 || outputs.size() != 1 || inputs[0].rank() != 3 ||
        inputs[1].rank() != 2 || !inputs[0].is_plain() ||
        !inputs[1].is_plain() || !inputs[0].is_dense() ||
        !inputs[1].is_dense()) {
      return Status::InvalidArgument(
          "CausalConv1d requires dense samples [B,W,C] and weights [C,K]");
    }
    const auto type = inputs[0].data_type();
    if ((type != DataType::kFloat16 && type != DataType::kBFloat16 &&
         type != DataType::kFloat32) ||
        inputs[1].data_type() != type) {
      return Status::InvalidArgument(
          "CausalConv1d requires matching F16/BF16/F32 inputs");
    }
    const auto samples = inputs[0].dimensions();
    const auto weights = inputs[1].dimensions();
    if (samples[0] <= 0 || samples[1] <= 0 || samples[2] <= 0 ||
        weights[0] != samples[2] || weights[1] <= 0 ||
        samples[1] < weights[1]) {
      return Status::InvalidArgument(
          "CausalConv1d channel/kernel dimensions do not match");
    }
    const std::array shape{samples[0], samples[1] - weights[1] + 1,
                           samples[2]};
    auto output = TensorDesc::dense(type, shape);
    if (!output)
      return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

inline Operation causal_conv1d() noexcept {
  return Operation::custom(CausalConv1d{});
}

} // namespace hyle::compute::model_ops
