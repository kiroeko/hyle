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

// Expert-selecting matrix multiplication.  The expert id is chosen for every
// token/slot pair while the public tensors keep their natural model layout:
//   weights [E,N,K], activations [T,S,K], ids [T,S] -> [T,S,N].
// The first admitted contract is dense F32; exact quantized storage mappings
// are separate provider capabilities rather than implicit conversions here.
struct GroupedMatMul final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4d554c4944000002ULL};
  static constexpr std::string_view operation_name = "grouped-matmul";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{3, 3}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 3 || outputs.size() != 1 || inputs[0].rank() != 3 ||
        inputs[1].rank() != 3 || inputs[2].rank() != 2 ||
        !inputs[0].is_dense() || !inputs[1].is_dense() ||
        !inputs[2].is_dense())
      return Status::InvalidArgument(
          "GroupedMatMul requires dense [E,N,K], [T,S,K], ids [T,S]");
    if (inputs[0].data_type() != DataType::kFloat32 ||
        inputs[1].data_type() != DataType::kFloat32 ||
        inputs[2].data_type() != DataType::kInt32)
      return Status::InvalidArgument(
          "GroupedMatMul requires F32 weights/activations and I32 ids");
    const auto weights = inputs[0].dimensions();
    const auto activations = inputs[1].dimensions();
    const auto ids = inputs[2].dimensions();
    if (weights[0] <= 0 || weights[1] <= 0 || weights[2] <= 0 ||
        activations[0] <= 0 || activations[1] <= 0 ||
        activations[2] != weights[2] || ids[0] != activations[0] ||
        ids[1] != activations[1])
      return Status::InvalidArgument("GroupedMatMul dimensions do not match");
    const std::array shape{activations[0], activations[1], weights[1]};
    auto output = TensorDesc::dense(DataType::kFloat32, shape);
    if (!output)
      return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

// DeepSeek V4 multi-stream hyper-connection primitives.  hc is deliberately
// fixed to four because that is the exact mature-provider/model contract.
struct HyperConnectionCombine final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4843434f4d420003ULL};
  static constexpr std::string_view operation_name = "hyper-connection-combine";
  float epsilon = 1.0e-6F;
  std::int32_t iterations = 20;

  constexpr OperationTraits operation_traits() const noexcept {
    return {{3, 3}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 3 || outputs.size() != 1 || inputs[0].rank() != 2 ||
        inputs[1].rank() != 1 || inputs[2].rank() != 1 ||
        !inputs[0].is_dense() || !inputs[1].is_dense() ||
        !inputs[2].is_dense())
      return Status::InvalidArgument(
          "HyperConnectionCombine requires dense [T,24], [3], [24]");
    if (inputs[0].data_type() != DataType::kFloat32 ||
        inputs[1].data_type() != DataType::kFloat32 ||
        inputs[2].data_type() != DataType::kFloat32 || !(epsilon >= 0.0F) ||
        iterations <= 0)
      return Status::InvalidArgument(
          "HyperConnectionCombine requires F32 inputs and valid cold parameters");
    const auto mixes = inputs[0].dimensions();
    if (mixes[0] <= 0 || mixes[1] != 24 ||
        inputs[1].dimensions()[0] != 3 || inputs[2].dimensions()[0] != 24)
      return Status::InvalidArgument(
          "HyperConnectionCombine dimensions do not match hc=4");
    const std::array shape{mixes[0], std::int64_t{4}, std::int64_t{4}};
    auto output = TensorDesc::dense(DataType::kFloat32, shape);
    if (!output)
      return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

struct HyperConnectionPre final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4843505245000004ULL};
  static constexpr std::string_view operation_name = "hyper-connection-pre";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{2, 2}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 2 || outputs.size() != 1 || inputs[0].rank() != 3 ||
        inputs[1].rank() != 2 || !inputs[0].is_dense() ||
        !inputs[1].is_dense() ||
        inputs[0].data_type() != DataType::kFloat32 ||
        inputs[1].data_type() != DataType::kFloat32)
      return Status::InvalidArgument(
          "HyperConnectionPre requires dense F32 [T,4,D], [T,4]");
    const auto value = inputs[0].dimensions();
    const auto weights = inputs[1].dimensions();
    if (value[0] <= 0 || value[1] != 4 || value[2] <= 0 ||
        weights[0] != value[0] || weights[1] != 4)
      return Status::InvalidArgument(
          "HyperConnectionPre dimensions do not match hc=4");
    const std::array shape{value[0], value[2]};
    auto output = TensorDesc::dense(DataType::kFloat32, shape);
    if (!output)
      return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

struct HyperConnectionPost final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4843504f53540005ULL};
  static constexpr std::string_view operation_name = "hyper-connection-post";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{4, 4}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 4 || outputs.size() != 1 || inputs[0].rank() != 2 ||
        inputs[1].rank() != 3 || inputs[2].rank() != 2 ||
        inputs[3].rank() != 3)
      return Status::InvalidArgument(
          "HyperConnectionPost requires [T,D], [T,4,D], [T,4], [T,4,4]");
    for (const auto &input : inputs)
      if (!input.is_dense() || input.data_type() != DataType::kFloat32)
        return Status::InvalidArgument(
            "HyperConnectionPost requires dense F32 inputs");
    const auto value = inputs[0].dimensions();
    const auto residual = inputs[1].dimensions();
    const auto post = inputs[2].dimensions();
    const auto combine = inputs[3].dimensions();
    if (value[0] <= 0 || value[1] <= 0 || residual[0] != value[0] ||
        residual[1] != 4 || residual[2] != value[1] ||
        post[0] != value[0] || post[1] != 4 ||
        combine[0] != value[0] || combine[1] != 4 || combine[2] != 4)
      return Status::InvalidArgument(
          "HyperConnectionPost dimensions do not match hc=4");
    auto output = TensorDesc::dense(DataType::kFloat32, residual);
    if (!output)
      return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

inline Operation grouped_matmul() noexcept {
  return Operation::custom(GroupedMatMul{});
}
inline Operation hyper_connection_combine(float epsilon = 1.0e-6F,
                                           std::int32_t iterations = 20) noexcept {
  return Operation::custom(HyperConnectionCombine{epsilon, iterations});
}
inline Operation hyper_connection_pre() noexcept {
  return Operation::custom(HyperConnectionPre{});
}
inline Operation hyper_connection_post() noexcept {
  return Operation::custom(HyperConnectionPost{});
}

} // namespace hyle::compute::model_ops
