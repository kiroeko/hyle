#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

#include "hyle/compute/operation.h"

namespace hyle::compute::model_ops {

// Native model clamp. Finite values and infinities are clamped to [min,max].
// NaN data and signed-zero bit patterns follow the selected native backend;
// this operation does not promise Binary minimum/maximum's NaN semantics.
struct Clamp final {
  float minimum;
  float maximum;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x434c414d50000001ULL};
  static constexpr std::string_view operation_name = "model-clamp";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{1, 1}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (std::isnan(minimum) || std::isnan(maximum) || minimum > maximum ||
        inputs.size() != 1 || outputs.size() != 1)
      return Status::InvalidArgument("Clamp requires ordered non-NaN bounds and one input/output");
    const auto &input = inputs[0];
    if (!input.valid() || !input.is_plain() || !input.is_dense() ||
        input.data_type() != DataType::kFloat32 || input.rank() < 1 ||
        input.rank() > 4 || !input.element_count() || *input.element_count() == 0)
      return Status::InvalidArgument("Clamp requires nonempty plain dense rank 1-4 F32");
    outputs[0] = input;
    return Status::Ok();
  }
};

inline Operation clamp(float minimum, float maximum) noexcept {
  return Operation::custom(Clamp{minimum, maximum});
}

// Complete DeepSeek score -> biased TopK -> unbiased normalized weights.
// Ties may select any valid top-k member; this is not stable generic TopK.
// The CUDA SGLang carrier consumes finite F32 model logits and bias. Other
// routes may express this model fragment with their mature typed operations.
struct SqrtSoftplusRouting final {
  float scaling = 1.0F;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x445334524f555408ULL};
  static constexpr std::string_view operation_name = "sqrt-softplus-routing";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{2, 2}, 2, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (!std::isfinite(scaling) || scaling <= 0 || inputs.size() != 2 ||
        outputs.size() != 2)
      return Status::InvalidArgument("SqrtSoftplusRouting requires positive finite scale and two inputs/outputs");
    for (const auto &input : inputs)
      if (!input.valid() || !input.is_plain() || !input.is_dense() ||
          input.data_type() != DataType::kFloat32)
        return Status::InvalidArgument("SqrtSoftplusRouting requires plain dense F32");
    if (inputs[0].rank() != 2 || inputs[1].rank() != 1)
      return Status::InvalidArgument("SqrtSoftplusRouting requires logits [T,E] and bias [E]");
    const auto shape = inputs[0].dimensions();
    if (shape[0] <= 0 || (shape[1] != 256 && shape[1] != 384) ||
        inputs[1].dimensions()[0] != shape[1])
      return Status::InvalidArgument("SqrtSoftplusRouting requires nonempty tokens and E256/384");
    const std::array output_shape{shape[0], std::int64_t{6}};
    auto weights = TensorDesc::dense(DataType::kFloat32, output_shape);
    auto ids = TensorDesc::dense(DataType::kInt32, output_shape);
    if (!weights) return std::move(weights).error();
    if (!ids) return std::move(ids).error();
    outputs[0] = *weights;
    outputs[1] = *ids;
    return Status::Ok();
  }
};

inline Operation sqrt_softplus_routing(float scaling = 1.0F) noexcept {
  return Operation::custom(SqrtSoftplusRouting{scaling});
}

// DeepSeek hash layers select six experts from a token-to-expert table.
// Inputs: F32 logits [T,E], I64 token IDs [T], I32 table [V,6]. Outputs:
// F32 weights [T,6] and I32 IDs [T,6], in table order (duplicates retained).
// Weights use sqrt(softplus(logit))/(sum+1e-20); model scaling is separate.
// Invalid token/selected-expert IDs fail Completion without an invalid read.
// Failed outputs cannot be consumed; there is no cross-row rollback promise.
struct SqrtSoftplusHashRouting final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4453344841534801ULL};
  static constexpr std::string_view operation_name = "sqrt-softplus-hash-routing";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{3, 3}, 2, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 3 || outputs.size() != 2)
      return Status::InvalidArgument("Hash routing requires three inputs and two outputs");
    for (const auto &input : inputs)
      if (!input.valid() || !input.is_plain() || !input.is_dense())
        return Status::InvalidArgument("Hash routing requires plain dense inputs");
    if (inputs[0].data_type() != DataType::kFloat32 || inputs[0].rank() != 2 ||
        inputs[1].data_type() != DataType::kInt64 || inputs[1].rank() != 1 ||
        inputs[2].data_type() != DataType::kInt32 || inputs[2].rank() != 2)
      return Status::InvalidArgument("Hash routing requires F32 logits, I64 tokens and an I32 table");
    const auto logits = inputs[0].dimensions();
    const auto table = inputs[2].dimensions();
    if (logits[0] <= 0 || (logits[1] != 256 && logits[1] != 384) ||
        inputs[1].dimensions()[0] != logits[0] || table[0] <= 0 || table[1] != 6)
      return Status::InvalidArgument("Hash routing requires nonempty T/V, E256/384 and six table columns");
    const std::array shape{logits[0], std::int64_t{6}};
    auto weights = TensorDesc::dense(DataType::kFloat32, shape);
    auto ids = TensorDesc::dense(DataType::kInt32, shape);
    if (!weights) return std::move(weights).error();
    if (!ids) return std::move(ids).error();
    outputs[0] = *weights;
    outputs[1] = *ids;
    return Status::Ok();
  }
};

inline Operation sqrt_softplus_hash_routing() noexcept {
  return Operation::custom(SqrtSoftplusHashRouting{});
}

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

// Single-token Qwen recurrent region. Q/K are already normalized and use
// GatedDeltaNet's head mapping. Inputs are Q, K, V, raw a, raw b, negative
// decay coefficient, dt bias, state; outputs are attended values and state.
// The provider owns softplus(a + dt) * negative_a and sigmoid(b) internally.
// Their numerical qualification belongs to the complete recurrent region,
// independently of the generic Unary softplus contract.
struct RecurrentGatedDeltaNet final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x47444e4741540007ULL};
  static constexpr std::string_view operation_name = "recurrent-gated-delta-net";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{8, 8}, 2, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 8 || outputs.size() != 2)
      return Status::InvalidArgument(
          "RecurrentGatedDeltaNet requires eight inputs and two outputs");
    for (const auto &input : inputs)
      if (input.rank() != 4 || !input.is_plain() || !input.is_dense() ||
          input.data_type() != DataType::kFloat32)
        return Status::InvalidArgument(
            "RecurrentGatedDeltaNet requires plain dense rank-four F32 inputs");
    const auto q = inputs[0].dimensions();
    const auto v = inputs[2].dimensions();
    if (q[0] != 1 || q[1] != 1 || q[3] != 128 ||
        (v[2] != 16 && v[2] != 48 && v[2] != 128) ||
        (q[2] != 16 && q[2] != v[2]))
      return Status::InvalidArgument(
          "RecurrentGatedDeltaNet requires B1/T1/D128, V16/48/128 and K16 or equal heads");
    for (std::size_t i = 3; i != 7; ++i) {
      const auto gate = inputs[i].dimensions();
      if (gate[0] != 1 || gate[1] != 1 || gate[2] != v[2] || gate[3] != 1)
        return Status::InvalidArgument(
            "RecurrentGatedDeltaNet gates and parameters require [1,1,H,1]");
    }
    const std::array recurrent{inputs[0], inputs[1], inputs[2], inputs[3],
                               inputs[4], inputs[7]};
    return compute::infer_output_descriptions(Operation(GatedDeltaNet{}),
                                               recurrent, outputs);
  }
};

inline Operation recurrent_gated_delta_net() noexcept {
  return Operation::custom(RecurrentGatedDeltaNet{});
}

// Expert-selecting matrix multiplication.  The expert id is chosen for every
// token/slot pair while the public tensors keep their natural model layout:
//   weights [E,N,K], activations [T,S,K], ids [T,S] -> [T,S,N].
// Plain dense expert weights may be F32 or BF16. Activations, accumulation and
// output remain F32; exact quantized storage mappings are separate provider
// capabilities rather than implicit conversions here.
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
    if ((inputs[0].data_type() != DataType::kFloat32 &&
         (inputs[0].data_type() != DataType::kBFloat16 ||
          !inputs[0].is_plain())) ||
        inputs[1].data_type() != DataType::kFloat32 ||
        inputs[2].data_type() != DataType::kInt32)
      return Status::InvalidArgument(
          "GroupedMatMul requires F32 or plain BF16 weights, F32 activations "
          "and I32 ids");
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

// DeepSeek/GLM learned sparse-attention score region.  We expose model-order
// tensors while the provider retains its native score kernel:
//   query   [S,T,H,D]
//   key     [S,K,1,D]
//   weights [S,1,T,H] (already prescaled)
//   mask    [Sm,1,T,K] (F16, Sm broadcasts across S)
//   output  [S,1,T,K].
// The first common four-route domain matches the original Metal kernel;
// route-specific supersets remain provider qualifications.
struct LightningIndexer final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4c494e4458000006ULL};
  static constexpr std::string_view operation_name = "lightning-indexer";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{4, 4}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 4 || outputs.size() != 1)
      return Status::InvalidArgument("LightningIndexer requires four inputs");
    for (const auto &input : inputs)
      if (input.rank() != 4 || !input.is_dense())
        return Status::InvalidArgument(
            "LightningIndexer requires dense rank-four inputs");
    if (inputs[0].data_type() != DataType::kFloat32 ||
        inputs[1].data_type() != DataType::kFloat32 ||
        inputs[2].data_type() != DataType::kFloat32 ||
        inputs[3].data_type() != DataType::kFloat16)
      return Status::InvalidArgument(
          "LightningIndexer requires F32 query/key/weights and F16 mask");
    const auto q = inputs[0].dimensions();
    const auto k = inputs[1].dimensions();
    const auto weights = inputs[2].dimensions();
    const auto mask = inputs[3].dimensions();
    if (q[0] <= 0 || q[1] <= 0 || q[2] != 64 || q[3] != 128 ||
        k[0] != q[0] || k[1] <= 0 || k[2] != 1 || k[3] != q[3] ||
        weights[0] != q[0] || weights[1] != 1 || weights[2] != q[1] ||
        weights[3] != q[2] || mask[0] <= 0 || q[0] % mask[0] != 0 ||
        mask[1] != 1 || mask[2] != q[1] || mask[3] != k[1])
      return Status::InvalidArgument(
          "LightningIndexer dimensions do not match D128/H64 sparse scoring");
    const std::array shape{q[0], std::int64_t{1}, q[1], k[1]};
    auto output = TensorDesc::dense(DataType::kFloat32, shape);
    if (!output)
      return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

inline Operation grouped_matmul() noexcept {
  return Operation::custom(GroupedMatMul{});
}
// Model linear projection: activations [T,K], stored output-major weights
// [E,K] -> F32 scores [T,E]. The ordinary MatMul contract retains its
// same-dtype output; this operation binds a mature native BF16 product.
struct BFloatLinearF32 final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4d41544d554c4633ULL};
  static constexpr std::string_view operation_name = "bfloat-linear-f32";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{2, 2}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 2 || outputs.size() != 1 || inputs[0].rank() != 2 ||
        inputs[1].rank() != 2 || !inputs[0].is_plain() ||
        !inputs[1].is_plain() || !inputs[0].is_dense() ||
        !inputs[1].is_dense() ||
        inputs[0].data_type() != DataType::kBFloat16 ||
        inputs[1].data_type() != DataType::kBFloat16)
      return Status::InvalidArgument(
          "BFloatLinearF32 requires plain dense BF16 [T,K] and [E,K]");
    const auto lhs = inputs[0].dimensions();
    const auto rhs = inputs[1].dimensions();
    if (lhs[0] <= 0 || lhs[1] <= 0 || rhs[0] <= 0 || rhs[1] != lhs[1])
      return Status::InvalidArgument("BFloatLinearF32 dimensions do not match");
    const std::array shape{lhs[0], rhs[0]};
    auto result = TensorDesc::dense(DataType::kFloat32, shape);
    if (!result)
      return std::move(result).error();
    outputs[0] = *result;
    return Status::Ok();
  }
};

inline Operation bfloat_linear_f32() noexcept {
  return Operation::custom(BFloatLinearF32{});
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
inline Operation lightning_indexer() noexcept {
  return Operation::custom(LightningIndexer{});
}

} // namespace hyle::compute::model_ops
