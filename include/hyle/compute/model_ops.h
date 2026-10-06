#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

#include "hyle/compute/operation.h"
#include "hyle/compute/low_precision.h"

namespace hyle::compute::model_ops {

enum class WeightArtifactFormat : std::uint32_t {
  kNf4,
  kNf4DoubleQuant,
  kAwqGemm4,
  kCompressedTensorsSymmetric4,
};

// Explicit cold preparation of dense original model weights. Run once and
// retain the resulting [N,K] tensor as a constant for mature MatMul/Gather.
// This operation neither loads a model nor caches mutable input payloads.
// NF4 inputs: U8[N,K/2], F32[G] absmax, F32[16] codebook, G=N*K/group.
// Nested NF4: U8[N,K/2], U8[G], F32[ceil(G/256)], F32[256], F32 scalar
// offset, F32[16]. AWQ: I32[K,N/8], I32[K/group,N/8], float[K/group,N].
// CT symmetric: I32[N,K/8], float[N,K/group], original low-first q minus8.
// Packing/group/precision come from the artifact, never its filename.
struct ArtifactWeightDecode final {
  WeightArtifactFormat format = WeightArtifactFormat::kNf4;
  std::uint32_t group_size = 64;
  // Unknown chooses the original scale dtype for AWQ/CT and BF16 for the
  // admitted NF4 artifact. F16/BF16/F32 overrides are explicit numeric choices.
  DataType output_type = DataType::kUnknown;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                            0x57474445434f0001ULL};
  static constexpr std::string_view operation_name = "artifact-weight-decode";
  constexpr OperationTraits operation_traits() const noexcept {
    const std::uint16_t count = format == WeightArtifactFormat::kNf4DoubleQuant ? 6 :
        (format == WeightArtifactFormat::kCompressedTensorsSymmetric4 ? 2 : 3);
    return {{count, count}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept;
};

inline Operation artifact_weight_decode(WeightArtifactFormat format,
    std::uint32_t group_size, DataType output_type = DataType::kUnknown) noexcept {
  return Operation::custom(ArtifactWeightDecode{format, group_size, output_type});
}

enum class Float8ScaleRounding : std::uint32_t { kNone, kPowerOfTwo };

enum class BlockQuantizationFormat : std::uint32_t {
  kFloat8E4M3Block64,
  kFloat4E2M1Block32,
  kFloat8E4M3Block128
};

// Quantize and dequantize finite BF16 activations, preserving a model's QAT
// boundary without exposing packed cache storage. Per contiguous last-axis
// group, scale = 2^ceil(log2(max(absmax, floor)/max_code)), in F32.
// E4M3FN uses groups of64 or128, max_code448, floor1e-4; E2M1 uses groups of32,
// max_code6, floor6*2^-126. Codes use saturated nearest-even rounding; decode
// is F32 followed by BF16 storage. Finite inputs may overflow after rounding.
// Input/output are distinct dense BF16 matrices of identical shape. This is
// not the round-even scale policy used by some other MX quantizers.
struct BlockQuantizationRoundtrip final {
  BlockQuantizationFormat format = BlockQuantizationFormat::kFloat8E4M3Block64;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x514e545254500001ULL};
  static constexpr std::string_view operation_name = "block-quantization-roundtrip";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{1, 1}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 1 || outputs.size() != 1 ||
        (format != BlockQuantizationFormat::kFloat8E4M3Block64 &&
         format != BlockQuantizationFormat::kFloat4E2M1Block32 &&
         format != BlockQuantizationFormat::kFloat8E4M3Block128))
      return Status::InvalidArgument("Quantization roundtrip requires one input and a known format");
    const auto &input = inputs[0];
    const int group = format == BlockQuantizationFormat::kFloat4E2M1Block32 ? 32
        : (format == BlockQuantizationFormat::kFloat8E4M3Block64 ? 64 : 128);
    if (!input.valid() || !input.is_plain() || !input.is_dense() || input.rank() != 2 ||
        input.data_type() != DataType::kBFloat16 || input.dimensions()[0] <= 0 ||
        input.dimensions()[1] <= 0 || input.dimensions()[1] % group != 0)
      return Status::InvalidArgument("Quantization roundtrip requires a positive dense BF16 matrix of whole groups");
    outputs[0] = input;
    return Status::Ok();
  }
};

inline Operation block_quantization_roundtrip(BlockQuantizationFormat format) noexcept {
  return Operation::custom(BlockQuantizationRoundtrip{format});
}

// Decode an existing block128 weight for one-time native weight preparation.
// Input is the logical [K,N] encoded weight; output is physical row-major
// [N,K] in BF16/F32. This explicit layout is ready for model-owned splitting
// or packing. Execute once, then bind the resulting storage as a constant;
// descriptor preparation itself never reads payloads or creates a cache.
struct Float8WeightDecode final {
  DataType output_type = DataType::kBFloat16;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4650384445430001ULL};
  static constexpr std::string_view operation_name = "float8-weight-decode";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{1, 1}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 1 || outputs.size() != 1 || inputs[0].rank() != 2 ||
        (output_type != DataType::kBFloat16 && output_type != DataType::kFloat32))
      return Status::InvalidArgument("FP8 weight decode requires one block128 matrix and BF16/F32 output");
    const auto k = inputs[0].dimensions()[0], n = inputs[0].dimensions()[1];
    auto encoded = float8_weight_description(k, n, Float8ScaleMode::kFloat32Block128);
    if (!encoded || inputs[0] != *encoded)
      return Status::InvalidArgument("FP8 weight decode requires the exact E4M3FN/F32 block128 storage");
    auto output = TensorDesc::dense(output_type, std::array{n, k});
    if (!output) return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

inline Operation float8_weight_decode(DataType output_type = DataType::kBFloat16) noexcept {
  return Operation::custom(Float8WeightDecode{output_type});
}

// Complete model linear: dynamically quantize finite [M,K] activations by
// groups of 128 to E4M3FN, then multiply the already encoded block128 weight.
// scale = max(group_absmax, absmax_floor)/448, optionally rounded upward to a
// power of two; codes use saturated nearest-even conversion. Scales remain
// F32 values; power-of-two rounding does not mean a group32 storage format.
// BF16/F32 output preserves activation dtype after F32 accumulation. The
// floor is explicit: ordinary SGLang uses 1e-10, official DeepSeek uses 1e-4.
struct Float8BlockLinear final {
  Float8ScaleRounding scale_rounding = Float8ScaleRounding::kNone;
  float absmax_floor = 1.0e-10F;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4650384c494e0001ULL};
  static constexpr std::string_view operation_name = "float8-block-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{2, 2}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 2 || outputs.size() != 1 ||
        (scale_rounding != Float8ScaleRounding::kNone &&
         scale_rounding != Float8ScaleRounding::kPowerOfTwo) ||
        !std::isfinite(absmax_floor) || absmax_floor <= 0 ||
        !std::isnormal(absmax_floor / 448.0F))
      return Status::InvalidArgument("FP8 linear requires a known rounding mode and positive normal scale floor");
    const auto &a = inputs[0];
    const auto &b = inputs[1];
    if (!a.valid() || !a.is_plain() || !a.is_dense() || a.rank() != 2 ||
        b.rank() != 2 || (a.data_type() != DataType::kFloat32 &&
                         a.data_type() != DataType::kBFloat16))
      return Status::InvalidArgument("FP8 linear requires dense BF16/F32 [M,K] and block128 weight [K,N]");
    const auto m = a.dimensions()[0], k = a.dimensions()[1], n = b.dimensions()[1];
    auto weight = float8_weight_description(k, n, Float8ScaleMode::kFloat32Block128);
    if (m <= 0 || !weight || b != *weight)
      return Status::InvalidArgument("FP8 linear requires positive dimensions and the exact E4M3FN/block128 weight layout");
    auto output = TensorDesc::dense(a.data_type(), std::array{m, n});
    if (!output) return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

inline Operation float8_block_linear(
    Float8ScaleRounding rounding = Float8ScaleRounding::kNone,
    float absmax_floor = 1.0e-10F) noexcept {
  return Operation::custom(Float8BlockLinear{rounding, absmax_floor});
}

// Weights [E,N,K], finite BF16/F32 activations [T,A,K], I32 expert IDs [T,S]
// -> [T,S,N] in activation dtype. A=1 shares each token across selected
// experts; A=S preserves distinct per-expert activations for the down leg.
// Quantization follows Float8BlockLinear with ordinary F32 group128 scales.
// Duplicate IDs are valid. Negative or out-of-range IDs fail Completion;
// outputs of a failed invocation must not be consumed.
struct GroupedFloat8Linear final {
  float absmax_floor = 1.0e-10F;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4650384752500001ULL};
  static constexpr std::string_view operation_name = "grouped-float8-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{3, 3}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 3 || outputs.size() != 1 ||
        !std::isfinite(absmax_floor) || absmax_floor <= 0 ||
        !std::isnormal(absmax_floor / 448.0F))
      return Status::InvalidArgument("grouped FP8 linear requires a positive normal scale floor");
    const auto &w = inputs[0];
    const auto &a = inputs[1];
    const auto &ids = inputs[2];
    if (w.rank() != 3 || !a.valid() || !a.is_plain() || !a.is_dense() || a.rank() != 3 ||
        (a.data_type() != DataType::kFloat32 && a.data_type() != DataType::kBFloat16) ||
        !ids.valid() || !ids.is_plain() || !ids.is_dense() || ids.rank() != 2 ||
        ids.data_type() != DataType::kInt32)
      return Status::InvalidArgument("grouped FP8 linear requires encoded [E,N,K], dense BF16/F32 [T,A,K], I32 IDs [T,S]");
    const auto t = a.dimensions()[0], slots = ids.dimensions()[1];
    auto weight = float8_grouped_weight_description(
        w.dimensions()[0], w.dimensions()[1], w.dimensions()[2]);
    if (!weight || w != *weight || t <= 0 || slots <= 0 || ids.dimensions()[0] != t ||
        (a.dimensions()[1] != 1 && a.dimensions()[1] != slots) ||
        a.dimensions()[2] != w.dimensions()[2])
      return Status::InvalidArgument("grouped FP8 dimensions or encoded layout do not match");
    auto output = TensorDesc::dense(a.data_type(), std::array{t, slots, w.dimensions()[1]});
    if (!output) return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

inline Operation grouped_float8_linear(float absmax_floor = 1.0e-10F) noexcept {
  return Operation::custom(GroupedFloat8Linear{absmax_floor});
}

// Original FP4 experts: encoded E2M1/F32 group32 [E,N,K], finite dense BF16
// activations [T,A,K], I32 IDs [T,S] -> BF16 [T,S,N]. A=1 or S; K%128=0.
// Each activation group128 is quantized to E4M3FN with absmax floor1e-4,
// scale rounded upward to power2 and saturated nearest-even codes; dot and
// scaling accumulate in F32. No second Q8 activation conversion is implied.
// Duplicate IDs are valid; out-of-range IDs fail Completion. Failed outputs
// are not consumable, and no cross-row rollback is promised.
struct GroupedFloat4Linear final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4650344752500001ULL};
  static constexpr std::string_view operation_name = "grouped-float4-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{3, 3}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 3 || outputs.size() != 1)
      return Status::InvalidArgument("grouped FP4 linear requires three inputs and one output");
    const auto &w = inputs[0];
    const auto &a = inputs[1];
    const auto &ids = inputs[2];
    if (w.rank() != 3 || !a.valid() || !a.is_plain() || !a.is_dense() || a.rank() != 3 ||
        a.data_type() != DataType::kBFloat16 || !ids.valid() || !ids.is_plain() ||
        !ids.is_dense() || ids.rank() != 2 || ids.data_type() != DataType::kInt32)
      return Status::InvalidArgument("grouped FP4 linear requires encoded [E,N,K], BF16 [T,A,K], I32 [T,S]");
    const auto t = a.dimensions()[0], slots = ids.dimensions()[1];
    auto weight = float4_grouped_weight_description(
        w.dimensions()[0], w.dimensions()[1], w.dimensions()[2]);
    if (!weight || w != *weight || w.dimensions()[2] % 128 != 0 || t <= 0 ||
        slots <= 0 || ids.dimensions()[0] != t ||
        (a.dimensions()[1] != 1 && a.dimensions()[1] != slots) ||
        a.dimensions()[2] != w.dimensions()[2])
      return Status::InvalidArgument("grouped FP4 dimensions or encoded layout do not match");
    auto output = TensorDesc::dense(DataType::kBFloat16, std::array{t, slots, w.dimensions()[1]});
    if (!output) return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

inline Operation grouped_float4_linear() noexcept {
  return Operation::custom(GroupedFloat4Linear{});
}

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

// DeepSeek's complete masked attention with a learned, unmasked sink per head.
// Q/K/V are BSHD F32. The sink participates only in the softmax denominator.
// The explicit additive mask already describes causality and cache selection.
struct SinkAttention final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x53494e4b41545401ULL};
  static constexpr std::string_view operation_name = "sink-attention";
  float scale = 1.0F;

  constexpr OperationTraits operation_traits() const noexcept {
    return {{5, 5}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 5 || outputs.size() != 1 ||
        !std::isfinite(scale) || scale <= 0)
      return Status::InvalidArgument("SinkAttention requires five inputs, one output and positive finite scale");
    for (const auto &input : inputs)
      if (!input.valid() || !input.is_plain() || !input.is_dense() ||
          input.data_type() != DataType::kFloat32)
        return Status::InvalidArgument("SinkAttention requires plain dense F32 inputs");
    if (inputs[0].rank() != 4 || inputs[1].rank() != 4 || inputs[2] != inputs[1] ||
        inputs[3].rank() != 2 || inputs[4].rank() != 1)
      return Status::InvalidArgument("SinkAttention requires Q/K/V BSHD, mask [T,S] and sinks [H]");
    const auto q = inputs[0].dimensions();
    const auto k = inputs[1].dimensions();
    const auto mask = inputs[3].dimensions();
    if (q[0] != 1 || k[0] != 1 || q[1] <= 0 || k[1] <= 0 ||
        (q[2] != 64 && q[2] != 128) || k[2] != 1 || q[3] != 512 || k[3] != 512 ||
        mask[0] != q[1] || mask[1] != k[1] || inputs[4].dimensions()[0] != q[2])
      return Status::InvalidArgument("SinkAttention requires B1, H64/128, KV1, D512 and matching mask/sinks");
    const std::array score_shape{q[0], q[2], q[1], k[1]};
    auto scores = TensorDesc::dense(DataType::kFloat32, score_shape);
    if (!scores) return std::move(scores).error();
    auto score_bytes = scores->required_bytes();
    if (!score_bytes) return std::move(score_bytes).error();
    outputs[0] = inputs[0];
    return Status::Ok();
  }
};

inline Operation sink_attention(float scale) noexcept {
  return Operation::custom(SinkAttention{scale});
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
