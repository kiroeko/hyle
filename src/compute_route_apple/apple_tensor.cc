#include "apple_tensor.h"
#include "apple_carrier.h"
#include "hyle/compute/detail/apple_plan.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include <mlx/device.h>
#include <mlx/fast.h>
#include <mlx/fast_primitives.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/utils.h>

#include "apple_internal.h"
#include "apple_checks.h"
#include "apple_mature_indexing.h"
#include "apple_mature_artifact.h"
#include "apple_mature_reduction.h"
#include "../compute_provider_ggml_metal/native_program.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/sampling_ops.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
Result<DataType> logical_type(mx::Dtype type) {
  if (type == mx::float32)
    return DataType::kFloat32;
  if (type == mx::float16)
    return DataType::kFloat16;
  if (type == mx::bfloat16)
    return DataType::kBFloat16;
  if (type == mx::int32)
    return DataType::kInt32;
  if (type == mx::int64)
    return DataType::kInt64;
  if (type == mx::bool_)
    return DataType::kBool;
  if (type == mx::uint8)
    return DataType::kUInt8;
  if (type == mx::uint32)
    return DataType::kUInt32;
  return std::unexpected(Status::Unimplemented(
      "Apple tensor lowering admits F32/F16/BF16/I32/I64/Bool storage"));
}
Result<mx::Dtype> native_type(DataType type) {
  switch (type) {
  case DataType::kFloat32:
    return mx::float32;
  case DataType::kFloat16:
    return mx::float16;
  case DataType::kBFloat16:
    return mx::bfloat16;
  case DataType::kInt32:
    return mx::int32;
  case DataType::kInt64:
    return mx::int64;
  case DataType::kBool:
    return mx::bool_;
  case DataType::kUInt8:
    return mx::uint8;
  case DataType::kUInt32:
    return mx::uint32;
  default:
    return std::unexpected(Status::Unimplemented(
        "Apple tensor output requires F32/F16/BF16/I32/I64/Bool storage"));
  }
}
bool floating(mx::Dtype type) {
  return type == mx::float32 || type == mx::float16 || type == mx::bfloat16;
}
Result<TensorDesc> description(const mx::array &input) {
  HYLE_ASSIGN_OR_RETURN(auto type, logical_type(input.dtype()));
  if (input.ndim() > kMaxTensorRank)
    return std::unexpected(
        Status::Unimplemented("Apple tensor rank exceeds the schema limit"));
  std::array<std::int64_t, kMaxTensorRank> shape{};
  std::copy(input.shape().begin(), input.shape().end(), shape.begin());
  // These are logical native arrays, not imported Mem storage. MLX owns
  // strides and any layout materialization inside its compiled region.
  return TensorDesc::dense(type, {shape.data(), input.ndim()});
}
Result<mx::Shape> native_shape(const TensorDesc &description) {
  mx::Shape shape;
  shape.reserve(description.rank());
  for (auto dimension : description.dimensions()) {
    if (dimension > (std::numeric_limits<int>::max)())
      return std::unexpected(
          Status::OutOfRange("Apple dimension exceeds the native int range"));
    shape.push_back(static_cast<int>(dimension));
  }
  return shape;
}
mx::array scalar(ScalarValue value, mx::Stream stream) {
  const auto bits = value.bits();
  switch (value.data_type()) {
  case DataType::kFloat32:
    return mx::array(std::bit_cast<float>(static_cast<std::uint32_t>(bits)));
  case DataType::kFloat16:
  case DataType::kBFloat16:
    return mx::view(mx::array(static_cast<std::uint16_t>(bits)),
                    value.data_type() == DataType::kFloat16 ? mx::float16
                                                            : mx::bfloat16,
                    stream);
  case DataType::kInt32:
    return mx::array(
        std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(bits)));
  case DataType::kInt64:
    return mx::array(std::bit_cast<std::int64_t>(bits));
  case DataType::kBool:
    return mx::array(bits != 0);
  default:
    throw std::invalid_argument("Apple scalar type is not admitted");
  }
}
mx::array unary(UnaryKind kind, const mx::array &x, mx::Stream stream) {
  const mx::array zero(0.0F), one(1.0F);
  switch (kind) {
  case UnaryKind::kAbs:
    return mx::abs(x, stream);
  case UnaryKind::kNeg:
    return mx::negative(x, stream);
  case UnaryKind::kExp:
    return mx::exp(x, stream);
  case UnaryKind::kLog:
    return mx::log(x, stream);
  case UnaryKind::kSqrt:
    return mx::sqrt(x, stream);
  case UnaryKind::kRsqrt:
    return mx::rsqrt(x, stream);
  case UnaryKind::kTanh:
    return mx::tanh(x, stream);
  case UnaryKind::kSigmoid:
    return mx::sigmoid(x, stream);
  case UnaryKind::kRelu:
    return mx::where(mx::isnan(x, stream), x,
                     mx::where(mx::greater(x, zero, stream), x, zero, stream),
                     stream);
  case UnaryKind::kSoftplus:
    return mx::logaddexp(x, zero, stream);
  case UnaryKind::kSilu:
  case UnaryKind::kGelu:
  case UnaryKind::kGeluTanh: {
    auto result = x;
    if (kind == UnaryKind::kSilu) {
      result = mx::multiply(x, mx::sigmoid(x, stream), stream);
    } else {
      auto inner = mx::multiply(x, mx::array(0.70710678118654752440F), stream);
      if (kind == UnaryKind::kGelu) {
        inner = mx::erf(inner, stream);
      } else {
        inner = mx::multiply(
            mx::multiply(x, mx::array(0.79788456080286535588F), stream),
            mx::add(one,
                    mx::multiply(mx::array(0.044715F), mx::square(x, stream),
                                 stream),
                    stream),
            stream);
        inner = mx::tanh(inner, stream);
      }
      result = mx::multiply(mx::multiply(mx::array(0.5F), x, stream),
                            mx::add(one, inner, stream), stream);
    }
    // The limiting activation value is +0, not (-Inf)*0 -> NaN.
    return mx::where(
        mx::equal(x, mx::array(-std::numeric_limits<float>::infinity()),
                  stream),
        zero, result, stream);
  }
  }
  throw std::invalid_argument("Apple unary mode is invalid");
}
mx::array binary(BinaryKind kind, const mx::array &left, const mx::array &right,
                 mx::Stream stream) {
  switch (kind) {
  case BinaryKind::kAdd:
    return mx::add(left, right, stream);
  case BinaryKind::kSubtract:
    return mx::subtract(left, right, stream);
  case BinaryKind::kMultiply:
    return mx::multiply(left, right, stream);
  case BinaryKind::kDivide:
    return mx::divide(left, right, stream);
  case BinaryKind::kMinimum:
  case BinaryKind::kMaximum: {
    auto value = kind == BinaryKind::kMinimum
                     ? mx::minimum(left, right, stream)
                     : mx::maximum(left, right, stream);
    // HYLE's elementwise min/max have fmin/fmax semantics: one NaN leaves
    // the other operand. Reductions have a separate NaN contract.
    return mx::where(mx::isnan(left, stream), right,
                     mx::where(mx::isnan(right, stream), left, value, stream),
                     stream);
  }
  }
  throw std::invalid_argument("Apple binary mode is invalid");
}
bool admitted(const Operation &operation) noexcept {
  if (is_mlx_tensor_custom(operation)) return true;
  switch (operation.kind()) {
  case OperationKind::kView:
  case OperationKind::kCopy:
  case OperationKind::kFill:
  case OperationKind::kCast:
  case OperationKind::kUnary:
  case OperationKind::kBinary:
  case OperationKind::kSelect:
  case OperationKind::kConcat:
  case OperationKind::kGatedActivation:
  case OperationKind::kRmsNorm:
  case OperationKind::kLayerNorm:
  case OperationKind::kReduce:
  case OperationKind::kSoftmax:
  case OperationKind::kTopK:
  case OperationKind::kGather:
  case OperationKind::kScatter:
  case OperationKind::kScan:
  case OperationKind::kMatMul:
  case OperationKind::kRoPE:
  case OperationKind::kHadamard:
  case OperationKind::kScaledDotProductAttention:
    return true;
  default:
    return false;
  }
}

mx::array lower_rope(const RoPE &rope, std::span<const mx::array> inputs,
                     mx::Stream stream) {
  const auto &data = inputs[0];
  const auto rank = data.ndim();
  const auto sequence_axis = rope.sequence_axis < 0
                                 ? rope.sequence_axis + static_cast<int>(rank)
                                 : rope.sequence_axis;
  const auto pairs = static_cast<int>(rope.rotary_dimensions / 2);

  mx::Shape position_shape(rank, 1);
  if (inputs[1].ndim() == 1) {
    position_shape[sequence_axis] = data.shape(sequence_axis);
  } else {
    for (std::size_t axis = 0; axis < inputs[1].ndim(); ++axis)
      position_shape[axis] = inputs[1].shape(static_cast<int>(axis));
  }
  auto positions = mx::reshape(mx::astype(inputs[1], mx::float32, stream),
                               position_shape, stream);

  mx::Shape frequency_shape(rank, 1);
  frequency_shape.back() = pairs;
  if (inputs[2].ndim() == 2)
    frequency_shape[sequence_axis] = data.shape(sequence_axis);
  auto frequencies = mx::reshape(mx::astype(inputs[2], mx::float32, stream),
                                 frequency_shape, stream);
  auto angles = mx::multiply(positions, frequencies, stream);
  auto cosines = mx::astype(mx::cos(angles, stream), data.dtype(), stream);
  auto sines = mx::astype(mx::sin(angles, stream), data.dtype(), stream);

  mx::Shape start(rank, 0), stop = data.shape(), strides(rank, 1);
  mx::array first = data;
  mx::array second = data;
  if (rope.layout == RopeLayout::kInterleaved) {
    stop.back() = static_cast<int>(rope.rotary_dimensions);
    strides.back() = 2;
    first = mx::slice(data, start, stop, strides, stream);
    start.back() = 1;
    second = mx::slice(data, start, stop, strides, stream);
  } else {
    stop.back() = pairs;
    first = mx::slice(data, start, stop, stream);
    start.back() = pairs;
    stop.back() = static_cast<int>(rope.rotary_dimensions);
    second = mx::slice(data, start, stop, stream);
  }
  auto rotated_first =
      mx::subtract(mx::multiply(first, cosines, stream),
                   mx::multiply(second, sines, stream), stream);
  auto rotated_second = mx::add(mx::multiply(first, sines, stream),
                                mx::multiply(second, cosines, stream), stream);
  mx::array rotated = rotated_first;
  if (rope.layout == RopeLayout::kInterleaved) {
    auto interleaved =
        mx::concatenate({mx::expand_dims(rotated_first, -1, stream),
                         mx::expand_dims(rotated_second, -1, stream)},
                        -1, stream);
    auto rotary_shape = data.shape();
    rotary_shape.back() = static_cast<int>(rope.rotary_dimensions);
    rotated = mx::reshape(interleaved, rotary_shape, stream);
  } else {
    rotated = mx::concatenate({rotated_first, rotated_second}, -1, stream);
  }
  if (rope.rotary_dimensions < static_cast<std::size_t>(data.shape(-1))) {
    start = mx::Shape(rank, 0);
    start.back() = static_cast<int>(rope.rotary_dimensions);
    rotated = mx::concatenate(
        {rotated, mx::slice(data, start, data.shape(), stream)}, -1, stream);
  }
  return rotated;
}

constexpr int kAttentionQueryBlock = 16;

bool known_gpu_without_nax() noexcept {
  try {
    const auto &info = mx::device_info(mx::Device::gpu);
    const auto found = info.find("architecture");
    if (found == info.end())
      return false;
    const auto *arch = std::get_if<std::string>(&found->second);
    // The pinned native NAX path requires generation >=17 (>=18 on phones).
    // Recognize only known Apple generations 13-16. Unknown/new identifiers
    // keep the native selector, including its OS/build/environment decisions.
    return arch && arch->size() == 13 && arch->starts_with("applegpu_g1") &&
        (*arch)[11] >= '3' && (*arch)[11] <= '6' &&
        (arch->back() == 'g' || arch->back() == 's' ||
         arch->back() == 'd' || arch->back() == 'p');
  } catch (...) {
    return false;
  }
}

bool chunk_attention_math(const mx::array &query, const mx::array &key,
                          const mx::array &value, AttentionPath path) {
  const auto rows = std::int64_t{query.shape(2)};
  const auto keys = std::int64_t{key.shape(2)};
  const auto depth = query.shape(3);
  const auto value_depth = std::int64_t{value.shape(3)};
  if (path != AttentionPath::kMath || query.shape(0) == 0 || depth == 0 || value_depth == 0 ||
      rows < 1024 || keys == 0 ||
      (rows - kAttentionQueryBlock) * keys <= 2 * rows * value_depth)
    return false;

  return true;
}

mx::array attention_math_block(const ScaledDotProductAttention &attention,
                               const mx::array &query,
                               const mx::array &transposed_key,
                               const mx::array &value,
                               const std::optional<mx::array> &mask,
                               int kv_heads, int repeats, mx::Stream stream) {
  auto q = mx::multiply(mx::array(attention.scale, query.dtype()), query, stream);
  if (repeats > 1)
    q = mx::unflatten(q, 1, {kv_heads, repeats}, stream);
  auto scores = mx::matmul(q, transposed_key, stream);
  if (mask) {
    auto block_mask = *mask;
    if (repeats > 1 && block_mask.ndim() >= 3) {
      block_mask = block_mask.shape(-3) == 1
          ? mx::expand_dims(block_mask, -3, stream)
          : mx::unflatten(block_mask, -3, {kv_heads, repeats}, stream);
    }
    if (block_mask.dtype() == mx::bool_) {
      // Preserve the original fallback's uniform completely masked row.
      scores = mx::where(block_mask, scores,
          mx::array(mx::finfo(scores.dtype()).min, scores.dtype()), stream);
    } else {
      scores = mx::add(scores,
                       mx::astype(block_mask, query.dtype(), stream), stream);
    }
  }
  scores = mx::softmax(scores, std::vector<int>{-1}, true, stream);
  auto result = mx::matmul(scores, value, stream);
  return repeats > 1 ? mx::flatten(result, 1, 2, stream) : result;
}

mx::array attention_math_blocks(const ScaledDotProductAttention &attention,
                                const mx::array &query, const mx::array &key,
                                const mx::array &value,
                                const std::optional<mx::array> &mask,
                                mx::Stream stream) {
  const int rows = query.shape(2), keys = key.shape(2);
  const int kv_heads = key.shape(1);
  const int repeats = query.shape(1) / kv_heads;
  const bool causal = !mask && attention.causal;
  auto k = repeats > 1 ? mx::expand_dims(key, 2, stream) : key;
  auto v = repeats > 1 ? mx::expand_dims(value, 2, stream) : value;
  auto transposed_key = mx::swapaxes(k, -1, -2, stream);
  std::optional<mx::array> broadcast_mask;
  if (mask)
    broadcast_mask = mx::broadcast_to(*mask,
        {query.shape(0), query.shape(1), rows, keys}, stream);
  const auto key_indices = mx::arange(0, keys, mx::int32, stream);
  std::vector<mx::array> blocks;
  blocks.reserve(static_cast<std::size_t>((rows - 1) / kAttentionQueryBlock + 1));
  for (int begin = 0; begin < rows;) {
    const int end = begin + std::min(kAttentionQueryBlock, rows - begin);
    auto start = mx::Shape(4, 0), stop = query.shape();
    start[2] = begin;
    stop[2] = end;
    std::vector<mx::array> sources{mx::slice(query, start, stop, stream)};
    if (broadcast_mask) {
      stop[3] = keys;
      sources.push_back(mx::slice(*broadcast_mask, start, stop, stream));
    } else if (causal) {
      sources.push_back(mx::arange(keys - rows + begin, keys - rows + end,
                                   mx::int32, stream));
    }
    // Also sequence mask generation, not only QK: MLX otherwise schedules
    // independent block masks ahead of their consumers. This remains a pure
    // lazy graph; Metal owns command-buffer retirement and buffer lifetimes.
    if (!blocks.empty())
      sources = mx::depends(sources, {blocks.back()});
    std::optional<mx::array> block_mask;
    if (mask || causal)
      block_mask = causal
          ? mx::greater_equal(mx::expand_dims(sources[1], 1, stream),
                              mx::expand_dims(key_indices, 0, stream), stream)
          : sources[1];
    blocks.push_back(attention_math_block(attention, sources[0], transposed_key,
                                          v, block_mask, kv_heads, repeats, stream));
    begin = end;
  }
  return mx::concatenate(std::move(blocks), 2, stream);
}

mx::array lower_attention(const ScaledDotProductAttention &attention,
                          std::span<const mx::array> inputs,
                          mx::Stream stream) {
  auto query = inputs[0];
  auto key = inputs[1];
  auto value = inputs[2];
  const bool sequence_first =
      attention.layout == AttentionLayout::kBatchSequenceHeadDepth;
  if (sequence_first) {
    query = mx::transpose(query, {0, 2, 1, 3}, stream);
    key = mx::transpose(key, {0, 2, 1, 3}, stream);
    value = mx::transpose(value, {0, 2, 1, 3}, stream);
  }
  std::optional<mx::array> mask;
  std::string mask_mode;
  if (attention.mask != MaskKind::kNone) {
    mask = inputs[3];
    mask_mode = "array";
  } else if (attention.causal) {
    mask_mode = "causal";
  }
  const auto path = attention_path(attention, query, key, value, stream);
  // Keep the native mask dtype admission before choosing a different graph.
  const bool compatible_mask = !mask ||
      mx::promote_types(mask->dtype(), query.dtype()) == query.dtype();
  auto result = stream.device == mx::Device::gpu && compatible_mask &&
          chunk_attention_math(query, key, value, path)
      ? attention_math_blocks(attention, query, key, value, mask, stream)
      : mx::fast::scaled_dot_product_attention(
            query, key, value, attention.scale, mask_mode, mask, std::nullopt,
            path == AttentionPath::kForcedFused, stream);
  return sequence_first ? mx::transpose(result, {0, 2, 1, 3}, stream) : result;
}
} // namespace

AttentionPath attention_path(const ScaledDotProductAttention &attention,
    const mx::array &query, const mx::array &key, const mx::array &value,
    mx::Stream stream) {
  if (stream.device != mx::Device::gpu)
    return AttentionPath::kNative;
  // Preserve the existing narrow/full-float force domain, including the F32
  // log2 scale guard. Explicit masks retain their original fallback semantics.
  const bool force_fused =
      ((query.dtype() == mx::float16 || query.dtype() == mx::bfloat16) ||
       (query.dtype() == mx::float32 &&
        std::abs(static_cast<double>(attention.scale)) *
                static_cast<double>(1.4426950408889634F) <=
            static_cast<double>(std::numeric_limits<float>::max()) &&
        (query.shape(3) == 192 ||
         (query.shape(3) == 256 && attention.causal)) &&
        known_gpu_without_nax())) &&
      attention.mask == MaskKind::kNone &&
      query.shape(0) > 0 && query.shape(2) > 8 && key.shape(2) > 0 &&
      query.shape(3) == value.shape(3) &&
      (query.shape(3) == 192 || query.shape(3) == 256) &&
      (!attention.causal || query.shape(2) <= key.shape(2));
  if (force_fused)
    return AttentionPath::kForcedFused;
  const bool array_mask = attention.mask != MaskKind::kNone;
  // The installed native selector reads only shapes/dtype, s.device and GPU
  // capability/environment. It neither evaluates nor uses an owner encoder.
  return mx::fast::ScaledDotProductAttention::use_fallback(query, key, value,
      array_mask || attention.causal, array_mask, !array_mask && attention.causal,
      false, false, false, stream) ? AttentionPath::kMath : AttentionPath::kNative;
}

bool is_mlx_tensor_custom(const Operation &operation) noexcept {
  return operation.custom_as<model_ops::GatherElements>().has_value() ||
         operation.custom_as<model_ops::CausalConv1d>().has_value() ||
         operation.custom_as<model_ops::SinkAttention>().has_value() ||
         operation.custom_as<model_ops::ArtifactWeightDecode>().has_value() ||
         operation.custom_as<model_ops::MlxQuantized>().has_value() ||
         operation.custom_as<sampling_ops::RandomBits>().has_value() ||
         operation.custom_as<sampling_ops::Categorical>().has_value();
}

Result<bool> may_lower_operation(const Operation &operation,
                                 std::span<const TensorDesc> inputs) noexcept {
  // Test the original MLX branches before another carrier's restrictions.
  // In particular, typed MLX and mature float operations need not satisfy
  // ggml Metal's narrower shape/encoding limits.
  if (is_mlx_exact_gguf_gather(operation, inputs) ||
      is_mlx_float_operation(operation, inputs) || admitted(operation) ||
      operation.get_if<QuantizedMatMul>())
    return true;
  auto native = ggml_metal::NativeProgram::validate(operation, inputs);
  if (native.ok()) return true;
  if (native.IsUnimplemented() && !native.has_native_error()) return false;
  return std::unexpected(std::move(native));
}

// Cold-validated ordinary blocks, including a one-row tail, stay on math SDPA.
Result<std::vector<mx::array>> lower_attention_math_block(
    const Operation &operation, std::span<const mx::array> inputs,
    mx::Stream stream) noexcept {
  try {
    const auto *attention = operation.get_if<ScaledDotProductAttention>();
    if (!attention ||
        inputs.size() != (attention->mask != MaskKind::kNone || attention->causal ? 4U : 3U) ||
        (attention->causal && attention->mask != MaskKind::kNone) ||
        attention->accumulation_type != DataType::kFloat32)
      return std::unexpected(Status::InvalidArgument(
          "invalid Apple ordinary attention math block"));
    auto query = inputs[0], key = inputs[1], value = inputs[2];
    const bool sequence_first =
        attention->layout == AttentionLayout::kBatchSequenceHeadDepth;
    if (sequence_first) {
      query = mx::transpose(query, {0, 2, 1, 3}, stream);
      key = mx::transpose(key, {0, 2, 1, 3}, stream);
      value = mx::transpose(value, {0, 2, 1, 3}, stream);
    }
    const int kv_heads = key.shape(1), repeats = query.shape(1) / kv_heads;
    auto k = repeats > 1 ? mx::expand_dims(key, 2, stream) : key;
    auto v = repeats > 1 ? mx::expand_dims(value, 2, stream) : value;
    const auto mask = inputs.size() == 4
        ? std::optional<mx::array>{inputs[3]} : std::nullopt;
    auto result = attention_math_block(*attention, query,
        mx::swapaxes(k, -1, -2, stream), v, mask, kv_heads, repeats, stream);
    return std::vector{sequence_first
        ? mx::transpose(result, {0, 2, 1, 3}, stream) : result};
  } catch (...) {
    return std::unexpected(native_exception());
  }
}

Result<std::vector<mx::array>> lower_tensor(const Operation &operation,
                                            std::span<const mx::array> inputs,
                                            mx::Stream stream,
                                            const PreparedMlxAffineMetadata *prepared) noexcept {
  try {
    const auto causal_conv = operation.custom_as<model_ops::CausalConv1d>();
    const auto mlx_quantized = operation.custom_as<model_ops::MlxQuantized>();
    const auto artifact_decode = operation.custom_as<model_ops::ArtifactWeightDecode>();
    const auto sink_attention = operation.custom_as<model_ops::SinkAttention>();
    if (!admitted(operation))
      return std::unexpected(Status::Unimplemented(
          "Apple typed operation has no admitted MLX lowering"));
    if (!operation_traits(operation).inputs.accepts(inputs.size()))
      return std::unexpected(Status::InvalidArgument(
          "Apple typed operation input count mismatch"));
    if (stream.device.type != mx::Device::gpu)
      return std::unexpected(Status::InvalidArgument(
          "Apple typed lowering requires a GPU stream"));
    std::vector<TensorDesc> descriptions;
    descriptions.reserve(inputs.size());
    for (const auto &input : inputs) {
      HYLE_ASSIGN_OR_RETURN(auto desc, description(input));
      descriptions.push_back(desc);
    }
    std::vector<TensorDesc> outputs(operation_traits(operation).outputs);
    HYLE_RETURN_IF_ERROR(
        infer_output_descriptions(operation, descriptions, outputs));
    if (operation.custom_as<sampling_ops::RandomBits>() ||
        operation.custom_as<sampling_ops::Categorical>())
      return lower_sampling(operation, inputs, stream);
    const auto &output = outputs[0];
    HYLE_ASSIGN_OR_RETURN(auto type, native_type(output.data_type()));
    HYLE_ASSIGN_OR_RETURN(auto shape, native_shape(output));
    if (artifact_decode) {
      if (output.element_count().value() > static_cast<std::uint64_t>(INT32_MAX))
        return std::unexpected(Status::OutOfRange(
            "Apple artifact preparation exceeds native flat index range"));
      HYLE_ASSIGN_OR_RETURN(auto decoded,
          lower_mature_artifact(*artifact_decode, inputs, shape, type, stream));
      return std::vector{std::move(decoded)};
    }
    if (mlx_quantized) {
      const auto &op = *mlx_quantized;
      const bool affine = op.format == model_ops::MlxWeightFormat::kAffine;
      const bool embedding = op.kind == model_ops::MlxQuantizedKind::kEmbedding;
      const bool grouped = op.kind == model_ops::MlxQuantizedKind::kGrouped;
      const bool transpose_weight =
          op.orientation == model_ops::MlxMatmulOrientation::kTransposeWeight;
      const int bits = static_cast<int>(op.bits);
      const std::string mode = affine ? "affine" : bits == 4 ? "mxfp4" : "mxfp8";
      const std::size_t w = embedding ? 0 : 1;
      const auto weights = inputs[w];
      const auto scales = inputs[w + 1];
      const std::optional<mx::array> bias = affine
          ? std::optional<mx::array>{inputs[w + 2]} : std::nullopt;
      if (prepared) {
        if (!affine || embedding)
          return std::unexpected(Status::InvalidArgument(
              "Apple prepared affine metadata differs from the operation"));
        const auto matches = [](const auto &value, const mx::array &source) {
          return !value || (value->status() == mx::array::available &&
              value->dtype() == mx::float32 && value->shape() == source.shape());
        };
        if (!matches(prepared->scales, scales) || !matches(prepared->biases, *bias))
          return std::unexpected(Status::InvalidArgument(
              "Apple prepared affine metadata is not its ready F32 operand"));
      }
      std::optional<mx::array> status;
      std::optional<mx::array> ids;
      if (embedding || grouped) {
        const auto raw = inputs.back();
        auto valid = mx::logical_and(mx::greater_equal(raw, mx::array(0), stream),
            mx::less(raw, mx::array(weights.shape(0)), stream), stream);
        ids = mx::where(valid, raw, mx::array(0), stream);
        status = mx::where(mx::all(valid, false, stream), mx::array(0),
                           mx::array(kIndexOutOfRange), stream);
      }
      mx::array result = weights;
      if (embedding) {
        auto selected_w = mx::take(weights, *ids, 0, stream);
        auto selected_s = mx::take(scales, *ids, 0, stream);
        auto selected_b = mx::take(*bias, *ids, 0, stream);
        result = mx::dequantize(selected_w, selected_s, selected_b,
             static_cast<int>(op.group_size), bits, mode, std::nullopt, type, stream);
      } else {
        // Affine dots use the F32 decoded-weight equation. A low-type QMM
        // otherwise rounds decoded weights in its native threadgroup tile.
        // Keep packed W and the original final output rounding below.
        const auto compute_type = affine ? mx::float32 : type;
        auto source_activation = inputs[0];
        if (grouped && source_activation.shape(1) != 1) {
          // Native Broadcast metadata may still be lazy. Remove only this
          // exact slot broadcast; equal values never imply shared storage.
          if (source_activation.has_primitive() &&
              dynamic_cast<const mx::Broadcast *>(&source_activation.primitive()) &&
              source_activation.inputs().size() == 1) {
            const auto &origin = source_activation.inputs()[0];
            if (origin.ndim() == 3 && origin.shape(0) == source_activation.shape(0) &&
                origin.shape(1) == 1 && origin.shape(2) == source_activation.shape(2))
              source_activation = origin;
          }
          if (source_activation.shape(1) != 1 && source_activation.strides()[1] == 0) {
            auto stop = source_activation.shape(); stop[1] = 1;
            source_activation = mx::slice(source_activation, {0, 0, 0}, stop, stream);
          }
        }
        // Collapse the shared slot before conversion, so affine promotion
        // converts T*K values rather than materializing a T*S*K broadcast.
        auto activation = mx::astype(source_activation, compute_type, stream);
        auto scale = affine && prepared && prepared->scales
            ? *prepared->scales
            : affine ? mx::astype(scales, compute_type, stream) : scales;
        auto biases = bias
            ? std::optional<mx::array>{prepared && prepared->biases
                  ? *prepared->biases : mx::astype(*bias, compute_type, stream)}
            : std::nullopt;
        if (grouped) {
          const auto count64 = std::int64_t{ids->shape(0)} * ids->shape(1);
          if (count64 > (std::numeric_limits<int>::max)())
            return std::unexpected(Status::OutOfRange("MLX selected expert batch exceeds native range"));
          // Upstream indices_or_default casts I32 without forwarding its
          // stream. Supply U32 on this stream so that internal cast is a no-op.
          if (activation.shape(1) == 1) {
            auto lhs = mx::reshape(mx::arange(0, activation.shape(0), mx::uint32, stream),
                                   {activation.shape(0), 1}, stream);
            auto selected = mx::astype(*ids, mx::uint32, stream);
            result = mx::gather_qmm(activation, weights, scale, biases, lhs,
                selected, transpose_weight, static_cast<int>(op.group_size), bits, mode, false, stream);
          } else {
            const int count = static_cast<int>(count64);
            auto rows = mx::reshape(activation, {count, 1, activation.shape(2)}, stream);
            auto selected = mx::astype(mx::reshape(*ids, {count}, stream), mx::uint32, stream);
            result = mx::gather_qmm(rows, weights, scale, biases, std::nullopt,
                selected, transpose_weight, static_cast<int>(op.group_size), bits, mode, false, stream);
          }
          result = mx::reshape(result, shape, stream);
        } else {
          result = mx::quantized_matmul(activation, weights, scale, biases, transpose_weight,
                       static_cast<int>(op.group_size), bits, mode, stream);
        }
      }
      result = mx::astype(result, type, stream);
      if (status) {
        HYLE_ASSIGN_OR_RETURN(auto checked,
            with_device_status(std::move(result), std::move(*status), stream));
        result = std::move(checked);
      }
      return std::vector{std::move(result)};
    }
    if (sink_attention) {
      auto query = mx::transpose(inputs[0], {0, 2, 1, 3}, stream);
      auto key = mx::transpose(inputs[1], {0, 2, 1, 3}, stream);
      auto value = mx::transpose(inputs[2], {0, 2, 1, 3}, stream);
      auto mask = inputs[3];
      if (mask.ndim() == 3)
        mask = mx::reshape(mask,
            {mask.shape(0), 1, mask.shape(1), mask.shape(2)}, stream);
      auto result = mx::fast::scaled_dot_product_attention(query, key, value,
          sink_attention->scale, "array", mask, inputs[4], false, stream);
      return std::vector{mx::transpose(result, {0, 2, 1, 3}, stream)};
    }
    if (causal_conv) {
      if (!floating(inputs[0].dtype()) ||
          inputs[1].dtype() != inputs[0].dtype())
        return std::unexpected(Status::Unimplemented(
            "Apple MLX CausalConv1d requires matching floating inputs"));
      const auto channels = inputs[0].shape(-1);
      auto weights = mx::expand_dims(inputs[1], -1, stream);
      return std::vector{
          mx::conv1d(inputs[0], weights, 1, 0, 1, channels, stream)};
    }
    if (operation.kind() == OperationKind::kGather ||
        operation.custom_as<model_ops::GatherElements>() ||
        operation.kind() == OperationKind::kScan ||
        operation.kind() == OperationKind::kScatter ||
        operation.kind() == OperationKind::kTopK)
      return lower_mature_indexing(operation, inputs, shape, stream);
    if (const auto *fill = operation.get_if<Fill>())
      return std::vector{
          mx::full(shape, scalar(fill->value, stream), type, stream)};
    if (const auto *view = operation.get_if<View>()) {
      switch (view->kind()) {
      case ViewKind::kReshape:
        return std::vector{mx::reshape(inputs[0], shape, stream)};
      case ViewKind::kBroadcast:
        return std::vector{mx::broadcast_to(inputs[0], shape, stream)};
      case ViewKind::kPermute: {
        std::vector<int> axes;
        for (auto axis : view->permutation())
          axes.push_back(axis < 0 ? axis + static_cast<int>(inputs[0].ndim())
                                  : axis);
        return std::vector{mx::transpose(inputs[0], axes, stream)};
      }
      case ViewKind::kSlice: {
        const auto axis =
            view->axis() < 0 ? view->axis() + static_cast<int>(inputs[0].ndim())
                             : view->axis();
        mx::Shape start(inputs[0].ndim(), 0), stop = inputs[0].shape(),
                                              step(inputs[0].ndim(), 1);
        start[axis] = static_cast<int>(view->start());
        stop[axis] = static_cast<int>(
            view->count() == 0
                ? view->start()
                : view->start() + (view->count() - 1) * view->step() + 1);
        step[axis] = static_cast<int>(std::min<std::int64_t>(
            view->step(), (std::numeric_limits<int>::max)()));
        return std::vector{mx::slice(inputs[0], start, stop, step, stream)};
      }
      }
    }
    if (operation.kind() == OperationKind::kCopy) {
      // MLX Copy is a shared-buffer identity, while public Copy produces a
      // dense snapshot. Use the original native copy primitives once, through
      // same-width integer storage so floating payload/sign bits stay exact.
      const auto width = inputs[0].itemsize();
      const auto bits = width == 1 ? mx::uint8 : width == 2 ? mx::uint16
          : width == 4 ? mx::uint32 : mx::uint64;
      const auto raw = mx::view(inputs[0], bits, stream);
      auto copied = inputs[0].flags().row_contiguous
          ? mx::full(shape, raw, bits, stream)
          : mx::contiguous(raw, false, stream);
      return std::vector{mx::view(copied, type, stream)};
    }
    if (operation.kind() == OperationKind::kSelect) {
      // Bool storage admits any nonzero byte. Normalize through its raw
      // carrier before selection, without touching either selected payload.
      const auto condition = mx::not_equal(mx::view(inputs[0], mx::uint8, stream),
          mx::array(std::uint8_t{0}), stream);
      if (!floating(type))
        return std::vector{mx::where(condition, inputs[1], inputs[2], stream)};
      // Selection is storage-exact. A compiled floating selection can
      // canonicalize BF16 NaNs; same-width native views keep payload/sign bits
      // out of floating constant conversion without introducing arithmetic.
      const auto bits = type == mx::float32 ? mx::uint32 : mx::uint16;
      return std::vector{
          mx::view(mx::where(condition, mx::view(inputs[1], bits, stream),
                             mx::view(inputs[2], bits, stream), stream),
                   type, stream)};
    }
    if (const auto *concat = operation.get_if<Concat>())
      return std::vector{
          mx::concatenate(std::vector<mx::array>(inputs.begin(), inputs.end()),
                          concat->axis, stream)};
    if (operation.kind() == OperationKind::kCast) {
      if (inputs[0].dtype() == mx::int32 && type == mx::float32)
        return std::vector{mx::astype(inputs[0], type, stream)};
      if (floating(inputs[0].dtype()) && type == mx::bool_) {
        // Floating truth is an IEEE storage test: both signed zeros are
        // false; subnormals, infinities and NaNs are true even under FTZ.
        const bool wide = inputs[0].dtype() == mx::float32;
        const auto raw = mx::view(inputs[0], wide ? mx::uint32 : mx::uint16, stream);
        const auto magnitude = mx::bitwise_and(raw, wide
            ? mx::array(std::uint32_t{0x7fffffff})
            : mx::array(std::uint16_t{0x7fff}), stream);
        return std::vector{mx::not_equal(magnitude, wide
            ? mx::array(std::uint32_t{0}) : mx::array(std::uint16_t{0}), stream)};
      }
      if (inputs[0].dtype() == mx::bool_ && floating(type)) {
        const auto truth = mx::not_equal(mx::view(inputs[0], mx::uint8, stream),
            mx::array(std::uint8_t{0}), stream);
        return std::vector{mx::astype(truth, type, stream)};
      }
    }
    if (!floating(inputs[0].dtype()) || !floating(type))
      return std::unexpected(Status::Unimplemented(
          "Apple arithmetic and Cast require F32/F16/BF16"));
    if (operation.kind() == OperationKind::kReduce ||
        operation.kind() == OperationKind::kSoftmax)
      return lower_mature_reduction(operation, inputs, shape, stream);
    if (operation.kind() == OperationKind::kCast)
      return std::vector{mx::astype(inputs[0], type, stream)};
    if (const auto *matmul = operation.get_if<MatMul>()) {
      if (matmul->accumulation_type != DataType::kFloat32 ||
          !floating(inputs[1].dtype()))
        return std::unexpected(Status::Unimplemented(
            "Apple MLX MatMul requires floating inputs and F32 accumulation"));
      return std::vector{mx::matmul(inputs[0], inputs[1], stream)};
    }
    if (const auto *rope = operation.get_if<RoPE>())
      return std::vector{lower_rope(*rope, inputs, stream)};
    if (const auto *hadamard = operation.get_if<Hadamard>()) {
      if (hadamard->accumulation_type != DataType::kFloat32)
        return std::unexpected(Status::Unimplemented(
            "Apple MLX Hadamard requires F32 accumulation"));
      const auto scale =
          1.0F / std::sqrt(static_cast<float>(inputs[0].shape(-1)));
      return std::vector{mx::hadamard_transform(inputs[0], scale, stream)};
    }
    if (const auto *attention = operation.get_if<ScaledDotProductAttention>()) {
      if (attention->accumulation_type != DataType::kFloat32 ||
          (attention->causal && attention->mask != MaskKind::kNone))
        return std::unexpected(Status::Unimplemented(
            "Apple MLX SDPA requires F32 accumulation and one mask mode"));
      return std::vector{lower_attention(*attention, inputs, stream)};
    }
    auto first = mx::astype(inputs[0], mx::float32, stream);
    if (const auto *value = operation.get_if<Unary>())
      return std::vector{
          mx::astype(unary(value->kind, first, stream), type, stream)};
    if (const auto *value = operation.get_if<Binary>()) {
      // The schema and floating domain are already checked. MLX's fused
      // empty binary kernel can form a zero Metal grid during cold tracing.
      if (std::find(shape.begin(), shape.end(), 0) != shape.end())
        return std::vector{mx::zeros(shape, type, stream)};
      return std::vector{
          mx::astype(binary(value->kind, first,
                            mx::astype(inputs[1], mx::float32, stream), stream),
                     type, stream)};
    }
    if (const auto *gated = operation.get_if<GatedActivation>()) {
      auto up = first;
      if (gated->inputs == GatedInputKind::kSeparate) {
        up = mx::astype(inputs[1], mx::float32, stream);
      } else {
        auto halves = mx::split(first, 2, -1, stream);
        first = std::move(halves[0]);
        up = std::move(halves[1]);
      }
      const auto kind = gated->activation == GatedActivationKind::kSilu
                            ? UnaryKind::kSilu
                        : gated->activation == GatedActivationKind::kGelu
                            ? UnaryKind::kGelu
                            : UnaryKind::kGeluTanh;
      return std::vector{mx::astype(
          mx::multiply(unary(kind, first, stream), up, stream), type, stream)};
    }
    const auto *rms = operation.get_if<RmsNorm>();
    const auto *layer = operation.get_if<LayerNorm>();
    if (rms == nullptr && layer == nullptr)
      return std::unexpected(
          Status::Internal("Apple typed lowering not bound"));
    if ((rms ? rms->accumulation_type : layer->accumulation_type) !=
        DataType::kFloat32)
      return std::unexpected(Status::Unimplemented(
          "Apple normalization requires explicit F32 accumulation"));
    HYLE_ASSIGN_OR_RETURN(auto count, output.element_count());
    if (count == 0)
      return std::vector{mx::zeros(shape, type, stream)};
    std::optional<mx::array> weight, bias;
    if (inputs.size() > 1)
      weight = mx::astype(inputs[1], mx::float32, stream);
    if (inputs.size() > 2)
      bias = mx::astype(inputs[2], mx::float32, stream);
    auto result =
        rms ? mx::fast::rms_norm(first, weight, rms->epsilon, stream)
            : mx::fast::layer_norm(first, weight, bias, layer->epsilon, stream);
    return std::vector{mx::astype(result, type, stream)};
  } catch (...) {
    return std::unexpected(native_exception());
  }
}
} // namespace hyle::compute::apple::detail
