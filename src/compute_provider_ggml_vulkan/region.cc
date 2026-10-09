#include "region.h"
#include "native_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>

#include "ggml-backend.h"
#include "ggml.h"
#include "explicit_rope.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

namespace hyle::compute::ggml_vulkan::detail {
namespace {
bool floating(DataType type) noexcept {
  return type == DataType::kFloat32 || type == DataType::kFloat16 ||
         type == DataType::kBFloat16;
}
ggml_type native_type(DataType type) noexcept {
  if (type == DataType::kFloat16)
    return GGML_TYPE_F16;
  if (type == DataType::kBFloat16)
    return GGML_TYPE_BF16;
  return type == DataType::kInt32 ? GGML_TYPE_I32 : GGML_TYPE_F32;
}
ggml_type native_quantized_type(const TensorDesc &description) noexcept {
  const auto type = gguf_quantized_type(description.storage_format());
  if (!type)
    return GGML_TYPE_COUNT;
  switch (*type) {
  case GgufQuantizedType::kQ4_0:
    return GGML_TYPE_Q4_0;
  case GgufQuantizedType::kQ4_1:
    return GGML_TYPE_Q4_1;
  case GgufQuantizedType::kQ2K:
    return GGML_TYPE_Q2_K;
  case GgufQuantizedType::kQ3K:
    return GGML_TYPE_Q3_K;
  case GgufQuantizedType::kQ4K:
    return GGML_TYPE_Q4_K;
  case GgufQuantizedType::kQ5K:
    return GGML_TYPE_Q5_K;
  case GgufQuantizedType::kQ6K:
    return GGML_TYPE_Q6_K;
  case GgufQuantizedType::kQ8_0:
    return GGML_TYPE_Q8_0;
  case GgufQuantizedType::kIq3Xxs:
    return GGML_TYPE_IQ3_XXS;
  case GgufQuantizedType::kIq2Xxs:
    return GGML_TYPE_IQ2_XXS;
  case GgufQuantizedType::kIq2Xs:
    return GGML_TYPE_IQ2_XS;
  case GgufQuantizedType::kIq2S:
    return GGML_TYPE_IQ2_S;
  case GgufQuantizedType::kIq1S:
    return GGML_TYPE_IQ1_S;
  case GgufQuantizedType::kIq1M:
    return GGML_TYPE_IQ1_M;
  case GgufQuantizedType::kIq3S:
    return GGML_TYPE_IQ3_S;
  case GgufQuantizedType::kIq4Nl:
    return GGML_TYPE_IQ4_NL;
  case GgufQuantizedType::kIq4Xs:
    return GGML_TYPE_IQ4_XS;
  case GgufQuantizedType::kMxfp4:
    return GGML_TYPE_MXFP4;
  default:
    break; // Extended IQ1 formats still need original Vulkan qualification.
  }
  return GGML_TYPE_COUNT;
}
bool native_quantized(const TensorDesc &description) noexcept {
  return native_quantized_type(description) != GGML_TYPE_COUNT;
}
bool quantized_weight_domain(const TensorDesc &description) noexcept {
  const auto type = gguf_quantized_type(description.storage_format());
  if (!type || !native_quantized(description) || description.rank() != 2 ||
      (description.data_type() != DataType::kFloat32 &&
       description.data_type() != DataType::kFloat16))
    return false;
  const auto canonical = gguf_weight_description(*type, description.data_type(),
                                                 description.dimensions()[0],
                                                 description.dimensions()[1]);
  return canonical && *canonical == description;
}
bool quantized_grouped_weight_domain(const TensorDesc &description) noexcept {
  const auto type = gguf_quantized_type(description.storage_format());
  if (!type || !native_quantized(description) || description.rank() != 3 ||
      description.data_type() != DataType::kFloat32)
    return false;
  const auto shape = description.dimensions();
  const auto canonical = gguf_grouped_weight_description(
      *type, description.data_type(), shape[0], shape[1], shape[2]);
  return canonical && *canonical == description;
}
bool native_embedding_table(const TensorDesc &description) noexcept {
  const auto type = gguf_quantized_type(description.storage_format());
  if (!type ||
      (*type != GgufQuantizedType::kQ2K && *type != GgufQuantizedType::kQ3K &&
       *type != GgufQuantizedType::kQ4_0 && *type != GgufQuantizedType::kQ4_1 &&
       *type != GgufQuantizedType::kQ5K &&
       *type != GgufQuantizedType::kQ4K && *type != GgufQuantizedType::kQ6K &&
       *type != GgufQuantizedType::kQ8_0) ||
      description.data_type() != DataType::kFloat32 ||
      description.rank() != 2 || description.block_axis() != 1)
    return false;
  const auto stored = gguf_weight_description(*type, DataType::kFloat32,
                                              description.dimensions()[1],
                                              description.dimensions()[0]);
  if (!stored)
    return false;
  const auto natural = stored->transposed();
  return natural && *natural == description;
}
bool tensor_domain(const TensorDesc &description) noexcept {
  return description.valid() &&
         (native_embedding_table(description) ||
          quantized_weight_domain(description) ||
          quantized_grouped_weight_domain(description) ||
          (description.is_plain() && description.is_dense() &&
           (floating(description.data_type()) ||
            description.data_type() == DataType::kInt32))) &&
         description.element_count() && *description.element_count() > 0 &&
         *description.element_count() <= INT32_MAX;
}
bool same_shape(const TensorDesc &a, const TensorDesc &b) noexcept {
  return a.rank() == b.rank() &&
         std::ranges::equal(a.dimensions(), b.dimensions());
}

std::int64_t columns(const TensorDesc &description) noexcept {
  return description.rank() == 0 ? 1 : description.dimensions().back();
}
std::array<std::int64_t, GGML_MAX_DIMS>
native_shape(const TensorDesc &description) noexcept {
  std::array<std::int64_t, GGML_MAX_DIMS> shape{1, 1, 1, 1};
  if (description.rank() <= GGML_MAX_DIMS)
    std::ranges::reverse_copy(description.dimensions(), shape.begin());
  else {
    shape[0] = columns(description);
    shape[1] =
        static_cast<std::int64_t>(*description.element_count()) / shape[0];
  }
  return shape;
}
ggml_tensor *native_tensor(ggml_context *context,
                           const TensorDesc &description) noexcept {
  auto shape = native_shape(description);
  if (native_quantized(description)) {
    const auto blocked = description.block_axis();
    if (!blocked || description.rank() > GGML_MAX_DIMS)
      return nullptr;
    shape.fill(1);
    shape[0] = description.dimensions()[*blocked];
    std::size_t target = 1;
    for (std::size_t axis = description.rank(); axis-- > 0;)
      if (axis != *blocked)
        shape[target++] = description.dimensions()[axis];
    return ggml_new_tensor(context, native_quantized_type(description),
                           static_cast<int>(description.rank()), shape.data());
  }
  return ggml_new_tensor(context, native_type(description.data_type()),
                         GGML_MAX_DIMS, shape.data());
}
ggml_tensor *restore_shape(ggml_context *context, ggml_tensor *value,
                           const TensorDesc &description) {
  const auto shape = native_shape(description);
  if (std::ranges::equal(shape, value->ne))
    return value;
  return ggml_reshape_4d(context, value, shape[0], shape[1], shape[2],
                         shape[3]);
}
std::int64_t extent_product(std::span<const std::int64_t> shape) noexcept {
  std::int64_t result = 1;
  for (const auto extent : shape)
    result *= extent;
  return result;
}
ggml_tensor *matrix(ggml_context *context, ggml_tensor *value,
                    const TensorDesc &description) {
  const auto width = columns(description);
  const auto rows =
      static_cast<std::int64_t>(*description.element_count()) / width;
  if (value->ne[0] == width && value->ne[1] == rows && value->ne[2] == 1 &&
      value->ne[3] == 1)
    return value;
  return ggml_reshape_2d(context, value, width, rows);
}
ggml_tensor *indexed_rows(ggml_context *context, ggml_tensor *value,
                          std::int64_t inner, std::int64_t rows,
                          std::int64_t outer) {
  value = ggml_reshape_3d(context, value, inner, rows, outer);
  if (outer != 1) {
    value = ggml_permute(context, value, 0, 2, 1, 3);
    value = ggml_cont(context, value);
  }
  return ggml_reshape_2d(context, value, inner * outer, rows);
}
ggml_tensor *restore_indexed_rows(ggml_context *context, ggml_tensor *value,
                                  std::int64_t inner, std::int64_t rows,
                                  std::int64_t outer) {
  if (outer != 1) {
    value = ggml_reshape_3d(context, value, inner, outer, rows);
    value = ggml_permute(context, value, 0, 2, 1, 3);
    value = ggml_cont(context, value);
  }
  return value;
}
ggml_tensor *expand_broadcast(ggml_context *context, ggml_tensor *value,
                              const TensorDesc &source,
                              const TensorDesc &output) {
  value = matrix(context, value, source);
  const auto width = columns(output);
  const auto rows = static_cast<std::int64_t>(*output.element_count()) / width;
  if (same_shape(source, output) || *source.element_count() == 1 ||
      (value->ne[0] == width && value->ne[1] == 1) ||
      (value->ne[0] == 1 && value->ne[1] == rows))
    return value;
  std::array<std::int64_t, kMaxTensorRank> dimensions;
  dimensions.fill(1);
  std::ranges::copy(source.dimensions(),
                    dimensions.begin() + output.rank() - source.rank());
  auto shape = std::span(dimensions).first(output.rank());
  for (std::size_t i = shape.size(); i-- > 0;) {
    if (shape[i] == output.dimensions()[i])
      continue;
    const auto inner = extent_product(shape.subspan(i + 1));
    const auto outer = extent_product(shape.first(i));
    value = ggml_reshape_3d(context, value, inner, 1, outer);
    value =
        ggml_repeat_4d(context, value, inner, output.dimensions()[i], outer, 1);
    shape[i] = output.dimensions()[i];
  }
  return matrix(context, value, output);
}
ggml_tensor *reduction_rows(ggml_context *context, ggml_tensor *value,
                            const TensorDesc &source, Axes axes) {
  std::array<bool, kMaxTensorRank> selected{};
  std::array<std::int64_t, kMaxTensorRank> dimensions{};
  std::ranges::copy(source.dimensions(), dimensions.begin());
  auto shape = std::span(dimensions).first(source.rank());
  if (axes.is_all())
    std::fill_n(selected.begin(), source.rank(), true);
  else
    for (const auto axis : axes.values())
      selected[axis < 0 ? axis + static_cast<int>(source.rank()) : axis] = true;
  std::int64_t width = 1;
  for (std::size_t i = 0; i < shape.size(); ++i)
    if (selected[i])
      width *= shape[i];
  // Move selected dimension groups to the suffix. Shape arithmetic is cold;
  // actual data permutation belongs to the original Vulkan cont operation.
  auto target = shape.size();
  for (auto end = shape.size(); end != 0;) {
    if (!selected[end - 1]) {
      --end;
      continue;
    }
    auto begin = end - 1;
    while (begin != 0 && selected[begin - 1])
      --begin;
    if (end != target) {
      const auto outer = extent_product(shape.first(begin));
      const auto group = extent_product(shape.subspan(begin, end - begin));
      const auto between = extent_product(shape.subspan(end, target - end));
      const auto inner = extent_product(shape.subspan(target));
      value = ggml_view_4d(
          context, value, inner, group, between, outer,
          static_cast<std::size_t>(inner * between) * sizeof(float),
          static_cast<std::size_t>(inner) * sizeof(float),
          static_cast<std::size_t>(inner * between * group) * sizeof(float), 0);
      value = ggml_cont(context, value);
      std::rotate(shape.begin() + begin, shape.begin() + end,
                  shape.begin() + target);
    }
    target -= end - begin;
    end = begin;
  }
  return ggml_reshape_2d(context, value, width,
                         static_cast<std::int64_t>(*source.element_count()) /
                             width);
}
ggml_unary_op unary_operation(UnaryKind kind) noexcept {
  switch (kind) {
  case UnaryKind::kAbs:
    return GGML_UNARY_OP_ABS;
  case UnaryKind::kNeg:
    return GGML_UNARY_OP_NEG;
  case UnaryKind::kExp:
    return GGML_UNARY_OP_EXP;
  case UnaryKind::kTanh:
    return GGML_UNARY_OP_TANH;
  case UnaryKind::kSigmoid:
    return GGML_UNARY_OP_SIGMOID;
  case UnaryKind::kSilu:
    return GGML_UNARY_OP_SILU;
  case UnaryKind::kSoftplus:
    return GGML_UNARY_OP_SOFTPLUS;
  default:
    return GGML_UNARY_OP_COUNT;
  }
}
struct Imports final {
  NativeDevice device;
  std::vector<ggml_backend_buffer *> buffers;
  ~Imports() {
    auto active = device.activate();
    for (auto *buffer : buffers)
      ggml_backend_buffer_free(buffer);
  }
};
} // namespace

bool region_supports(const Operation &operation,
                     std::span<const TensorDesc> inputs,
                     std::span<const TensorDesc> outputs) noexcept {
  if (inputs.empty() || outputs.empty() ||
      !std::ranges::all_of(outputs, tensor_domain) ||
      !std::ranges::all_of(inputs, tensor_domain))
    return false;
  const bool grouped =
      operation.custom_as<model_ops::GroupedMatMul>().has_value();
  const bool embedding =
      operation.get_if<Gather>() && native_embedding_table(inputs[0]);
  if (!operation.get_if<MatMul>() && !grouped &&
      !embedding && (!std::ranges::all_of(inputs, [](const auto &input) {
        return input.is_plain();
      }) || !std::ranges::all_of(outputs, [](const auto &output) {
        return output.is_plain();
      })))
    return false;
  if (operation.get_if<Copy>())
    return inputs.size() == 1 && outputs.size() == 1 &&
           outputs[0].data_type() == inputs[0].data_type();
  if (operation.get_if<Concat>())
    return outputs.size() == 1 &&
           std::ranges::all_of(inputs, [&](const auto &input) {
             return input.data_type() == outputs[0].data_type();
           });
  if (operation.get_if<Cast>())
    return inputs.size() == 1 && outputs.size() == 1 &&
           floating(inputs[0].data_type()) &&
           floating(outputs[0].data_type()) &&
           (outputs[0].data_type() != DataType::kBFloat16 ||
            inputs[0].data_type() == DataType::kBFloat16);
  if (operation.custom_as<model_ops::Clamp>()) {
    std::array<TensorDesc, 1> inferred;
    return outputs.size() == 1 &&
           infer_output_descriptions(operation, inputs, inferred).ok() &&
           inferred[0] == outputs[0];
  }
  if (operation.custom_as<model_ops::SinkAttention>()) {
    std::array<TensorDesc, 1> inferred;
    return outputs.size() == 1 &&
           infer_output_descriptions(operation, inputs, inferred).ok() &&
           inferred[0] == outputs[0] &&
           *inputs[3].element_count() <=
               std::size_t{INT32_MAX} / sizeof(float) / inputs[0].dimensions()[2];
  }
  if (operation.custom_as<model_ops::CausalConv1d>())
    return inputs.size() == 2 && outputs.size() == 1 &&
           inputs[0].data_type() == DataType::kFloat32 &&
           inputs[1].data_type() == DataType::kFloat32 &&
           outputs[0].data_type() == DataType::kFloat32;
  if (grouped) {
    if (inputs.size() != 3 || outputs.size() != 1)
      return false;
    if (quantized_grouped_weight_domain(inputs[0]))
      return inputs[1].is_plain() && inputs[1].is_dense() &&
             inputs[1].data_type() == DataType::kFloat32 &&
             inputs[2].is_plain() && inputs[2].is_dense() &&
             inputs[2].data_type() == DataType::kInt32 &&
             outputs[0].is_plain() && outputs[0].is_dense() &&
             outputs[0].data_type() == DataType::kFloat32;
    return inputs[0].is_plain() && inputs[0].is_dense() &&
           (inputs[0].data_type() == DataType::kFloat32 ||
            inputs[0].data_type() == DataType::kBFloat16) &&
           inputs[1].is_plain() && inputs[1].is_dense() &&
           inputs[1].data_type() == DataType::kFloat32 &&
           inputs[2].is_plain() && inputs[2].is_dense() &&
           inputs[2].data_type() == DataType::kInt32 &&
           outputs[0].is_plain() && outputs[0].is_dense() &&
           outputs[0].data_type() == DataType::kFloat32;
  }
  if (operation.custom_as<model_ops::BFloatLinearF32>())
    return inputs.size() == 2 && outputs.size() == 1 &&
           outputs[0].data_type() == DataType::kFloat32;
  if (operation.custom_as<model_ops::HyperConnectionCombine>() ||
      operation.custom_as<model_ops::HyperConnectionPre>() ||
      operation.custom_as<model_ops::HyperConnectionPost>())
    return outputs.size() == 1 &&
           std::ranges::all_of(
               inputs,
               [](const auto &description) {
                 return description.is_dense() &&
                        (description.data_type() == DataType::kFloat32 ||
                         description.data_type() == DataType::kInt32);
               }) &&
           outputs[0].data_type() == DataType::kFloat32;
  if (operation.custom_as<model_ops::LightningIndexer>())
    return outputs.size() == 1 && inputs.size() == 4 && inputs[0].is_dense() &&
           inputs[1].is_dense() && inputs[2].is_dense() &&
           inputs[3].is_dense() &&
           inputs[0].data_type() == DataType::kFloat32 &&
           inputs[1].data_type() == DataType::kFloat32 &&
           inputs[2].data_type() == DataType::kFloat32 &&
           inputs[3].data_type() == DataType::kFloat16 &&
           outputs[0].data_type() == DataType::kFloat32;
  if (const auto *matmul = operation.get_if<MatMul>()) {
    if (inputs.size() == 2 && quantized_weight_domain(inputs[1]))
      return inputs.size() == 2 && outputs.size() == 1 &&
             matmul->accumulation_type == DataType::kFloat32 &&
             inputs[0].rank() == 2 && inputs[0].is_plain() &&
             inputs[0].is_dense() && outputs[0].rank() == 2 &&
             outputs[0].is_plain() && outputs[0].is_dense() &&
             inputs[0].data_type() == inputs[1].data_type() &&
             outputs[0].data_type() == inputs[0].data_type() &&
             (inputs[0].data_type() == DataType::kFloat32 ||
              inputs[0].data_type() == DataType::kFloat16);
    if (inputs.size() != 2 || outputs.size() != 1 ||
        matmul->accumulation_type != DataType::kFloat32 ||
        inputs[0].rank() < 2 || inputs[0].rank() > GGML_MAX_DIMS ||
        inputs[1].rank() < 2 || inputs[1].rank() > GGML_MAX_DIMS ||
        !inputs[0].is_dense() || !inputs[1].is_dense() ||
        inputs[0].data_type() != inputs[1].data_type() ||
        outputs[0].data_type() != inputs[0].data_type() ||
        !floating(inputs[0].data_type()))
      return false;
    std::array<std::int64_t, 2> left_batch{1, 1}, right_batch{1, 1};
    const auto fill_batch = [](const TensorDesc &description,
                               std::array<std::int64_t, 2> &batch) {
      const auto count = description.rank() - 2;
      for (std::size_t i = 0; i < count; ++i)
        batch[2 - count + i] = description.dimensions()[i];
    };
    fill_batch(inputs[0], left_batch);
    fill_batch(inputs[1], right_batch);
    return left_batch[0] % right_batch[0] == 0 &&
           left_batch[1] % right_batch[1] == 0;
  }
  if (const auto *rope = operation.get_if<RoPE>())
    return inputs.size() == 3 && outputs.size() == 1 && inputs[0].rank() == 4 &&
           rope->sequence_axis == 1 && inputs[0].is_dense() &&
           (inputs[0].data_type() == DataType::kFloat32 ||
            inputs[0].data_type() == DataType::kFloat16) &&
           inputs[1].rank() == 1 && inputs[1].data_type() == DataType::kInt32 &&
           inputs[2].rank() == 1 &&
           inputs[2].data_type() == DataType::kFloat32 &&
           outputs[0] == inputs[0];
  if (const auto *hadamard = operation.get_if<Hadamard>()) {
    const auto width = inputs[0].dimensions().back();
    return inputs.size() == 1 && outputs.size() == 1 && inputs[0].is_dense() &&
           outputs[0] == inputs[0] &&
           (inputs[0].data_type() == DataType::kFloat32 ||
            inputs[0].data_type() == DataType::kFloat16) &&
           hadamard->accumulation_type == DataType::kFloat32 &&
           (width == 64 || width == 128 || width == 256 || width == 512);
  }
  if (const auto *attention = operation.get_if<ScaledDotProductAttention>()) {
    if (inputs.size() != 3 || outputs.size() != 1 ||
        attention->layout != AttentionLayout::kBatchSequenceHeadDepth ||
        attention->mask != MaskKind::kNone ||
        attention->accumulation_type != DataType::kFloat32 ||
        !std::isfinite(attention->scale))
      return false;
    const auto type = inputs[0].data_type();
    if (type != DataType::kFloat32 && type != DataType::kFloat16)
      return false;
    for (const auto &input : inputs)
      if (!input.is_dense() || input.rank() != 4 || input.data_type() != type)
        return false;
    const auto query = inputs[0].dimensions();
    const auto key = inputs[1].dimensions();
    const auto value = inputs[2].dimensions();
    const std::array mla_shape{query[0], query[1], query[2], std::int64_t{512}};
    const auto mla_output = TensorDesc::dense(DataType::kFloat32, mla_shape);
    const bool absorbed_mla = type == DataType::kFloat32 &&
        (query[2] == 64 || query[2] == 96) &&
        key[2] == 1 && query[3] == 576 && value[3] == 512 && value[0] == key[0] &&
        value[1] == key[1] && value[2] == key[2] && mla_output && outputs[0] == *mla_output;
    return query[0] == key[0] && query[1] > 0 && key[1] >= query[1] &&
           query[2] > 0 && key[2] > 0 && query[2] % key[2] == 0 &&
           query[3] == key[3] &&
           ((inputs[1] == inputs[2] && outputs[0] == inputs[0]) || absorbed_mla);
  }
  const bool recurrent =
      operation.custom_as<model_ops::RecurrentGatedDeltaNet>().has_value();
  if (operation.get_if<GatedDeltaNet>() || recurrent) {
    std::array<TensorDesc, 2> inferred;
    if (inputs.size() != (recurrent ? 8U : 6U) || outputs.size() != 2 ||
        !infer_output_descriptions(operation, inputs, inferred).ok() ||
        !std::ranges::equal(inferred, outputs) ||
        !std::ranges::all_of(inputs, [](const auto &input) {
          return input.rank() == 4 && input.data_type() == DataType::kFloat32;
        }))
      return false;
    const auto q = inputs[0].dimensions();
    const auto v = inputs[2].dimensions();
    const auto depth = q[3];
    return v[2] % q[2] == 0 && depth == v[3] &&
           (depth == 16 || depth == 32 || depth == 64 || depth == 128) &&
           outputs[0].data_type() == DataType::kFloat32 &&
           outputs[1].data_type() == DataType::kFloat32 &&
           *outputs[0].element_count() + *outputs[1].element_count() <=
               std::size_t{INT32_MAX};
  }
  if (const auto *topk = operation.get_if<TopK>())
    return inputs.size() == 1 && outputs.size() == 2 && inputs[0].rank() > 0 &&
           inputs[0].rank() <= GGML_MAX_DIMS &&
           inputs[0].data_type() == DataType::kFloat32 &&
           outputs[0].data_type() == DataType::kFloat32 &&
           outputs[1].data_type() == DataType::kInt32 &&
           (topk->axis == -1 ||
            topk->axis == static_cast<std::int32_t>(inputs[0].rank()) - 1);
  if (const auto *gather = operation.get_if<Gather>()) {
    if (native_embedding_table(inputs[0]))
      return inputs.size() == 2 && outputs.size() == 1 &&
             gather->bounds == IndexBoundsPolicy::kUnchecked &&
             (gather->axis == 0 || gather->axis == -2) &&
             inputs[1].is_plain() && inputs[1].is_dense() &&
             inputs[1].data_type() == DataType::kInt32 &&
             inputs[1].rank() == 1 && outputs[0].is_plain() &&
             outputs[0].is_dense() &&
             outputs[0].data_type() == DataType::kFloat32 &&
             outputs[0].rank() == 2 &&
             outputs[0].dimensions()[0] == inputs[1].dimensions()[0] &&
             outputs[0].dimensions()[1] == inputs[0].dimensions()[1];
    return inputs.size() == 2 && outputs.size() == 1 &&
           gather->bounds == IndexBoundsPolicy::kUnchecked &&
           (gather->axis == 0 || gather->axis == -2) &&
           inputs[0].data_type() == DataType::kInt32 && inputs[0].rank() == 2 &&
           inputs[1].data_type() == DataType::kInt32 && inputs[1].rank() == 1 &&
           outputs[0].data_type() == DataType::kInt32;
  }
  if (const auto *scatter = operation.get_if<Scatter>())
    return inputs.size() == 3 && outputs.size() == 1 &&
           scatter->kind == ScatterKind::kAdd &&
           scatter->bounds == IndexBoundsPolicy::kUnchecked &&
           inputs[1].data_type() == DataType::kInt32 &&
           (inputs[0].data_type() == DataType::kFloat32 ||
            inputs[0].data_type() == DataType::kFloat16) &&
           inputs[2].data_type() == inputs[0].data_type() &&
           outputs[0].data_type() == inputs[0].data_type();
  if (outputs.size() != 1 ||
      (outputs[0].data_type() != DataType::kFloat32 &&
       outputs[0].data_type() != DataType::kFloat16) ||
      !std::ranges::all_of(inputs, [](const auto &input) {
        return floating(input.data_type());
      }))
    return false;
  if (const auto *norm = operation.get_if<RmsNorm>())
    return norm->accumulation_type == DataType::kFloat32;
  if (const auto *norm = operation.get_if<LayerNorm>())
    return norm->accumulation_type == DataType::kFloat32;
  if (const auto *reduce = operation.get_if<Reduce>()) {
    return inputs.size() == 1 && inputs[0].rank() > 0 &&
           (reduce->kind == ReductionKind::kSum ||
            reduce->kind == ReductionKind::kMean) &&
           (reduce->accumulation_type == DataType::kFloat32 ||
            reduce->accumulation_type == DataType::kUnknown);
  }
  if (const auto *softmax = operation.get_if<Softmax>())
    return inputs[0].rank() > 0 && softmax->kind == SoftmaxKind::kSoftmax &&
           (softmax->accumulation_type == DataType::kFloat32 ||
            softmax->accumulation_type == DataType::kUnknown) &&
           ((softmax->mask == MaskKind::kNone && inputs.size() == 1) ||
            (softmax->mask == MaskKind::kAdditive && inputs.size() == 2 &&
             inputs[0].data_type() == inputs[1].data_type()));
  if (const auto *scan = operation.get_if<Scan>())
    return inputs.size() == 1 && inputs[0].data_type() == DataType::kFloat32 &&
           outputs[0].data_type() == DataType::kFloat32 &&
           inputs[0].rank() > 0 && inputs[0].rank() <= GGML_MAX_DIMS &&
           !scan->exclusive &&
           (scan->axis == -1 ||
            scan->axis == static_cast<std::int32_t>(inputs[0].rank()) - 1);
  if (const auto *unary = operation.get_if<Unary>())
    return unary_operation(unary->kind) != GGML_UNARY_OP_COUNT ||
           unary->kind == UnaryKind::kLog || unary->kind == UnaryKind::kSqrt ||
           unary->kind == UnaryKind::kRsqrt;
  if (const auto *binary = operation.get_if<Binary>())
    return inputs.size() == 2 && (binary->kind == BinaryKind::kAdd ||
                                  binary->kind == BinaryKind::kSubtract ||
                                  binary->kind == BinaryKind::kMultiply ||
                                  binary->kind == BinaryKind::kDivide);
  if (const auto *gate = operation.get_if<GatedActivation>())
    return gate->activation == GatedActivationKind::kSilu;
  return false;
}

struct Region::State final {
  NativeDevice device;
  std::shared_ptr<Imports> imports;
  std::vector<ggml_tensor *> ports;
  std::vector<ggml_tensor *> views;
  NativeGraph graph;
  NativeWork work; // Retires before graph/imports/device.
  bool bound = false;
  bool pending = false;
};

Result<Region> Region::create(NativeDevice device,
                              const RegionRecipe &recipe) noexcept {
  if (!device || recipe.nodes.empty() || recipe.outputs.empty() ||
      recipe.values.size() > INT32_MAX / 8 ||
      recipe.nodes.size() > INT32_MAX / 16)
    return std::unexpected(
        Status::InvalidArgument("ggml Vulkan region recipe"));
  try {
    std::vector<bool> available(recipe.values.size(), false);
    for (const auto input : recipe.inputs) {
      if (input >= available.size() || available[input] ||
          !tensor_domain(recipe.values[input]))
        return std::unexpected(
            Status::InvalidArgument("ggml Vulkan region input"));
      available[input] = true;
    }
    for (const auto &node : recipe.nodes) {
      if (node.outputs.size() != operation_traits(node.operation).outputs)
        return std::unexpected(
            Status::InvalidArgument("ggml Vulkan region SSA"));
      std::vector<TensorDesc> inputs;
      for (const auto input : node.inputs) {
        if (input >= available.size() || !available[input])
          return std::unexpected(
              Status::InvalidArgument("ggml Vulkan region topology"));
        inputs.push_back(recipe.values[input]);
      }
      std::vector<TensorDesc> inferred(node.outputs.size());
      HYLE_RETURN_IF_ERROR(
          infer_output_descriptions(node.operation, inputs, inferred));
      if (!region_supports(node.operation, inputs, inferred))
        return std::unexpected(
            Status::Unimplemented("ggml Vulkan region semantic domain"));
      for (std::size_t port = 0; port < node.outputs.size(); ++port) {
        const auto output = node.outputs[port];
        if (output >= available.size() || available[output] ||
            inferred[port] != recipe.values[output])
          return std::unexpected(
              Status::InvalidArgument("ggml Vulkan region output SSA/schema"));
        available[output] = true;
      }
    }
    std::vector<bool> published(recipe.values.size(), false);
    for (const auto output : recipe.outputs) {
      if (output >= available.size() || !available[output] ||
          published[output] ||
          std::ranges::find(recipe.inputs, output) != recipe.inputs.end())
        return std::unexpected(
            Status::InvalidArgument("ggml Vulkan region output"));
      published[output] = true;
    }
    // TopK/Scan: at most 10/16 nodes including reshape/view/cont adapters.
    // LayerNorm: input reshape, GroupNorm, restored shape, optional Mul/Add.
    // Row reduction: up to three exact row pieces, concat and shape restore.
    // GatedDeltaNet: fused native result, two views and two owning ports.
    // Other operations need at most three (Rsqrt: Fill, Sqrt, Div).
    auto capacity = recipe.inputs.size() + 16;
    for (const auto &node : recipe.nodes) {
      capacity += node.operation.custom_as<model_ops::RecurrentGatedDeltaNet>() ? 9
                  : node.operation.get_if<Scan>()            ? 16
                  : node.operation.get_if<Reduce>()        ? 12
                  : node.operation.get_if<Scatter>()       ? 10
                  : node.operation.get_if<TopK>()          ? 10
                  : node.operation.get_if<LayerNorm>()     ? 5
                  : node.operation.get_if<GatedDeltaNet>() ? 5
                                                           : 3;
      // Layout translations use at most four nodes per public dimension and
      // two per input port, in addition to the established native recipe.
      auto rank = recipe.values[node.outputs[0]].rank();
      for (const auto input : node.inputs)
        rank = (std::max)(rank, recipe.values[input].rank());
      capacity += 4 * rank + 2 * node.inputs.size() + 4;
    }
    if (capacity > INT32_MAX / 4)
      return std::unexpected(
          Status::ResourceExhausted("ggml Vulkan region node capacity"));
    const auto graph_bytes = ggml_graph_overhead_custom(capacity, false);
    if (capacity > (SIZE_MAX - graph_bytes) / ggml_tensor_overhead())
      return std::unexpected(
          Status::ResourceExhausted("ggml Vulkan region metadata capacity"));
    HYLE_ASSIGN_OR_RETURN(
        auto arena,
        MetadataArena::create(capacity * ggml_tensor_overhead() + graph_bytes,
                              mem::system_allocator()));
    Region result;
    result.state_ = std::make_shared<State>();
    auto &state = *result.state_;
    state.device = std::move(device);
    state.imports = std::make_shared<Imports>();
    state.imports->device = state.device;
    const auto port_count = recipe.inputs.size() + recipe.outputs.size();
    state.ports.reserve(port_count);
    state.imports->buffers.reserve(port_count);
    auto active = state.device.activate();
    auto *context = arena.get();
    std::vector<ggml_tensor *> tensors(recipe.values.size());
    for (const auto input : recipe.inputs) {
      const auto &description = recipe.values[input];
      tensors[input] = native_tensor(context, description);
      if (tensors[input] == nullptr)
        return std::unexpected(
            Status::Unimplemented("ggml Vulkan input tensor description"));
      ggml_set_input(tensors[input]);
      state.ports.push_back(tensors[input]);
    }
    for (const auto &node : recipe.nodes) {
      const bool arithmetic =
          node.operation.get_if<Unary>() || node.operation.get_if<Binary>() ||
          node.operation.get_if<GatedActivation>() ||
          node.operation.get_if<RmsNorm>() ||
          node.operation.get_if<LayerNorm>() ||
          node.operation.get_if<Reduce>() || node.operation.get_if<Softmax>();
      std::vector<ggml_tensor *> arguments;
      arguments.reserve(node.inputs.size());
      for (const auto input : node.inputs) {
        auto *value = tensors[input];
        arguments.push_back(arithmetic && value->type != GGML_TYPE_F32
                                ? ggml_cast(context, value, GGML_TYPE_F32)
                                : value);
      }
      auto *left = arguments[0];
      ggml_tensor *output = nullptr;
      if (node.operation.get_if<Copy>()) {
        output = ggml_dup(context, left);
      } else if (const auto *cast = node.operation.get_if<Cast>()) {
        const auto target = native_type(cast->output_type);
        if (left->type == target)
          output = ggml_dup(context, left);
        else {
          if (left->type != GGML_TYPE_F32 && target != GGML_TYPE_F32)
            left = ggml_cast(context, left, GGML_TYPE_F32);
          output = ggml_cast(context, left, target);
        }
      } else if (const auto *concat = node.operation.get_if<Concat>()) {
        const auto &description = recipe.values[node.outputs[0]];
        const auto axis = static_cast<std::size_t>(
            concat->axis < 0
                ? concat->axis + static_cast<int>(description.rank())
                : concat->axis);
        const auto inner =
            extent_product(description.dimensions().subspan(axis + 1));
        const auto outer = extent_product(description.dimensions().first(axis));
        for (const auto input : node.inputs) {
          auto *part =
              ggml_reshape_3d(context, tensors[input], inner,
                              recipe.values[input].dimensions()[axis], outer);
          output = output ? ggml_concat(context, output, part, 1) : part;
        }
        if (node.inputs.size() == 1)
          output = ggml_dup(context, output);
      } else if (const auto clamp = node.operation.custom_as<model_ops::Clamp>()) {
        output = ggml_clamp(context, arguments[0], clamp->minimum, clamp->maximum);
      } else if (node.operation.custom_as<model_ops::CausalConv1d>()) {
        auto *samples =
            ggml_cont(context, ggml_permute(context, arguments[0], 1, 0, 2, 3));
        output = ggml_ssm_conv(context, samples, arguments[1]);
      } else if (node.operation.custom_as<model_ops::GroupedMatMul>()) {
        output =
            ggml_mul_mat_id(context, arguments[0], arguments[1], arguments[2]);
        if (!ggml_prec_set_acc(output, GGML_PREC_F32))
          return std::unexpected(Status::Unimplemented(
              "original ggml grouped F32 accumulation unavailable"));
      } else if (const auto descriptor =
                     node.operation
                         .custom_as<model_ops::HyperConnectionCombine>()) {
        output =
            ggml_dsv4_hc_comb(context, arguments[0], arguments[1], arguments[2],
                              descriptor->epsilon, descriptor->iterations);
      } else if (node.operation.custom_as<model_ops::HyperConnectionPre>()) {
        output = ggml_dsv4_hc_pre(context, arguments[0], arguments[1]);
      } else if (node.operation.custom_as<model_ops::HyperConnectionPost>()) {
        output = ggml_dsv4_hc_post(context, arguments[0], arguments[1],
                                   arguments[2], arguments[3]);
      } else if (node.operation.custom_as<model_ops::LightningIndexer>()) {
        output = ggml_lightning_indexer(context, arguments[0], arguments[1],
                                        arguments[2], arguments[3]);
      } else if (node.operation.get_if<MatMul>() ||
                 node.operation.custom_as<model_ops::BFloatLinearF32>()) {
        // The fixed upstream Vulkan small-matvec selector admits a BF16
        // activation for up to eight rows (or one native row per batch), but
        // its selected pipeline
        // accepts only F32/F16/Q8_1 on that port. Keep the BF16 weight and
        // FP32 accumulation while widening only the tiny activation boundary;
        // otherwise the backend aborts instead of returning unsupported.
        const bool small_matvec =
            left->ne[1] == 1 ||
            (left->ne[1] <= 8 && left->ne[2] * left->ne[3] == 1);
        if (left->type == GGML_TYPE_BF16 && small_matvec)
          left = ggml_cast(context, left, GGML_TYPE_F32);
        auto *right = arguments[1];
        const bool quantized =
            quantized_weight_domain(recipe.values[node.inputs[1]]);
        if (node.operation.get_if<MatMul>() && !quantized) {
          right = ggml_transpose(context, right);
          if (!ggml_is_contiguous(right))
            right = ggml_cont(context, right);
        }
        auto *product = ggml_mul_mat(context, right, left);
        if (!ggml_prec_set_acc(product, GGML_PREC_F32))
          return std::unexpected(
              Status::Unimplemented("ggml Vulkan MatMul F32 accumulation"));
        const auto target =
            native_type(recipe.values[node.outputs[0]].data_type());
        output = product->type == target ? product
                                         : ggml_cast(context, product, target);
      } else if (const auto *rope = node.operation.get_if<RoPE>()) {
        output = ggml_runtime::explicit_rope(
            context, left, arguments[1], arguments[2], rope->rotary_dimensions,
            rope->layout == RopeLayout::kInterleaved);
      } else if (node.operation.get_if<Hadamard>()) {
        const auto width = recipe.values[node.inputs[0]].dimensions().back();
        auto *source = left->type == GGML_TYPE_F32
                           ? left
                           : ggml_cast(context, left, GGML_TYPE_F32);
        auto *transform =
            ggml_new_tensor_2d(context, GGML_TYPE_F32, width, width);
        output = ggml_mul_mat(context, transform, source);
        ggml_mul_mat_set_hint(output, GGML_HINT_SRC0_IS_HADAMARD);
        if (!ggml_prec_set_acc(output, GGML_PREC_F32))
          return std::unexpected(
              Status::Unimplemented("ggml Vulkan Hadamard F32 accumulation"));
        if (recipe.values[node.outputs[0]].data_type() == DataType::kFloat16)
          output = ggml_cast(context, output, GGML_TYPE_F16);
      } else if (const auto sink = node.operation.custom_as<model_ops::SinkAttention>()) {
        auto *query = ggml_permute(context, arguments[0], 0, 2, 1, 3);
        auto *key = ggml_permute(context, arguments[1], 0, 2, 1, 3);
        auto *value = ggml_permute(context, arguments[2], 0, 2, 1, 3);
        auto *scores = ggml_mul_mat(context, key, query);
        if (!ggml_prec_set_acc(scores, GGML_PREC_F32))
          return std::unexpected(Status::Unimplemented("ggml Vulkan sink attention score accumulation"));
        auto *probabilities = ggml_soft_max_ext(context, scores, arguments[3], sink->scale, 0.0F);
        ggml_soft_max_add_sinks(probabilities, arguments[4]);
        value = ggml_cont(context, ggml_transpose(context, value));
        output = ggml_mul_mat(context, value, probabilities);
        if (!ggml_prec_set_acc(output, GGML_PREC_F32))
          return std::unexpected(Status::Unimplemented("ggml Vulkan sink attention value accumulation"));
        output = ggml_cont(context, ggml_permute(context, output, 0, 2, 1, 3));
      } else if (const auto *attention =
                     node.operation.get_if<ScaledDotProductAttention>()) {
        auto *query = ggml_permute(
            context, ggml_cast(context, left, GGML_TYPE_F32), 0, 2, 1, 3);
        auto *key = ggml_permute(
            context, ggml_cast(context, arguments[1], GGML_TYPE_F32), 0, 2, 1,
            3);
        auto *value = ggml_permute(
            context, ggml_cast(context, arguments[2], GGML_TYPE_F32), 0, 2, 1,
            3);
        auto *scores = ggml_mul_mat(context, key, query);
        if (!ggml_prec_set_acc(scores, GGML_PREC_F32))
          return std::unexpected(Status::Unimplemented(
              "ggml Vulkan attention score accumulation"));
        scores = ggml_scale(context, scores, attention->scale);
        if (attention->causal)
          scores = ggml_diag_mask_inf(
              context, scores, static_cast<int>(key->ne[1] - query->ne[1]));
        auto *probabilities = ggml_soft_max(context, scores);
        value = ggml_cont(context, ggml_transpose(context, value));
        output = ggml_mul_mat(context, value, probabilities);
        if (!ggml_prec_set_acc(output, GGML_PREC_F32))
          return std::unexpected(Status::Unimplemented(
              "ggml Vulkan attention value accumulation"));
        output = ggml_cont(context, ggml_permute(context, output, 0, 2, 1, 3));
        if (recipe.values[node.outputs[0]].data_type() == DataType::kFloat16)
          output = ggml_cast(context, output, GGML_TYPE_F16);
      } else if (node.operation.get_if<GatedDeltaNet>() ||
                 node.operation.custom_as<model_ops::RecurrentGatedDeltaNet>()) {
        auto *value = arguments[2];
        auto *gate = arguments[3];
        auto *beta = arguments[4];
        auto *initial = arguments[5];
        if (node.operation.custom_as<model_ops::RecurrentGatedDeltaNet>()) {
          gate = ggml_mul(context,
                          ggml_softplus(context,
                                        ggml_add(context, gate, arguments[6])),
                          arguments[5]);
          beta = ggml_sigmoid(context, beta);
          initial = arguments[7];
        }
        auto *fused = ggml_gated_delta_net(
            context, left, arguments[1], value, gate, beta, initial, 1);
        // Upstream owns the fused recurrence and concatenates scores/state.
        // Published ports have independent storage; initial state stays
        // read-only.
        auto *scores = ggml_view_4d(context, fused, value->ne[0], value->ne[1],
                                    value->ne[2], value->ne[3], value->nb[1],
                                    value->nb[2], value->nb[3], 0);
        auto *next =
            ggml_view_4d(context, fused, initial->ne[0], initial->ne[1],
                         initial->ne[2], initial->ne[3], initial->nb[1],
                         initial->nb[2], initial->nb[3], ggml_nbytes(value));
        output = ggml_cont(context, scores);
        tensors[node.outputs[1]] = ggml_cont(context, next);
      } else if (const auto *topk = node.operation.get_if<TopK>()) {
        const auto width = left->ne[0];
        const auto rows = ggml_nelements(left) / width;
        const auto count = static_cast<std::int64_t>(topk->count);
        auto *matrix = ggml_reshape_2d(context, left, width, rows);
        auto *sorted = ggml_argsort(context, matrix,
                                    topk->largest ? GGML_SORT_ORDER_DESC
                                                  : GGML_SORT_ORDER_ASC);
        auto *selected =
            ggml_view_2d(context, sorted, count, rows, sorted->nb[1], 0);
        std::int64_t shape[GGML_MAX_DIMS]{1, 1, 1, 1};
        std::ranges::reverse_copy(recipe.values[node.outputs[0]].dimensions(),
                                  shape);
        auto *indices = ggml_cont_4d(context, selected, shape[0], shape[1],
                                     shape[2], shape[3]);
        auto *row_indices = ggml_reshape_2d(context, indices, count, rows);
        // Reinterpret 32-bit storage, never numerically convert scores to I32.
        // Original GET_ROWS(I32) uses uint loads/stores, retaining NaN
        // payloads, signed zeros and subnormals. Contiguous published tensors
        // own ports.
        auto *raw = ggml_reshape_2d(context, matrix, width, rows);
        raw->type = GGML_TYPE_I32;
        auto *scalars = ggml_reshape_3d(context, raw, 1, width, rows);
        auto *gathered = ggml_get_rows(context, scalars, row_indices);
        auto *floats = ggml_reshape_4d(context, gathered, 1, count, rows, 1);
        floats->type = GGML_TYPE_F32;
        output = ggml_cont_4d(context, floats, shape[0], shape[1], shape[2],
                              shape[3]);
        tensors[node.outputs[1]] = indices;
      } else if (node.operation.get_if<Gather>()) {
        output = ggml_get_rows(context, left, arguments[1]);
      } else if (const auto *scatter = node.operation.get_if<Scatter>()) {
        const auto &base = recipe.values[node.inputs[0]];
        const auto axis = static_cast<std::size_t>(
            scatter->axis < 0
                ? scatter->axis + static_cast<std::int32_t>(base.rank())
                : scatter->axis);
        const auto inner = extent_product(base.dimensions().subspan(axis + 1));
        const auto outer = extent_product(base.dimensions().first(axis));
        const auto extent = base.dimensions()[axis];
        const auto count = static_cast<std::int64_t>(
            *recipe.values[node.inputs[1]].element_count());
        auto *base_value =
            arguments[0]->type == GGML_TYPE_F32
                ? arguments[0]
                : ggml_cast(context, arguments[0], GGML_TYPE_F32);
        auto *update_value =
            arguments[2]->type == GGML_TYPE_F32
                ? arguments[2]
                : ggml_cast(context, arguments[2], GGML_TYPE_F32);
        auto *base_rows =
            indexed_rows(context, base_value, inner, extent, outer);
        auto *update_rows =
            indexed_rows(context, update_value, inner, count, outer);
        auto *indices = ggml_reshape_1d(context, arguments[1], count);
        auto *delta =
            ggml_get_rows_back(context, update_rows, indices, base_rows);
        output = restore_indexed_rows(
            context, ggml_add(context, base_rows, delta), inner, extent, outer);
        if (base.data_type() == DataType::kFloat16)
          output = ggml_cast(context, output, GGML_TYPE_F16);
      } else if (node.operation.get_if<Scan>()) {
        const auto width = left->ne[0];
        const auto rows = ggml_nelements(left) / width;
        const auto tile = rows > 262144 ? 262144 : 512;
        if (rows > 512 && rows % tile != 0) {
          // Original single-pass CUMSUM rounds the XYZ row dispatch and has
          // no row guard. At most three native pieces cover it exactly.
          auto *matrix = ggml_reshape_2d(context, left, width, rows);
          std::int64_t offset = 0;
          while (offset < rows) {
            const auto remaining = rows - offset;
            const auto block = remaining > 262144 ? 262144
                               : remaining > 512  ? 512
                                                  : 1;
            const auto count = remaining / block * block;
            auto *part = ggml_view_2d(context, matrix, width, count,
                                      matrix->nb[1], offset * matrix->nb[1]);
            // The multi-pass variant requires descriptor-aligned storage.
            if (offset != 0)
              part = ggml_cont(context, part);
            part = ggml_cumsum(context, part);
            output = output ? ggml_concat(context, output, part, 1) : part;
            offset += count;
          }
          if (left->ne[2] != 1 || left->ne[3] != 1)
            output = ggml_cont_4d(context, output, left->ne[0], left->ne[1],
                                  left->ne[2], left->ne[3]);
        } else {
          output = ggml_cumsum(context, left);
        }
      } else if (const auto *reduce = node.operation.get_if<Reduce>()) {
        left = reduction_rows(context, left, recipe.values[node.inputs[0]],
                              reduce->axes);
        const auto row_reduce = [&](ggml_tensor *value) {
          return reduce->kind == ReductionKind::kMean
                     ? ggml_mean(context, value)
                     : ggml_sum_rows(context, value);
        };
        const auto rows = ggml_nelements(left) / left->ne[0];
        const auto tile = rows > 262144 ? 262144 : 512;
        if (rows > 512 && rows % tile != 0) {
          // SUM_ROWS has no padded-row guard. Preserve the upstream kernel,
          // dispatching at most three exact XYZ row tiles, as for CUMSUM.
          auto *matrix = ggml_reshape_2d(context, left, left->ne[0], rows);
          std::int64_t offset = 0;
          while (offset < rows) {
            const auto remaining = rows - offset;
            const auto block = remaining > 262144 ? 262144
                               : remaining > 512  ? 512
                                                  : 1;
            const auto count = remaining / block * block;
            auto *part = ggml_view_2d(context, matrix, left->ne[0], count,
                                      matrix->nb[1], offset * matrix->nb[1]);
            part = row_reduce(part);
            output = output ? ggml_concat(context, output, part, 1) : part;
            offset += count;
          }
          if (left->ne[2] != 1 || left->ne[3] != 1)
            output = ggml_cont_4d(context, output, 1, left->ne[1], left->ne[2],
                                  left->ne[3]);
        } else {
          output = row_reduce(left);
        }
      } else if (const auto *softmax = node.operation.get_if<Softmax>()) {
        const auto &description = recipe.values[node.inputs[0]];
        const auto axis = static_cast<std::size_t>(
            softmax->axis < 0
                ? softmax->axis + static_cast<int>(description.rank())
                : softmax->axis);
        const auto width = description.dimensions()[axis];
        const auto inner =
            extent_product(description.dimensions().subspan(axis + 1));
        const auto outer = extent_product(description.dimensions().first(axis));
        const auto rows = [&](ggml_tensor *value) {
          if (inner != 1) {
            value = ggml_reshape_3d(context, value, inner, width, outer);
            value =
                ggml_cont(context, ggml_permute(context, value, 1, 0, 2, 3));
          }
          return ggml_reshape_2d(context, value, width, inner * outer);
        };
        left = rows(left);
        ggml_tensor *mask = nullptr;
        if (node.inputs.size() == 2) {
          mask = expand_broadcast(context, arguments[1],
                                  recipe.values[node.inputs[1]], description);
          // Binary can broadcast a scalar/row/column directly. Softmax needs
          // the complete logical mask before changing its selected axis.
          if (ggml_nelements(mask) != ggml_nelements(left))
            mask = ggml_repeat_4d(context, mask, columns(description),
                                  ggml_nelements(left) / columns(description),
                                  1, 1);
          mask = rows(mask);
          // The public contract rounds the scale multiplication before mask
          // addition. A separate original Scale prevents fused-multiply-add.
          if (softmax->scale != 1.0F)
            left = ggml_scale(context, left, softmax->scale);
        }
        output = ggml_soft_max_ext(context, left, mask,
                                   mask ? 1.0F : softmax->scale, 0.0F);
        if (inner != 1) {
          output = ggml_reshape_3d(context, output, width, inner, outer);
          output =
              ggml_cont(context, ggml_permute(context, output, 1, 0, 2, 3));
        }
      } else if (const auto *norm = node.operation.get_if<RmsNorm>()) {
        output = ggml_rms_norm(context, left, norm->epsilon);
        if (node.inputs.size() > 1)
          output = ggml_mul(context, output, arguments[1]);
      } else if (const auto *layer = node.operation.get_if<LayerNorm>()) {
        // Original GroupNorm computes centered variance. One group per batch
        // makes each last-axis row a group without changing the shader.
        const auto rows = ggml_nelements(left) / left->ne[0];
        auto *grouped = ggml_reshape_4d(context, left, left->ne[0], 1, 1, rows);
        output = ggml_group_norm(context, grouped, 1, layer->epsilon);
        if (node.inputs.size() == 1) {
          // A published port must own storage, not alias an internal result.
          output = ggml_cont_4d(context, output, left->ne[0], left->ne[1],
                                left->ne[2], left->ne[3]);
        } else {
          output = ggml_reshape_4d(context, output, left->ne[0], left->ne[1],
                                   left->ne[2], left->ne[3]);
          output = ggml_mul(context, output, arguments[1]);
          if (node.inputs.size() > 2)
            output = ggml_add(context, output, arguments[2]);
        }
      } else if (const auto *unary = node.operation.get_if<Unary>()) {
        if (unary->kind == UnaryKind::kLog)
          output = ggml_log(context, left);
        else if (unary->kind == UnaryKind::kSqrt)
          output = ggml_sqrt(context, left);
        else if (unary->kind == UnaryKind::kRsqrt)
          output = ggml_div(context, ggml_fill(context, left, 1.0F),
                            ggml_sqrt(context, left));
        else
          output = ggml_unary(context, left, unary_operation(unary->kind));
      } else if (const auto *gate = node.operation.get_if<GatedActivation>()) {
        output = gate->inputs == GatedInputKind::kPackedHalves
                     ? ggml_swiglu(context, left)
                     : ggml_swiglu_split(context, left, arguments[1]);
      } else {
        auto *right = arguments[1];
        const auto kind = node.operation.get_if<Binary>()->kind;
        const auto &description = recipe.values[node.outputs[0]];
        if (description.rank() > GGML_MAX_DIMS) {
          left = expand_broadcast(context, left, recipe.values[node.inputs[0]],
                                  description);
          right = expand_broadcast(context, right,
                                   recipe.values[node.inputs[1]], description);
          const auto width = columns(description);
          const auto rows =
              static_cast<std::int64_t>(*description.element_count()) / width;
          if (left->ne[0] != width || left->ne[1] != rows)
            left = ggml_repeat_4d(context, left, width, rows, 1, 1);
        } else if (!same_shape(recipe.values[node.inputs[0]],
                               recipe.values[node.outputs[0]])) {
          if ((kind == BinaryKind::kAdd || kind == BinaryKind::kMultiply) &&
              same_shape(recipe.values[node.inputs[1]], description))
            std::swap(left, right);
          else {
            // ggml broadcasts only RHS. Preserve noncommutative order by
            // materializing LHS in the vendor graph and allocator.
            const auto shape = native_shape(description);
            left = ggml_repeat_4d(context, left, shape[0], shape[1], shape[2],
                                  shape[3]);
          }
        }
        switch (kind) {
        case BinaryKind::kAdd:
          output = ggml_add(context, left, right);
          break;
        case BinaryKind::kSubtract:
          output = ggml_sub(context, left, right);
          break;
        case BinaryKind::kMultiply:
          output = ggml_mul(context, left, right);
          break;
        case BinaryKind::kDivide:
          output = ggml_div(context, left, right);
          break;
        default:
          return std::unexpected(
              Status::Internal("ggml Vulkan unqualified binary recipe"));
        }
      }
      if (arithmetic &&
          output->type !=
              native_type(recipe.values[node.outputs[0]].data_type()))
        output =
            ggml_cast(context, output,
                      native_type(recipe.values[node.outputs[0]].data_type()));
      tensors[node.outputs[0]] =
          restore_shape(context, output, recipe.values[node.outputs[0]]);
    }
    auto *graph = ggml_new_graph_custom(context, capacity, false);
    for (const auto output : recipe.outputs) {
      ggml_set_output(tensors[output]);
      auto *root = tensors[output];
      while (root->view_src)
        root = root->view_src;
      // Shape restoration is a view. Bind the actual producer and protect a
      // published intermediate from native fusion and allocator reuse.
      ggml_set_output(root);
      ggml_build_forward_expand(graph, tensors[output]);
      state.ports.push_back(root);
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
      auto *node = ggml_graph_node(graph, i);
      if (node->view_src)
        state.views.push_back(node);
    }
    for (auto *tensor : state.ports) {
      HYLE_ASSIGN_OR_RETURN(auto buffer,
                            state.device.binding_buffer(ggml_nbytes(tensor)));
      state.imports->buffers.push_back(buffer);
      if (ggml_backend_tensor_alloc(buffer, tensor,
                                    ggml_backend_buffer_get_base(buffer)) !=
          GGML_STATUS_SUCCESS)
        return std::unexpected(
            Status::Internal("ggml Vulkan region port storage"));
    }
    HYLE_ASSIGN_OR_RETURN(state.graph,
                          NativeGraph::allocate(state.device, std::move(arena),
                                                graph, state.imports));
    HYLE_ASSIGN_OR_RETURN(state.work, state.device.create_work());
    return result;
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("ggml Vulkan typed region"));
  }
}

Status
Region::validate(std::span<const RegionBinding> bindings) const noexcept {
  if (!state_ || state_->pending || bindings.size() != state_->ports.size())
    return Status::FailedPrecondition("ggml Vulkan region binding state");
  auto active = state_->device.activate();
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    const auto &port = bindings[i];
    if (port.handle == VK_NULL_HANDLE && !port.owner)
      continue;
    const auto base = reinterpret_cast<std::uintptr_t>(
        ggml_backend_buffer_get_base(state_->imports->buffers[i]));
    if (port.handle == VK_NULL_HANDLE || !port.owner ||
        port.offset % sizeof(float) ||
        (ggml_is_quantized(state_->ports[i]->type) &&
         port.offset % state_->device.native().storage_buffer_alignment) ||
        port.capacity < ggml_nbytes(state_->ports[i]) ||
        port.offset > SIZE_MAX - port.capacity ||
        port.offset > UINTPTR_MAX - base)
      return Status::InvalidArgument("ggml Vulkan region native buffer range");
  }
  return Status::Ok();
}

Status Region::bind(std::span<const RegionBinding> bindings) noexcept {
  HYLE_RETURN_IF_ERROR(validate(bindings));
  auto active = state_->device.activate();
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    const auto &port = bindings[i];
    auto *buffer = state_->imports->buffers[i];
    const auto size = port.handle ? port.offset + port.capacity
                                  : ggml_nbytes(state_->ports[i]);
    state_->device.bind_buffer(buffer, port.handle, size, port.owner);
    state_->ports[i]->data = reinterpret_cast<void *>(
        reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(buffer)) +
        port.offset);
  }
  // ggml flattens view_src to the allocation root. Refresh the already frozen
  // views after rebinding a port with a different native byte origin.
  for (auto *view : state_->views)
    view->data = static_cast<char *>(view->view_src->data) + view->view_offs;
  state_->bound = std::ranges::all_of(
      bindings, [](const auto &port) { return port.handle != VK_NULL_HANDLE; });
  return Status::Ok();
}
WorkState Region::submit() noexcept {
  if (!state_ || !state_->bound)
    return {Status::FailedPrecondition("ggml Vulkan region ports are unbound")};
  const auto result =
      state_->work.submit(state_->graph.get(), state_->graph.lease());
  state_->pending = result.pending;
  return result;
}
WorkState Region::poll() noexcept {
  if (!state_)
    return {Status::FailedPrecondition("empty ggml Vulkan region")};
  const auto result = state_->work.poll();
  state_->pending = result.pending;
  return result;
}
WorkState Region::wait() noexcept {
  if (!state_)
    return {Status::FailedPrecondition("empty ggml Vulkan region")};
  const auto result = state_->work.wait();
  state_->pending = result.pending;
  return result;
}
VkFence Region::fence() const noexcept {
  return state_ ? state_->work.fence() : VK_NULL_HANDLE;
}
std::size_t Region::storage_bytes() const noexcept {
  return state_ ? state_->graph.storage_bytes() : 0;
}
} // namespace hyle::compute::ggml_vulkan::detail
