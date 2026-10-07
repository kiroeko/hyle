#include "tensor_program.h"

#include <algorithm>
#include <array>
#include <climits>

#include "explicit_rope.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

namespace hyle::compute::ggml_cuda::internal {
namespace {
ggml_type native_type(DataType type) noexcept {
  switch (type) {
  case DataType::kFloat32:
    return GGML_TYPE_F32;
  case DataType::kFloat16:
    return GGML_TYPE_F16;
  case DataType::kBFloat16:
    return GGML_TYPE_BF16;
  case DataType::kInt32:
    return GGML_TYPE_I32;
  default:
    return GGML_TYPE_COUNT;
  }
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
  case GgufQuantizedType::kIq1Xs:
    return GGML_TYPE_IQ1_XS;
  case GgufQuantizedType::kIq1Xxs:
    return GGML_TYPE_IQ1_XXS;
  case GgufQuantizedType::kIq1Xxxs:
    return GGML_TYPE_IQ1_XXXS;
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
  case GgufQuantizedType::kIq3S:
    return GGML_TYPE_IQ3_S;
  case GgufQuantizedType::kIq4Nl:
    return GGML_TYPE_IQ4_NL;
  case GgufQuantizedType::kIq4Xs:
    return GGML_TYPE_IQ4_XS;
  case GgufQuantizedType::kMxfp4:
    return GGML_TYPE_MXFP4;
  default:
    break;
  }
  return GGML_TYPE_COUNT;
}
ggml_type native_type(const TensorDesc &description) noexcept {
  const auto quantized = native_quantized_type(description);
  return quantized == GGML_TYPE_COUNT ? native_type(description.data_type())
                                      : quantized;
}
bool native_quantized(const TensorDesc &description) noexcept {
  return native_quantized_type(description) != GGML_TYPE_COUNT;
}
bool native_embedding_table(const TensorDesc &description) noexcept {
  const auto type = gguf_quantized_type(description.storage_format());
  return type &&
         (*type == GgufQuantizedType::kQ4_0 ||
          *type == GgufQuantizedType::kQ4_1 ||
          *type == GgufQuantizedType::kQ2K ||
          *type == GgufQuantizedType::kQ3K ||
          *type == GgufQuantizedType::kQ4K ||
          *type == GgufQuantizedType::kQ5K ||
          *type == GgufQuantizedType::kQ6K ||
          *type == GgufQuantizedType::kQ8_0) &&
         description.data_type() == DataType::kFloat32 &&
         description.rank() == 2 && description.block_axis() == 1;
}
bool floating(DataType type) noexcept {
  return type == DataType::kFloat32 || type == DataType::kFloat16 ||
         type == DataType::kBFloat16;
}
std::size_t elements(const TensorDesc &description) noexcept {
  return description.element_count().value();
}
std::int64_t columns(const TensorDesc &description) noexcept {
  return description.rank() == 0 ? 1 : description.dimensions().back();
}
std::int64_t rows(const TensorDesc &description) noexcept {
  const auto width = columns(description);
  return width == 0 ? 0
                    : static_cast<std::int64_t>(elements(description)) / width;
}
bool native_matmul_rhs(const TensorDesc &description) noexcept {
  auto physical = description.transposed();
  return physical && physical->is_dense();
}
bool representable(const TensorDesc &description,
                   bool integer_concat = false,
                   bool native_embedding_source = false) noexcept {
  const auto count = description.element_count();
  const auto bytes = description.required_bytes();
  // Native kernels use int element counts and unsigned 32-bit byte strides.
  // Leave room for rounded launch sizes and FP32 conversion intermediates.
  const auto element_limit = native_embedding_source ? std::size_t(INT32_MAX - 1024)
      : std::size_t(INT32_MAX - 1024) / sizeof(float);
  if (!description.valid() || !count || !bytes || *count == 0 ||
      *count > element_limit || *bytes > INT32_MAX ||
      !(floating(description.data_type()) ||
        (integer_concat && description.is_dense() &&
         description.data_type() == DataType::kInt32)))
    return false;
  if (native_quantized(description))
    return description.rank() > 0 && description.rank() <= GGML_MAX_DIMS &&
           description.storage_format().kind() == StorageFormatKind::kBlocked &&
           description.block_axis().has_value() &&
           std::ranges::all_of(description.byte_strides(), [](auto stride) {
             return stride > 0 && stride <= INT32_MAX;
           });
  return description.is_plain() &&
         (description.is_dense() || description.rank() <= GGML_MAX_DIMS) &&
         std::ranges::all_of(description.byte_strides(), [&](auto stride) {
           return stride <= INT32_MAX &&
                  stride % static_cast<std::int64_t>(
                               data_type_size(description.data_type())) ==
                      0;
         });
}
bool matrix_operand(const TensorDesc &input,
                    const TensorDesc &output) noexcept {
  return std::ranges::equal(input.dimensions(), output.dimensions()) ||
         elements(input) == 1 ||
         (elements(input) == static_cast<std::size_t>(columns(output)) &&
          columns(input) == columns(output)) ||
         (columns(input) == 1 && rows(input) == rows(output));
}
std::int32_t axis(std::int32_t value, std::size_t rank) noexcept {
  return value < 0 ? value + static_cast<std::int32_t>(rank) : value;
}

ggml_tensor *matrix(ggml_context *context, ggml_tensor *value,
                    const TensorDesc &description) {
  if (ggml_is_contiguous(value) && value->ne[0] == columns(description) &&
      value->ne[1] == rows(description) && value->ne[2] == 1 &&
      value->ne[3] == 1)
    return value;
  if (!ggml_is_contiguous(value))
    value = ggml_cont(context, value);
  return ggml_reshape_2d(context, value, columns(description),
                         rows(description));
}
ggml_tensor *native_shape(ggml_context *context, ggml_tensor *value,
                          const TensorDesc &description) {
  if (!ggml_is_contiguous(value) || description.rank() > GGML_MAX_DIMS)
    return matrix(context, value, description);
  std::array<std::int64_t, GGML_MAX_DIMS> shape{1, 1, 1, 1};
  for (std::size_t i = 0; i < description.rank(); ++i)
    shape[i] = description.dimensions()[description.rank() - i - 1];
  return ggml_reshape_4d(context, value, shape[0], shape[1], shape[2],
                         shape[3]);
}
ggml_tensor *as_float(ggml_context *context, ggml_tensor *value) {
  return value->type == GGML_TYPE_F32
             ? value
             : ggml_cast(context, value, GGML_TYPE_F32);
}
ggml_tensor *as_output(ggml_context *context, ggml_tensor *value,
                       const TensorDesc &description) {
  if (value->type != native_type(description.data_type()))
    value = ggml_cast(context, value, native_type(description.data_type()));
  return native_shape(context, value, description);
}

std::int64_t extent_product(std::span<const std::int64_t> shape) noexcept {
  std::int64_t result = 1;
  for (const auto extent : shape)
    result *= extent;
  return result;
}

ggml_tensor *expand_broadcast(ggml_context *context, ggml_tensor *value,
                              const TensorDesc &source,
                              const TensorDesc &output) {
  if (matrix_operand(source, output))
    return value;
  std::array<std::int64_t, kMaxTensorRank> dimensions;
  dimensions.fill(1);
  const auto padding = output.rank() - source.rank();
  std::ranges::copy(source.dimensions(), dimensions.begin() + padding);
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
    for (const auto requested : axes.values())
      selected[static_cast<std::size_t>(axis(requested, source.rank()))] = true;
  std::int64_t width = 1;
  for (std::size_t i = 0; i < shape.size(); ++i)
    if (selected[i])
      width *= shape[i];

  // Move selected groups to the suffix without reordering either group set.
  // This is cold layout translation; all materialization uses original cont.
  auto target = shape.size();
  for (std::size_t end = shape.size(); end != 0;) {
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
                         static_cast<std::int64_t>(elements(source)) / width);
}

ggml_tensor *axis_matrix(ggml_context *context, ggml_tensor *value,
                         const TensorDesc &description, std::size_t selected) {
  const auto shape = description.dimensions();
  const auto inner = extent_product(shape.subspan(selected + 1));
  const auto outer = extent_product(shape.first(selected));
  if (inner != 1) {
    value = ggml_reshape_3d(context, value, inner, shape[selected], outer);
    value = ggml_cont(context, ggml_permute(context, value, 1, 0, 2, 3));
  }
  return ggml_reshape_2d(context, value, shape[selected], outer * inner);
}

ggml_tensor *restore_axis(ggml_context *context, ggml_tensor *value,
                          const TensorDesc &description, std::size_t selected) {
  const auto shape = description.dimensions();
  const auto inner = extent_product(shape.subspan(selected + 1));
  if (inner != 1) {
    const auto outer = extent_product(shape.first(selected));
    value = ggml_reshape_3d(context, value, shape[selected], inner, outer);
    value = ggml_cont(context, ggml_permute(context, value, 1, 0, 2, 3));
  }
  return value;
}

} // namespace

bool tensor_operation_supports(Operation operation,
                               std::span<const TensorDesc> inputs,
                               std::span<const TensorDesc> outputs) noexcept {
  const bool integer_concat = operation.get_if<Concat>() && outputs.size() == 1 &&
                              outputs[0].data_type() == DataType::kInt32;
  if (inputs.empty() || outputs.empty() ||
      !std::ranges::all_of(outputs, [&](const auto &description) {
        return description.is_dense() && representable(description, integer_concat);
      }))
    return false;
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (!representable(inputs[i], integer_concat,
            i == 0 && operation.get_if<Gather>() && native_embedding_table(inputs[i])) &&
        !(((operation.get_if<RoPE>() && i == 1) ||
           (operation.get_if<Gather>() && i == 1) ||
           (operation.custom_as<model_ops::GroupedMatMul>() && i == 2)) &&
          inputs[i].valid() && inputs[i].is_plain() && inputs[i].is_dense() &&
          inputs[i].data_type() == DataType::kInt32 &&
          inputs[i].element_count() &&
          *inputs[i].element_count() <= (INT32_MAX - 1024) / sizeof(float)))
      return false;
  const bool has_quantized = std::ranges::any_of(inputs, native_quantized);
  if (has_quantized && !operation.get_if<MatMul>() &&
      !operation.custom_as<model_ops::GroupedMatMul>() &&
      !operation.get_if<Gather>())
    return false;
  std::vector<TensorDesc> inferred(outputs.size());
  if (!infer_output_descriptions(operation, inputs, inferred).ok() ||
      !std::ranges::equal(inferred, outputs))
    return false;
  const auto &output = outputs[0];
  if (operation.custom_as<model_ops::SinkAttention>())
    return *inputs[3].element_count() <=
           (INT32_MAX - 1024) / sizeof(float) / inputs[0].dimensions()[2];
  if (operation.custom_as<model_ops::Clamp>())
    return true; // Public inference has already validated the entire domain.
  if (operation.custom_as<model_ops::CausalConv1d>())
    return outputs.size() == 1 && inputs.size() == 2 &&
           inputs[0].data_type() == DataType::kFloat32 &&
           inputs[1].data_type() == DataType::kFloat32;
  if (operation.custom_as<model_ops::GroupedMatMul>())
    return outputs.size() == 1 && inputs.size() == 3 &&
           (!native_quantized(inputs[0]) ||
            (inputs[0].rank() == 3 && inputs[0].block_axis() == 2)) &&
           inputs[0].is_dense() && inputs[1].is_dense() &&
           inputs[2].is_dense() &&
           (inputs[0].data_type() == DataType::kFloat32 ||
            (inputs[0].is_plain() &&
             inputs[0].data_type() == DataType::kBFloat16)) &&
           inputs[1].data_type() == DataType::kFloat32 &&
           inputs[2].data_type() == DataType::kInt32 &&
           outputs[0].data_type() == DataType::kFloat32;
  if (operation.custom_as<model_ops::BFloatLinearF32>())
    return outputs.size() == 1 && inputs.size() == 2 &&
           output.data_type() == DataType::kFloat32;
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
    return outputs.size() == 1 && inputs.size() == 4 &&
           inputs[0].data_type() == DataType::kFloat32 &&
           inputs[1].data_type() == DataType::kFloat32 &&
           inputs[2].data_type() == DataType::kFloat32 &&
           inputs[3].data_type() == DataType::kFloat16 &&
           std::ranges::all_of(inputs,
                               [](const auto &description) {
                                 return description.is_dense();
                               }) &&
           outputs[0].data_type() == DataType::kFloat32;
  const bool recurrent =
      operation.custom_as<model_ops::RecurrentGatedDeltaNet>().has_value();
  if (operation.get_if<GatedDeltaNet>() || recurrent) {
    if (outputs.size() != 2 || inputs.size() != (recurrent ? 8U : 6U) ||
        !std::ranges::all_of(inputs, [](const auto &description) {
          return description.is_dense() &&
                 description.data_type() == DataType::kFloat32;
        }))
      return false;
    const auto q = inputs[0].dimensions();
    const auto v = inputs[2].dimensions();
    return v[2] % q[2] == 0 && q[3] == v[3] &&
           *outputs[0].element_count() + *outputs[1].element_count() <=
               std::size_t{INT32_MAX};
  }
  if (outputs.size() != 1)
    return false;
  if (operation.get_if<Copy>() || operation.get_if<Concat>())
    return true;
  if (operation.get_if<Cast>())
    return (floating(inputs[0].data_type()) && floating(output.data_type())) ||
           inputs[0].data_type() == output.data_type();
  if (const auto *unary = operation.get_if<Unary>()) {
    switch (unary->kind) {
    case UnaryKind::kAbs:
    case UnaryKind::kNeg:
    case UnaryKind::kExp:
    case UnaryKind::kLog:
    case UnaryKind::kSqrt:
    case UnaryKind::kRsqrt:
    case UnaryKind::kTanh:
    case UnaryKind::kSigmoid:
      return true;
    default:
      return false;
    }
  }
  if (const auto *binary = operation.get_if<Binary>())
    return binary->kind != BinaryKind::kMinimum &&
           binary->kind != BinaryKind::kMaximum;
  if (const auto *gate = operation.get_if<GatedActivation>())
    return gate->activation == GatedActivationKind::kSilu;
  if (const auto *rms = operation.get_if<RmsNorm>())
    return rms->accumulation_type == DataType::kFloat32;
  if (const auto *norm = operation.get_if<LayerNorm>())
    return norm->accumulation_type == DataType::kFloat32;
  if (const auto *reduce = operation.get_if<Reduce>())
    return (reduce->kind == ReductionKind::kSum ||
            reduce->kind == ReductionKind::kMean) &&
           (reduce->accumulation_type == DataType::kUnknown ||
            reduce->accumulation_type == DataType::kFloat32);
  if (const auto *softmax = operation.get_if<Softmax>())
    return softmax->kind == SoftmaxKind::kSoftmax &&
           softmax->mask != MaskKind::kBoolean &&
           softmax->accumulation_type == DataType::kFloat32;
  if (const auto *gather = operation.get_if<Gather>())
    return inputs.size() == 2 &&
           gather->bounds == IndexBoundsPolicy::kChecked &&
           (gather->axis == 0 || gather->axis == -2) &&
           native_embedding_table(inputs[0]) && inputs[1].rank() == 1 &&
           inputs[1].is_dense() && inputs[1].data_type() == DataType::kInt32 &&
           output.rank() == 2 && output.is_dense() &&
           output.data_type() == DataType::kFloat32;
  if (const auto *matmul = operation.get_if<MatMul>()) {
    const bool quantized_rhs = native_quantized(inputs[1]);
    if (inputs.size() != 2 || inputs[0].rank() < 2 || inputs[0].rank() > 4 ||
        inputs[1].rank() < 2 || inputs[1].rank() > 4 ||
        (!quantized_rhs && inputs[0].rank() == 2 && inputs[1].rank() == 2) ||
        !inputs[0].is_dense() || (!quantized_rhs && !inputs[1].is_dense()) ||
        (quantized_rhs &&
         (inputs[1].rank() != 2 || inputs[1].block_axis() != 0 ||
          !native_matmul_rhs(inputs[1]) || native_quantized(inputs[0]))) ||
        matmul->accumulation_type != DataType::kFloat32 ||
        (!quantized_rhs && inputs[0].data_type() != DataType::kFloat16 &&
         inputs[0].data_type() != DataType::kBFloat16) ||
        (quantized_rhs && !floating(inputs[0].data_type())) ||
        inputs[1].data_type() != inputs[0].data_type() ||
        output.data_type() != inputs[0].data_type())
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
    return inputs.size() == 3 && inputs[0].rank() == 4 &&
           rope->sequence_axis == 1 && inputs[0].is_dense() &&
           (inputs[0].data_type() == DataType::kFloat16 ||
            inputs[0].data_type() == DataType::kBFloat16 ||
            inputs[0].data_type() == DataType::kFloat32) &&
           inputs[1].rank() == 1 && inputs[1].is_dense() &&
           inputs[1].data_type() == DataType::kInt32 && inputs[2].rank() == 1 &&
           inputs[2].is_dense() && inputs[2].data_type() == DataType::kFloat32;
  if (const auto *hadamard = operation.get_if<Hadamard>())
    return inputs.size() == 1 && inputs[0].is_dense() &&
           floating(inputs[0].data_type()) &&
           hadamard->accumulation_type == DataType::kFloat32;
  return false;
}

ggml_tensor *create_tensor_input(ggml_context *context,
                                 const TensorDesc &description) {
  if (native_quantized(description)) {
    std::array<std::int64_t, GGML_MAX_DIMS> shape{1, 1, 1, 1};
    const auto blocked = *description.block_axis();
    shape[0] = description.dimensions()[blocked];
    std::size_t target = 1;
    for (std::size_t axis = description.rank(); axis-- > 0;)
      if (axis != blocked)
        shape[target++] = description.dimensions()[axis];
    return ggml_new_tensor(context, native_type(description),
                           static_cast<int>(description.rank()), shape.data());
  }
  if (description.rank() > GGML_MAX_DIMS)
    return ggml_new_tensor_2d(context, native_type(description.data_type()),
                              columns(description), rows(description));
  std::array<std::int64_t, GGML_MAX_DIMS> shape{1, 1, 1, 1};
  std::reverse_copy(description.dimensions().begin(),
                    description.dimensions().end(), shape.begin());
  auto *value = ggml_new_tensor(
      context, native_type(description.data_type()),
      static_cast<int>(std::max(std::size_t{1}, description.rank())),
      shape.data());
  if (!description.is_dense())
    for (std::size_t i = 0; i < description.rank(); ++i)
      value->nb[i] = static_cast<std::size_t>(
          description.byte_strides()[description.rank() - i - 1]);
  return value;
}

std::vector<ggml_tensor *>
lower_tensor_operation(ggml_context *context, const Recipe &recipe,
                       const Node &node, std::span<ggml_tensor *const> values,
                       std::vector<ScalarConstant> &constants,
                       std::vector<DenseConstant> &dense_constants,
                       std::span<const float> hadamard) {
  const auto &output = recipe.values[node.outputs[0]];
  const auto input = [&](std::size_t index) {
    const auto id = node.inputs[index];
    return matrix(context, values[id], recipe.values[id]);
  };
  const auto fp = [&](std::size_t index) {
    return as_float(context, input(index));
  };
  ggml_tensor *result = nullptr;
  if (node.operation.get_if<Copy>()) {
    result = ggml_dup(context, input(0));
  } else if (node.operation.get_if<Cast>()) {
    result = ggml_cast(context, input(0), native_type(output.data_type()));
  } else if (const auto *unary = node.operation.get_if<Unary>()) {
    auto *a = fp(0);
    switch (unary->kind) {
    case UnaryKind::kAbs:
      result = ggml_abs(context, a);
      break;
    case UnaryKind::kNeg:
      result = ggml_neg(context, a);
      break;
    case UnaryKind::kExp:
      result = ggml_exp(context, a);
      break;
    case UnaryKind::kLog:
      result = ggml_log(context, a);
      break;
    case UnaryKind::kSqrt:
      result = ggml_sqrt(context, a);
      break;
    case UnaryKind::kRsqrt: {
      auto *one = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);
      constants.push_back({one, 1.0F});
      result = ggml_div(context, ggml_repeat(context, one, a),
                        ggml_sqrt(context, a));
      break;
    }
    case UnaryKind::kTanh:
      result = ggml_tanh(context, a);
      break;
    case UnaryKind::kSigmoid:
      result = ggml_sigmoid(context, a);
      break;
    default:
      break; // Stable tails and non-finite domains need another mature carrier.
    }
  } else if (const auto *binary = node.operation.get_if<Binary>()) {
    auto *a =
        expand_broadcast(context, fp(0), recipe.values[node.inputs[0]], output);
    auto *b =
        expand_broadcast(context, fp(1), recipe.values[node.inputs[1]], output);
    if (a->ne[0] != columns(output) || a->ne[1] != rows(output))
      a = ggml_repeat_4d(context, a, columns(output), rows(output), 1, 1);
    switch (binary->kind) {
    case BinaryKind::kAdd:
      result = ggml_add(context, a, b);
      break;
    case BinaryKind::kSubtract:
      result = ggml_sub(context, a, b);
      break;
    case BinaryKind::kMultiply:
      result = ggml_mul(context, a, b);
      break;
    case BinaryKind::kDivide:
      result = ggml_div(context, a, b);
      break;
    default:
      break; // Rejected by the cold capability predicate.
    }
  } else if (const auto *gate = node.operation.get_if<GatedActivation>()) {
    auto *a = input(0);
    if (a->type == GGML_TYPE_BF16)
      a = as_float(context, a);
    if (gate->inputs == GatedInputKind::kPackedHalves) {
      result = ggml_swiglu(context, a);
    } else {
      auto *b = input(1);
      if (b->type == GGML_TYPE_BF16)
        b = as_float(context, b);
      result = ggml_swiglu_split(context, a, b);
    }
  } else if (const auto *rms = node.operation.get_if<RmsNorm>()) {
    result = ggml_rms_norm(context, fp(0), rms->epsilon);
    if (node.inputs.size() > 1)
      result = ggml_mul(context, result, fp(1));
  } else if (const auto *norm = node.operation.get_if<LayerNorm>()) {
    // The original CUDA norm uses E[x*x]-E[x]*E[x]. Compose the public
    // centered-variance contract with original kernels to avoid cancellation.
    auto *a = fp(0);
    auto *centered = ggml_sub(context, a, ggml_mean(context, a));
    auto *variance = ggml_mean(context, ggml_sqr(context, centered));
    auto *epsilon = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);
    constants.push_back({epsilon, norm->epsilon});
    auto *denominator =
        ggml_sqrt(context, ggml_add(context, variance, epsilon));
    result = ggml_div(context, centered, denominator);
    if (node.inputs.size() > 1)
      result = ggml_mul(context, result, fp(1));
    if (node.inputs.size() > 2)
      result = ggml_add(context, result, fp(2));
  } else if (const auto *reduce = node.operation.get_if<Reduce>()) {
    const auto &source = recipe.values[node.inputs[0]];
    auto *a = reduction_rows(context, fp(0), source, reduce->axes);
    result = reduce->kind == ReductionKind::kMean ? ggml_mean(context, a)
                                                  : ggml_sum_rows(context, a);
  } else if (const auto *softmax = node.operation.get_if<Softmax>()) {
    const auto selected =
        static_cast<std::size_t>(axis(softmax->axis, output.rank()));
    auto *a = axis_matrix(context, fp(0), output, selected);
    ggml_tensor *mask = nullptr;
    if (softmax->mask == MaskKind::kAdditive) {
      mask = expand_broadcast(context, fp(1), recipe.values[node.inputs[1]],
                              output);
      if (mask->ne[0] != columns(output) || mask->ne[1] != rows(output))
        mask =
            ggml_repeat_4d(context, mask, columns(output), rows(output), 1, 1);
      mask = axis_matrix(context, mask, output, selected);
    }
    result = ggml_soft_max_ext(context, a, mask, softmax->scale, 0.0F);
    result = restore_axis(context, result, output, selected);
  } else if (const auto *concat = node.operation.get_if<Concat>()) {
    const auto selected =
        static_cast<std::size_t>(axis(concat->axis, output.rank()));
    std::int64_t inner = 1;
    std::int64_t outer = 1;
    for (std::size_t i = 0; i < selected; ++i)
      outer *= output.dimensions()[i];
    for (std::size_t i = selected + 1; i < output.rank(); ++i)
      inner *= output.dimensions()[i];
    for (std::size_t i = 0; i < node.inputs.size(); ++i) {
      const auto &description = recipe.values[node.inputs[i]];
      if (elements(description) == 0)
        continue;
      auto *part = ggml_reshape_3d(context, input(i), inner,
                                   description.dimensions()[selected], outer);
      // This runtime admits I32 CPY/CONCAT but explicitly declines I32 DUP.
      // Keep indices as integers, including a one-input Concat's owned copy.
      if (!result && part->type == GGML_TYPE_I32)
        result = ggml_cpy(context, part, ggml_dup_tensor(context, part));
      else
        result = result ? ggml_concat(context, result, part, 1)
                        : ggml_dup(context, part);
    }
  } else if (node.operation.get_if<Gather>()) {
    result =
        ggml_get_rows(context, values[node.inputs[0]], values[node.inputs[1]]);
  } else if (const auto clamp = node.operation.custom_as<model_ops::Clamp>()) {
    result = ggml_clamp(context, input(0), clamp->minimum, clamp->maximum);
  } else if (node.operation.custom_as<model_ops::CausalConv1d>()) {
    auto *samples = ggml_cont(
        context, ggml_permute(context, values[node.inputs[0]], 1, 0, 2, 3));
    result = ggml_ssm_conv(context, samples, values[node.inputs[1]]);
  } else if (node.operation.custom_as<model_ops::GroupedMatMul>()) {
    auto *weights = values[node.inputs[0]];
    // Native BF16 MMF casts its F32 RHS to BF16 tensor-core tiles. This
    // plain-weight contract retains F32 activations; use the original cast
    // and F32 program. Quantized storage keeps its declared activation codec.
    if (recipe.values[node.inputs[0]].is_plain() &&
        recipe.values[node.inputs[0]].data_type() == DataType::kBFloat16)
      weights = ggml_cast(context, weights, GGML_TYPE_F32);
    result = ggml_mul_mat_id(context, weights,
                             values[node.inputs[1]], values[node.inputs[2]]);
    if (!ggml_prec_set_acc(result, GGML_PREC_F32) ||
        !ggml_prec_set_src(result, GGML_PREC_F32, 1))
      return {};
  } else if (const auto descriptor =
                 node.operation
                     .custom_as<model_ops::HyperConnectionCombine>()) {
    result = ggml_dsv4_hc_comb(context, values[node.inputs[0]],
                               values[node.inputs[1]], values[node.inputs[2]],
                               descriptor->epsilon, descriptor->iterations);
  } else if (node.operation.custom_as<model_ops::HyperConnectionPre>()) {
    result = ggml_dsv4_hc_pre(context, values[node.inputs[0]],
                              values[node.inputs[1]]);
  } else if (node.operation.custom_as<model_ops::HyperConnectionPost>()) {
    result = ggml_dsv4_hc_post(context, values[node.inputs[0]],
                               values[node.inputs[1]], values[node.inputs[2]],
                               values[node.inputs[3]]);
  } else if (node.operation.custom_as<model_ops::LightningIndexer>()) {
    result = ggml_lightning_indexer(
        context, values[node.inputs[0]], values[node.inputs[1]],
        values[node.inputs[2]], values[node.inputs[3]]);
  } else if (node.operation.get_if<MatMul>() ||
             node.operation.custom_as<model_ops::BFloatLinearF32>()) {
    auto *left = native_shape(context, values[node.inputs[0]],
                              recipe.values[node.inputs[0]]);
    auto *right = values[node.inputs[1]];
    if (node.operation.get_if<MatMul>() &&
        !native_quantized(recipe.values[node.inputs[1]])) {
      right = ggml_transpose(context, right);
      if (!ggml_is_contiguous(right))
        right = ggml_cont(context, right);
    }
    result = ggml_mul_mat(context, right, left);
    if (!ggml_prec_set_acc(result, GGML_PREC_F32))
      return {};
  } else if (const auto *rope = node.operation.get_if<RoPE>()) {
    result = ggml_runtime::explicit_rope(
        context, values[node.inputs[0]], values[node.inputs[1]],
        values[node.inputs[2]], rope->rotary_dimensions,
        rope->layout == RopeLayout::kInterleaved);
  } else if (node.operation.get_if<Hadamard>()) {
    const auto width = columns(output);
    if (hadamard.size() != static_cast<std::size_t>(width * width))
      return {};
    auto *transform = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, width);
    dense_constants.push_back(
        {transform, hadamard.data(), hadamard.size_bytes()});
    result = ggml_mul_mat(context, transform, as_float(context, input(0)));
    ggml_mul_mat_set_hint(result, GGML_HINT_SRC0_IS_HADAMARD);
    if (!ggml_prec_set_acc(result, GGML_PREC_F32))
      return {};
  } else if (node.operation.get_if<GatedDeltaNet>() ||
             node.operation.custom_as<model_ops::RecurrentGatedDeltaNet>()) {
    const auto recurrent_input = [&](std::size_t port) {
      return values[node.inputs[port]];
    };
    auto *decay = recurrent_input(3);
    auto *beta = recurrent_input(4);
    auto *initial = recurrent_input(5);
    if (node.operation.custom_as<model_ops::RecurrentGatedDeltaNet>()) {
      decay = ggml_mul(context,
                       ggml_softplus(context, ggml_add(context, decay,
                                                       recurrent_input(6))),
                       recurrent_input(5));
      beta = ggml_sigmoid(context, beta);
      initial = recurrent_input(7);
    }
    auto *packed = ggml_gated_delta_net(
        context, recurrent_input(0), recurrent_input(1), recurrent_input(2),
        decay, beta, initial, 1);
    const auto score_count = static_cast<std::int64_t>(
        *recipe.values[node.outputs[0]].element_count());
    const auto state_count = static_cast<std::int64_t>(
        *recipe.values[node.outputs[1]].element_count());
    auto *scores =
        ggml_cont(context, ggml_view_1d(context, packed, score_count, 0));
    auto *state =
        ggml_cont(context, ggml_view_1d(context, packed, state_count,
                                        static_cast<std::size_t>(score_count) *
                                            sizeof(float)));
    return {as_output(context, scores, recipe.values[node.outputs[0]]),
            as_output(context, state, recipe.values[node.outputs[1]])};
  }
  if (!result)
    return {};
  return {as_output(context, result, output)};
}

} // namespace hyle::compute::ggml_cuda::internal
