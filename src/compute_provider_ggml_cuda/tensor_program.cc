#include "tensor_program.h"

#include <algorithm>
#include <array>
#include <climits>

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
bool representable(const TensorDesc &description) noexcept {
  const auto count = description.element_count();
  const auto bytes = description.required_bytes();
  // Native kernels use int element counts and unsigned 32-bit byte strides.
  // Leave room for rounded launch sizes and FP32 conversion intermediates.
  return description.valid() && description.is_plain() && count && bytes &&
         *count != 0 && *count <= (INT32_MAX - 1024) / sizeof(float) &&
         *bytes <= INT32_MAX && floating(description.data_type()) &&
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
  return ggml_reshape_4d(context, value, shape[0], shape[1], shape[2], shape[3]);
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
  if (inputs.empty() || outputs.size() != 1 || !outputs[0].is_dense() ||
      !representable(outputs[0]))
    return false;
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (!representable(inputs[i]) &&
        !(operation.get_if<RoPE>() && i == 1 && inputs[i].valid() &&
          inputs[i].is_plain() && inputs[i].is_dense() &&
          inputs[i].data_type() == DataType::kInt32 &&
          inputs[i].element_count() &&
          *inputs[i].element_count() <= (INT32_MAX - 1024) / sizeof(float)))
      return false;
  const auto inferred = infer_single_output(operation, inputs);
  if (!inferred || *inferred != outputs[0])
    return false;
  const auto &output = outputs[0];
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
      return true;
    default:
      return false;
    }
  }
  if (const auto *binary = operation.get_if<Binary>())
    return binary->kind != BinaryKind::kMinimum &&
           binary->kind != BinaryKind::kMaximum;
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
  if (const auto *matmul = operation.get_if<MatMul>()) {
    if (inputs.size() != 2 || inputs[0].rank() < 2 || inputs[0].rank() > 4 ||
        inputs[1].rank() < 2 || inputs[1].rank() > 4 ||
        (inputs[0].rank() == 2 && inputs[1].rank() == 2) ||
        !inputs[0].is_dense() || !inputs[1].is_dense() ||
        matmul->accumulation_type != DataType::kFloat32 ||
        (inputs[0].data_type() != DataType::kFloat16 &&
         inputs[0].data_type() != DataType::kBFloat16) ||
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
            inputs[0].data_type() == DataType::kBFloat16) &&
           inputs[1].rank() == 1 && inputs[1].is_dense() &&
           inputs[1].data_type() == DataType::kInt32 &&
           inputs[2].rank() == 1 && inputs[2].is_dense() &&
           inputs[2].data_type() == DataType::kFloat32;
  if (const auto *hadamard = operation.get_if<Hadamard>())
    return inputs.size() == 1 && inputs[0].is_dense() &&
           floating(inputs[0].data_type()) &&
           hadamard->accumulation_type == DataType::kFloat32;
  return false;
}

ggml_tensor *create_tensor_input(ggml_context *context,
                                 const TensorDesc &description) {
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

ggml_tensor *lower_tensor_operation(ggml_context *context, const Recipe &recipe,
                                    const Node &node,
                                    std::span<ggml_tensor *const> values,
                                    std::vector<ScalarConstant> &constants,
                                    std::vector<DenseConstant> &dense_constants,
                                    std::span<const float> hadamard) {
  const auto &output = recipe.values[node.output];
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
      result = result ? ggml_concat(context, result, part, 1)
                      : ggml_dup(context, part);
    }
  } else if (node.operation.get_if<MatMul>()) {
    auto *left = native_shape(context, values[node.inputs[0]],
                              recipe.values[node.inputs[0]]);
    auto *right = ggml_transpose(context, values[node.inputs[1]]);
    if (!ggml_is_contiguous(right))
      right = ggml_cont(context, right);
    result = ggml_mul_mat(context, right, left);
    if (!ggml_prec_set_acc(result, GGML_PREC_F32))
      return nullptr;
  } else if (const auto *rope = node.operation.get_if<RoPE>()) {
    auto *positions = values[node.inputs[1]];
    auto *frequencies = values[node.inputs[2]];
    auto *one = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);
    constants.push_back({one, 1.0F});
    auto *factors = ggml_div(context, ggml_repeat(context, one, frequencies),
                             frequencies);
    const int mode = rope->layout == RopeLayout::kInterleaved
                         ? GGML_ROPE_TYPE_NORMAL
                         : GGML_ROPE_TYPE_NEOX;
    result = ggml_rope_ext(
        context, values[node.inputs[0]], positions, factors,
        static_cast<int>(rope->rotary_dimensions), mode, 0, 1.0F, 1.0F,
        0.0F, 1.0F, 0.0F, 0.0F);
  } else if (node.operation.get_if<Hadamard>()) {
    const auto width = columns(output);
    if (hadamard.size() != static_cast<std::size_t>(width * width))
      return nullptr;
    auto *transform =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, width, width);
    dense_constants.push_back(
        {transform, hadamard.data(), hadamard.size_bytes()});
    result = ggml_mul_mat(context, transform, as_float(context, input(0)));
    ggml_mul_mat_set_hint(result, GGML_HINT_SRC0_IS_HADAMARD);
    if (!ggml_prec_set_acc(result, GGML_PREC_F32))
      return nullptr;
  }
  return result ? as_output(context, result, output) : nullptr;
}

} // namespace hyle::compute::ggml_cuda::internal
