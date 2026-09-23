#include "program.h"
#include "standard_index.h"

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <utility>

#include "cpu_execution.h"
#include "hyle/compute/backends/cpu.h"

namespace hyle::compute::ggml_cpu {
namespace {

struct Configuration final {
  Options options;
  const internal::Extension *extension = nullptr;
};
struct Program final {
  struct IndexCheck final {
    std::uint32_t input;
    std::size_t count;
    std::int64_t limit;
  };
  Options options;
  internal::Recipe recipe;
  // Cold, immutable Sylvester matrices. ggml recognizes the explicit
  // Hadamard hint and retains ownership of the mature backend execution.
  std::vector<std::vector<float>> hadamards;
  // Cold causal masks for ggml flash attention. They are ordinary upstream
  // mask payloads, broadcast across batch and heads during replay.
  std::vector<std::vector<ggml_fp16_t>> attention_masks;
  std::vector<IndexCheck> index_checks;
  float one = 1.0F;
  float zero = 0.0F;
  float nan = std::numeric_limits<float>::quiet_NaN();
};

Status allocation_failure() noexcept {
  return Status::ResourceExhausted("ggml CPU program allocation failed");
}
Status exception_status() noexcept {
  try {
    throw;
  } catch (const std::bad_alloc &) {
    return allocation_failure();
  } catch (const std::exception &error) {
    return Status::Internal(error.what());
  }
}

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
  return description.valid() && description.is_plain() && count &&
         *count <= INT32_MAX &&
         native_type(description.data_type()) != GGML_TYPE_COUNT &&
         (description.is_dense() || description.rank() <= GGML_MAX_DIMS) &&
         std::ranges::all_of(description.byte_strides(), [&](auto stride) {
           return stride % static_cast<std::int64_t>(
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
bool native_matmul_rhs(const TensorDesc &description) noexcept {
  auto physical = description.transposed();
  return physical && physical->is_dense();
}
std::int32_t axis(std::int32_t value, std::size_t rank) noexcept {
  return value < 0 ? value + static_cast<std::int32_t>(rank) : value;
}

struct Invocation final {
  mem::AllocatorRef allocator;
  ggml_runtime::MetadataArena arena;
  ggml_cgraph *graph = nullptr;
  std::vector<ggml_tensor *> values;
  std::vector<ggml_tensor *> input_roots;
  std::vector<ggml_tensor *> output_roots;
  std::vector<ggml_tensor *> views;
  std::vector<mem::HostMapping> input_maps;
  std::vector<mem::HostMapping> output_maps;
  void *storage = nullptr;
  std::size_t storage_bytes = 0;
  // Declared last: workers/plan retire before their tensors and payload.
  ggml_runtime::CpuExecution execution;

  ~Invocation() {
    execution = {};
    if (storage)
      allocator.deallocate(storage, storage_bytes, GGML_MEM_ALIGN);
  }
  void bind_views() noexcept {
    for (auto *view : views)
      view->data =
          static_cast<std::byte *>(view->view_src->data) + view->view_offs;
  }
};

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

// Turn a logical [outer, rows, inner] tensor into the two-dimensional row
// table used by ggml indexed operations.  Moving outer next to inner keeps a
// single shared index vector valid for every prefix slice.
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

ggml_tensor *lower(Program &program, Invocation &invocation,
                   const internal::Node &node, std::size_t node_index) {
  auto *context = invocation.arena.get();
  const auto &recipe = program.recipe;
  const auto &output = recipe.values[node.output];
  const auto input = [&](std::size_t index) {
    const auto id = node.inputs[index];
    return matrix(context, invocation.values[id], recipe.values[id]);
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
      one->data = &program.one;
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
    result = ggml_norm(context, fp(0), norm->epsilon);
    if (node.inputs.size() > 1)
      result = ggml_mul(context, result, fp(1));
    if (node.inputs.size() > 2)
      result = ggml_add(context, result, fp(2));
  } else if (const auto *reduce = node.operation.get_if<Reduce>()) {
    const auto &source = recipe.values[node.inputs[0]];
    if (elements(source) == 0) {
      auto *identity = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);
      identity->data =
          reduce->kind == ReductionKind::kMean ? &program.nan : &program.zero;
      result = ggml_repeat_4d(context, identity, columns(output), rows(output),
                              1, 1);
    } else {
      auto *a = reduction_rows(context, fp(0), source, reduce->axes);
      result = reduce->kind == ReductionKind::kMean ? ggml_mean(context, a)
                                                    : ggml_sum_rows(context, a);
    }
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
  } else if (const auto *gather = node.operation.get_if<Gather>()) {
    const auto &source = recipe.values[node.inputs[0]];
    const auto selected =
        static_cast<std::size_t>(axis(gather->axis, source.rank()));
    std::int64_t inner = 1;
    std::int64_t outer = 1;
    for (std::size_t i = 0; i < selected; ++i)
      outer *= source.dimensions()[i];
    for (std::size_t i = selected + 1; i < source.rank(); ++i)
      inner *= source.dimensions()[i];
    auto *table = ggml_reshape_3d(context, input(0), inner,
                                  source.dimensions()[selected], outer);
    const auto count =
        static_cast<std::int64_t>(elements(recipe.values[node.inputs[1]]));
    auto *indices = ggml_reshape_1d(context, input(1), count);
    if (outer != 1)
      indices = ggml_repeat_4d(context, indices, count, outer, 1, 1);
    result = ggml_get_rows(context, table, indices);
  } else if (const auto *scatter = node.operation.get_if<Scatter>()) {
    // SET_ROWS cannot implement the public last-writer-wins assignment when
    // indices repeat.  GET_ROWS_BACK is, however, the original mature
    // reduction needed by Scatter add and deliberately accepts duplicates.
    const auto &base = recipe.values[node.inputs[0]];
    const auto selected =
        static_cast<std::size_t>(axis(scatter->axis, base.rank()));
    const auto inner = extent_product(base.dimensions().subspan(selected + 1));
    const auto outer = extent_product(base.dimensions().first(selected));
    const auto extent = base.dimensions()[selected];
    const auto count =
        static_cast<std::int64_t>(elements(recipe.values[node.inputs[1]]));
    if (count == 0) {
      result = ggml_dup(context, input(0));
    } else {
      auto *base_rows = indexed_rows(context, fp(0), inner, extent, outer);
      auto *update_rows = indexed_rows(context, fp(2), inner, count, outer);
      auto *indices = ggml_reshape_1d(context, input(1), count);
      auto *updates =
          ggml_get_rows_back(context, update_rows, indices, base_rows);
      result = restore_indexed_rows(
          context, ggml_add(context, base_rows, updates), inner, extent, outer);
    }
  } else if (const auto *matmul = node.operation.get_if<MatMul>()) {
    // Public RHS is [...,K,N]; ggml mul_mat consumes [K,N] as ne0=K,
    // ne1=N. Preserve up to two leading batch axes; ggml then broadcasts a
    // shared RHS across the corresponding activation batches.
    auto *left = native_shape(context, invocation.values[node.inputs[0]],
                              recipe.values[node.inputs[0]]);
    auto *right = ggml_transpose(context, invocation.values[node.inputs[1]]);
    if (!ggml_is_contiguous(right))
      right = ggml_cont(context, right);
    result = ggml_mul_mat(context, right, left);
    if (!ggml_prec_set_acc(result, GGML_PREC_F32))
      return nullptr;
    (void)matmul;
  } else if (const auto *rope = node.operation.get_if<RoPE>()) {
    auto *positions = invocation.values[node.inputs[1]];
    auto *frequencies = invocation.values[node.inputs[2]];
    auto *one = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);
    one->data = &program.one;
    auto *factors = ggml_div(context, ggml_repeat(context, one, frequencies),
                             frequencies);
    const int mode = rope->layout == RopeLayout::kInterleaved
                         ? GGML_ROPE_TYPE_NORMAL
                         : GGML_ROPE_TYPE_NEOX;
    result = ggml_rope_ext(
        context, invocation.values[node.inputs[0]], positions, factors,
        static_cast<int>(rope->rotary_dimensions), mode, 0, 1.0F, 1.0F,
        0.0F, 1.0F, 0.0F, 0.0F);
  } else if (node.operation.get_if<Hadamard>()) {
    const auto &values = program.hadamards[node_index];
    if (values.empty())
      return nullptr;
    const auto width = columns(output);
    auto *transform =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, width, width);
    transform->data = const_cast<float *>(values.data());
    result = ggml_mul_mat(context, transform, as_float(context, input(0)));
    ggml_mul_mat_set_hint(result, GGML_HINT_SRC0_IS_HADAMARD);
    if (!ggml_prec_set_acc(result, GGML_PREC_F32))
      return nullptr;
  } else if (const auto *attention =
                 node.operation.get_if<ScaledDotProductAttention>()) {
    auto *query = invocation.values[node.inputs[0]];
    auto *key = invocation.values[node.inputs[1]];
    auto *value = invocation.values[node.inputs[2]];
    const auto type = query->type;
    query = ggml_permute(context, as_float(context, query), 0, 2, 1, 3);
    key = ggml_permute(context, key, 0, 2, 1, 3);
    value = ggml_permute(context, value, 0, 2, 1, 3);
    if (type == GGML_TYPE_F16) {
      const auto &mask_values = program.attention_masks[node_index];
      if (mask_values.empty())
        return nullptr;
      auto *mask = ggml_new_tensor_2d(context, GGML_TYPE_F16, key->ne[1],
                                      query->ne[1]);
      mask->data = const_cast<ggml_fp16_t *>(mask_values.data());
      result = ggml_flash_attn_ext(context, query, key, value, mask,
                                   attention->scale, 0.0F, 0.0F);
      if (!ggml_prec_set_acc(result, GGML_PREC_F32))
        return nullptr;
    } else {
      key = as_float(context, key);
      value = as_float(context, value);
      auto *scores = ggml_mul_mat(context, key, query);
      if (!ggml_prec_set_acc(scores, GGML_PREC_F32))
        return nullptr;
      scores = ggml_scale(context, scores, attention->scale);
      scores = ggml_diag_mask_inf(
          context, scores, static_cast<int>(key->ne[1] - query->ne[1]));
      auto *probabilities = ggml_soft_max(context, scores);
      value = ggml_cont(context, ggml_transpose(context, value));
      result = ggml_mul_mat(context, value, probabilities);
      if (!ggml_prec_set_acc(result, GGML_PREC_F32))
        return nullptr;
      result = ggml_permute(context, result, 0, 2, 1, 3);
    }
  }
  return result ? as_output(context, result, output) : nullptr;
}

void destroy_invocation(void *value, mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<Invocation *>(value);
  if (!invocation)
    return;
  invocation->~Invocation();
  allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
}
Status create_invocation(void *value, mem::AllocatorRef allocator,
                         mem::BufferView, void **output) noexcept {
  *output = nullptr;
  auto &program = *static_cast<Program *>(value);
  const auto &recipe = program.recipe;
  void *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (!storage)
    return allocation_failure();
  auto *invocation = ::new (storage) Invocation;
  invocation->allocator = allocator;
  *output = invocation; // The caller owns partial-state cleanup on every error.
  try {
    invocation->input_maps.resize(recipe.inputs.size());
    invocation->output_maps.resize(recipe.outputs.size());
    if (std::ranges::all_of(recipe.outputs, [&](auto id) {
          return elements(recipe.values[id]) == 0;
        }))
      return Status::Ok();
    if (recipe.nodes.size() > (SIZE_MAX - 8) / 32 ||
        recipe.values.size() > (SIZE_MAX - 8 - recipe.nodes.size() * 32) / 4)
      return allocation_failure();
    const auto capacity =
        recipe.nodes.size() * 32 + recipe.values.size() * 4 + 8;
    auto arena =
        ggml_runtime::MetadataArena::for_graph(capacity, capacity, allocator);
    if (!arena)
      return std::move(arena).error();
    invocation->arena = std::move(*arena);
    auto *context = invocation->arena.get();
    invocation->graph = ggml_new_graph_custom(context, capacity, false);
    invocation->values.resize(recipe.values.size(), nullptr);
    for (const auto id : recipe.inputs) {
      const auto &description = recipe.values[id];
      ggml_tensor *tensor = nullptr;
      if ((description.is_dense() && description.rank() <= GGML_MAX_DIMS) ||
          elements(description) == 0) {
        std::array<std::int64_t, GGML_MAX_DIMS> dimensions{1, 1, 1, 1};
        for (std::size_t i = 0; i < description.rank(); ++i)
          dimensions[i] = (std::max)(
              description.dimensions()[description.rank() - i - 1],
              std::int64_t{1});
        tensor = ggml_new_tensor(context, native_type(description.data_type()),
                                 GGML_MAX_DIMS, dimensions.data());
      } else if (description.is_dense()) {
        tensor = ggml_new_tensor_2d(
            context, native_type(description.data_type()),
            (std::max)(columns(description), std::int64_t{1}),
            (std::max)(rows(description), std::int64_t{1}));
      } else {
        std::array<std::int64_t, GGML_MAX_DIMS> dimensions{1, 1, 1, 1};
        for (std::size_t i = 0; i < description.rank(); ++i)
          dimensions[i] = description.dimensions()[description.rank() - i - 1];
        tensor = ggml_new_tensor(context, native_type(description.data_type()),
                                 GGML_MAX_DIMS, dimensions.data());
        for (std::size_t i = 0; i < description.rank(); ++i)
          tensor->nb[i] = static_cast<std::size_t>(
              description.byte_strides()[description.rank() - i - 1]);
      }
      invocation->values[id] = tensor;
      invocation->input_roots.push_back(tensor);
    }
    for (std::size_t node_index = 0; node_index < recipe.nodes.size();
         ++node_index) {
      const auto &node = recipe.nodes[node_index];
      auto *result = lower(program, *invocation, node, node_index);
      if (!result)
        return Status::Internal(
            "ggml CPU lowering differs from its admitted schema");
      invocation->values[node.output] = result;
      ggml_build_forward_expand(invocation->graph, result);
    }
    for (const auto id : recipe.outputs) {
      auto *root = invocation->values[id];
      ggml_set_output(root);
      while (root->view_src)
        root = root->view_src;
      ggml_set_output(root);
      if (!ggml_is_contiguous(root) ||
          ggml_nbytes(root) != recipe.values[id].required_bytes().value() ||
          std::ranges::find(invocation->input_roots, root) !=
              invocation->input_roots.end() ||
          std::ranges::find(invocation->output_roots, root) !=
              invocation->output_roots.end())
        return Status::Unimplemented(
            "ggml CPU output cannot bind this native backing");
      invocation->output_roots.push_back(root);
    }
    std::vector<std::pair<ggml_tensor *, std::size_t>> allocations;
    for (auto *tensor = ggml_get_first_tensor(context); tensor;
         tensor = ggml_get_next_tensor(context, tensor)) {
      if (tensor->view_src) {
        invocation->views.push_back(tensor);
        continue;
      }
      if (tensor->data ||
          std::ranges::find(invocation->input_roots, tensor) !=
              invocation->input_roots.end() ||
          std::ranges::find(invocation->output_roots, tensor) !=
              invocation->output_roots.end())
        continue;
      const auto bytes = ggml_nbytes(tensor);
      const auto padding =
          (GGML_MEM_ALIGN - invocation->storage_bytes % GGML_MEM_ALIGN) %
          GGML_MEM_ALIGN;
      if (padding > SIZE_MAX - invocation->storage_bytes ||
          bytes > SIZE_MAX - invocation->storage_bytes - padding)
        return allocation_failure();
      invocation->storage_bytes += padding;
      allocations.emplace_back(tensor, invocation->storage_bytes);
      invocation->storage_bytes += bytes;
    }
    if (invocation->storage_bytes) {
      invocation->storage =
          allocator.allocate(invocation->storage_bytes, GGML_MEM_ALIGN);
      if (!invocation->storage)
        return allocation_failure();
      for (const auto &[tensor, offset] : allocations)
        tensor->data = static_cast<std::byte *>(invocation->storage) + offset;
    }
    auto execution = ggml_runtime::CpuExecution::prepare(
        invocation->graph, static_cast<int>(program.options.native_threads),
        allocator);
    if (!execution)
      return std::move(execution).error();
    invocation->execution = std::move(*execution);
    return Status::Ok();
  } catch (...) {
    return exception_status();
  }
}

void preserve(Status &first, Status next) noexcept {
  if (first.ok() && !next.ok())
    first = std::move(next);
}
Status close_mappings(Invocation &invocation, Status status,
                      std::span<const TensorView> outputs) noexcept {
  for (std::size_t i = 0; i < invocation.output_maps.size(); ++i) {
    auto &mapping = invocation.output_maps[i];
    if (mapping && status.ok() &&
        !outputs[i].storage().resource().properties().has(
            mem::MemoryProperty::kHostDeviceCoherent))
      preserve(status, mapping.flush());
    preserve(status, mapping.close());
  }
  for (auto &mapping : invocation.input_maps)
    preserve(status, mapping.close());
  return status;
}
Status run(Program &program, Invocation &invocation,
           std::span<const ConstTensorView> inputs,
           std::span<const TensorView> outputs) noexcept {
  if (!invocation.graph)
    return Status::Ok();
  const auto finish = [&](Status status) {
    return close_mappings(invocation, std::move(status), outputs);
  };
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto storage = inputs[i].storage();
    if (!storage.resource().properties().has(
            mem::MemoryProperty::kHostDeviceCoherent)) {
      auto status = storage.invalidate();
      if (!status.ok())
        return finish(std::move(status));
    }
    auto mapping = storage.map_host();
    if (!mapping)
      return finish(std::move(mapping).error());
    invocation.input_maps[i] = std::move(*mapping);
    invocation.input_roots[i]->data =
        const_cast<std::byte *>(invocation.input_maps[i].bytes().data());
  }
  // Indexed nodes are leaf boundaries. Validate current payload before the
  // original kernel can assert, and before mapping or modifying any output.
  for (const auto &check : program.index_checks) {
    const auto *bytes = invocation.input_maps[check.input].bytes().data();
    const auto &description =
        program.recipe.values[program.recipe.inputs[check.input]];
    for (std::size_t i = 0; i < check.count; ++i) {
      std::size_t offset = i * sizeof(std::int32_t);
      if (!description.is_dense()) {
        offset = 0;
        auto remaining = i;
        for (std::size_t dimension = description.rank(); dimension-- > 0;) {
          const auto extent =
              static_cast<std::size_t>(description.dimensions()[dimension]);
          offset +=
              (remaining % extent) *
              static_cast<std::size_t>(description.byte_strides()[dimension]);
          remaining /= extent;
        }
      }
      std::int32_t index = 0;
      std::memcpy(&index, bytes + offset, sizeof(index));
      if (index < 0 || index >= check.limit)
        return finish(
            Status::OutOfRange("ggml CPU Gather index is outside its source"));
    }
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto mapping = outputs[i].storage().map_host(mem::HostAccess::kWrite);
    if (!mapping)
      return finish(std::move(mapping).error());
    invocation.output_maps[i] = std::move(*mapping);
    auto bytes = invocation.output_maps[i].writable_bytes();
    if (!bytes)
      return finish(std::move(bytes).error());
    invocation.output_roots[i]->data = bytes->data();
  }
  invocation.bind_views();
  struct Call final {
    Invocation *invocation;
    Status terminal = Status::Ok();
  } call{&invocation};
  program.options.executor.parallel_for(
      1,
      [](void *value, std::size_t, std::size_t) noexcept {
        auto &work = *static_cast<Call *>(value);
        work.terminal = work.invocation->execution.run();
      },
      &call);
  return finish(std::move(call.terminal));
}
detail::EnqueueResult submit(void *value, void *slot, Queue &,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  auto &program = *static_cast<Program *>(value);
  if (inputs.size() != program.recipe.inputs.size() ||
      outputs.size() != program.recipe.outputs.size())
    return detail::EnqueueResult::rejected(
        Status::InvalidArgument("ggml CPU binding counts differ"));
  return detail::EnqueueResult::accepted_command(
      run(program, *static_cast<Invocation *>(slot), inputs, outputs));
}
void destroy_program(void *value, mem::AllocatorRef allocator) noexcept {
  auto *program = static_cast<Program *>(value);
  if (program) {
    program->~Program();
    allocator.deallocate(program, sizeof(Program), alignof(Program));
  }
}
AliasSupport alias_support(void *, std::size_t, std::size_t) noexcept {
  return {};
}
const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared = &destroy_program,
    .alias_support = &alias_support,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};

Status create_context(const Provider &provider, Device device, void *,
                      mem::AllocatorRef allocator, void **output) noexcept {
  *output = nullptr;
  if (device.backend_id() != cpu::kBackendId)
    return Status::Unimplemented("ggml CPU requires the CPU backend");
  const auto configuration =
      detail::ProviderAccess::payload<Configuration>(provider);
  if (configuration.options.native_threads == 0 ||
      configuration.options.native_threads > GGML_MAX_N_THREADS)
    return Status::InvalidArgument(
        "ggml CPU native_threads is outside the supported range");
  void *storage =
      allocator.allocate(sizeof(Configuration), alignof(Configuration));
  if (!storage)
    return allocation_failure();
  *output = ::new (storage) Configuration(configuration);
  return Status::Ok();
}
void destroy_context(void *value, mem::AllocatorRef allocator) noexcept {
  if (value) {
    static_cast<Configuration *>(value)->~Configuration();
    allocator.deallocate(value, sizeof(Configuration), alignof(Configuration));
  }
}
Status propose(void *context, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *output) noexcept {
  *output = {};
  const bool standard = standard_index::supports(
      request.operation, request.inputs, request.outputs);
  if (static_cast<Configuration *>(context)->extension ||
      !request.shape.exact() ||
      (!standard && !internal::supports(context, request.operation,
                                        request.inputs, request.outputs)))
    return Status::Unimplemented("ggml CPU operation domain is not admitted");
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = {.required_properties = mem::MemoryProperty::kHostAccessible,
                 .minimum_alignment = request.inputs[i].required_alignment()};
  for (std::size_t i = 0; i < outputs.size(); ++i)
    outputs[i] = {.required_properties = mem::MemoryProperty::kHostAccessible,
                  .minimum_alignment = request.outputs[i].required_alignment()};
  *output = {.route = standard ? standard_index::kRoute : 1,
             .runtime = standard ? standard_index::runtime()
                                 : internal::runtime(context),
             .workspace_resource = {.required_properties =
                                        mem::MemoryProperty::kHostAccessible},
             .cost = {.workspace = WorkspaceRequirement{},
                      .persistent_bytes = std::nullopt,
                      .copy_bytes = std::nullopt,
                      .repack_bytes = 0,
                      .synchronization_count = 0,
                      .dispatch_count = 1},
             .profile_key =
                 ProviderRouteKey{standard ? standard_index::kRoute : 1}};
  if (standard)
    output->cost = standard_index::costs();
  else if (request.operation.get_if<MatMul>() != nullptr &&
           !native_matmul_rhs(request.inputs[1])) {
    auto bytes = request.inputs[1].required_bytes();
    if (!bytes)
      return std::move(bytes).error();
    output->cost.repack_bytes = *bytes;
  }
  return Status::Ok();
}
ProviderPrepareOutcome prepare(void *context,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  if (request.route == standard_index::kRoute)
    return standard_index::prepare(
        static_cast<Configuration *>(context)->options, request, allocator,
        output);
  if (request.route != 1 || !request.shape.exact() ||
      !internal::supports(context, request.operation, request.inputs,
                          request.outputs))
    return ProviderPrepareOutcome::unsupported(
        Status::Unimplemented("ggml CPU prepare domain differs"));
  try {
    internal::Recipe recipe;
    recipe.values.assign(request.inputs.begin(), request.inputs.end());
    recipe.values.push_back(request.outputs[0]);
    internal::Node node{request.operation,
                        {},
                        static_cast<std::uint32_t>(request.inputs.size())};
    for (std::uint32_t i = 0; i < request.inputs.size(); ++i) {
      recipe.inputs.push_back(i);
      node.inputs.push_back(i);
    }
    recipe.outputs.push_back(node.output);
    recipe.nodes.push_back(std::move(node));
    return internal::compile(context, std::move(recipe), allocator, output);
  } catch (...) {
    return ProviderPrepareOutcome::error(exception_status());
  }
}
Status drain(void *) noexcept { return Status::Ok(); }
const void *extension(void *value, ProviderExtensionId id) noexcept {
  const auto *extension = static_cast<Configuration *>(value)->extension;
  return extension && extension->id == id ? extension->table : nullptr;
}
const detail::ProviderVTable kProvider{
    .identity = {.id = kProviderId,
                 .name = "ggml-cpu",
                 .version = "456172ec+hyle-resource-v1"},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose,
    .prepare = &prepare,
    .drain_context = &drain,
    .query_extension = &extension};
} // namespace

namespace internal {
ProviderRuntimeProperties runtime(void *value) noexcept {
  return {.mode = ProviderRuntimeMode::kSharedQueue,
          .workspace_owner = ProviderWorkspaceOwner::kProvider,
          .replay_class = ProviderReplayClass::kPrepared,
          .blocks_host = true,
          .uses_private_threads =
              static_cast<Configuration *>(value)->options.native_threads > 1,
          .supports_concurrent_invocations = true,
          .capture_compatible = false};
}
bool supports(void *context, Operation operation,
              std::span<const TensorDesc> inputs,
              std::span<const TensorDesc> outputs) noexcept {
  if (!context || inputs.empty() || outputs.size() != 1 ||
      !outputs[0].is_dense() || !representable(outputs[0]))
    return false;
  for (const auto &description : inputs)
    if (!representable(description))
      return false;
  const auto inferred = infer_single_output(operation, inputs);
  if (!inferred || *inferred != outputs[0])
    return false;
  const auto &output = outputs[0];
  const auto nonempty = std::ranges::all_of(
      inputs, [](const auto &d) { return elements(d) != 0; });
  if (!nonempty && elements(output) != 0 && !operation.get_if<Concat>() &&
      !operation.get_if<Reduce>() && !operation.get_if<Scatter>())
    return false;
  if (operation.get_if<Copy>() || operation.get_if<Concat>())
    return true;
  if (operation.get_if<Cast>())
    return (floating(inputs[0].data_type()) && floating(output.data_type())) ||
           inputs[0].data_type() == output.data_type();
  if (!floating(output.data_type()) && !operation.get_if<Gather>())
    return false;
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
  if (operation.get_if<Gather>())
    return inputs[1].data_type() == DataType::kInt32;
  if (const auto *scatter = operation.get_if<Scatter>())
    return inputs.size() == 3 && scatter->kind == ScatterKind::kAdd &&
           inputs[1].data_type() == DataType::kInt32 &&
           floating(inputs[0].data_type()) &&
           inputs[2].data_type() == inputs[0].data_type() &&
           output.data_type() == inputs[0].data_type();
  if (const auto *matmul = operation.get_if<MatMul>()) {
    if (!nonempty || elements(output) == 0 || inputs.size() != 2 ||
        inputs[0].rank() < 2 || inputs[0].rank() > 4 ||
        inputs[1].rank() < 2 || inputs[1].rank() > 4 ||
        (inputs[0].rank() > 2 && !inputs[0].is_dense()) ||
        (inputs[1].rank() > 2 && !inputs[1].is_dense() &&
         !native_matmul_rhs(inputs[1])) ||
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
  if (const auto *attention =
          operation.get_if<ScaledDotProductAttention>()) {
    if (inputs.size() != 3 || !attention->causal ||
        attention->layout != AttentionLayout::kBatchSequenceHeadDepth ||
        attention->mask != MaskKind::kNone ||
        attention->accumulation_type != DataType::kFloat32 ||
        !std::isfinite(attention->scale))
      return false;
    const auto type = inputs[0].data_type();
    if (type != DataType::kFloat16 && type != DataType::kBFloat16 &&
        type != DataType::kFloat32)
      return false;
    for (const auto &description : inputs)
      if (description.rank() != 4 || !description.is_dense() ||
          description.data_type() != type)
        return false;
    const auto q = inputs[0].dimensions();
    const auto k = inputs[1].dimensions();
    return q[0] > 0 && q[1] > 0 && k[1] >= q[1] && q[2] == 8 &&
           k[2] == 2 && q[3] == 256 && k[3] == 256 && q[0] == k[0] &&
           inputs[1] == inputs[2] && output == inputs[0];
  }
  return false;
}
bool region_operation(Operation operation) noexcept {
  return operation.get_if<Copy>() || operation.get_if<Cast>() ||
         operation.get_if<Unary>() || operation.get_if<Binary>() ||
         operation.get_if<RmsNorm>() || operation.get_if<LayerNorm>() ||
         operation.get_if<Reduce>() || operation.get_if<Concat>() ||
         operation.get_if<Scatter>() || operation.get_if<MatMul>() ||
         operation.get_if<RoPE>() || operation.get_if<Hadamard>() ||
         operation.get_if<ScaledDotProductAttention>();
}
ProviderPrepareOutcome compile(void *context, Recipe recipe,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  if (!context || recipe.nodes.empty() || recipe.outputs.empty())
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("ggml CPU recipe is empty"));
  try {
    std::vector<bool> available(recipe.values.size(), false);
    std::vector<Program::IndexCheck> index_checks;
    for (const auto id : recipe.inputs) {
      if (id >= available.size() || available[id])
        return ProviderPrepareOutcome::error(
            Status::InvalidArgument("ggml CPU input identity is invalid"));
      available[id] = true;
    }
    for (const auto &node : recipe.nodes) {
      std::vector<TensorDesc> inputs;
      for (const auto id : node.inputs) {
        if (id >= available.size() || !available[id])
          return ProviderPrepareOutcome::error(
              Status::InvalidArgument("ggml CPU recipe is not topological"));
        inputs.push_back(recipe.values[id]);
      }
      if (node.output >= available.size() || available[node.output] ||
          !supports(context, node.operation, inputs,
                    std::span(&recipe.values[node.output], 1)))
        return ProviderPrepareOutcome::unsupported(Status::Unimplemented(
            "ggml CPU recipe contains an unqualified node"));
      if (const auto *gather = node.operation.get_if<Gather>()) {
        const auto index = std::ranges::find(recipe.inputs, node.inputs[1]);
        if (index == recipe.inputs.end())
          return ProviderPrepareOutcome::unsupported(Status::Unimplemented(
              "ggml CPU dynamic indices require a checked boundary"));
        index_checks.push_back(
            {static_cast<std::uint32_t>(index - recipe.inputs.begin()),
             elements(recipe.values[node.inputs[1]]),
             inputs[0].dimensions()[static_cast<std::size_t>(
                 axis(gather->axis, inputs[0].rank()))]});
      } else if (const auto *scatter = node.operation.get_if<Scatter>()) {
        const auto index = std::ranges::find(recipe.inputs, node.inputs[1]);
        if (index == recipe.inputs.end())
          return ProviderPrepareOutcome::unsupported(Status::Unimplemented(
              "ggml CPU dynamic Scatter indices require a checked boundary"));
        index_checks.push_back(
            {static_cast<std::uint32_t>(index - recipe.inputs.begin()),
             elements(recipe.values[node.inputs[1]]),
             inputs[0].dimensions()[static_cast<std::size_t>(
                 axis(scatter->axis, inputs[0].rank()))]});
      }
      available[node.output] = true;
    }
    for (const auto id : recipe.outputs)
      if (id >= available.size() || !available[id])
        return ProviderPrepareOutcome::error(
            Status::InvalidArgument("ggml CPU output identity is invalid"));
    std::vector<std::vector<float>> hadamards(recipe.nodes.size());
    std::vector<std::vector<ggml_fp16_t>> attention_masks(recipe.nodes.size());
    for (std::size_t node_index = 0; node_index < recipe.nodes.size();
         ++node_index) {
      const auto &node = recipe.nodes[node_index];
      if (node.operation.get_if<Hadamard>()) {
        const auto width = static_cast<std::size_t>(
            recipe.values[node.output].dimensions().back());
        if (width > SIZE_MAX / width ||
            width * width > SIZE_MAX / sizeof(float))
          return ProviderPrepareOutcome::error(
              Status::OutOfRange("Hadamard matrix size overflows"));
        auto &matrix = hadamards[node_index];
        matrix.resize(width * width);
        const auto scale = 1.0F / std::sqrt(static_cast<float>(width));
        for (std::size_t row = 0; row < width; ++row)
          for (std::size_t column = 0; column < width; ++column)
            matrix[row * width + column] =
                std::popcount(row & column) % 2 == 0 ? scale : -scale;
      }
      if (node.operation.get_if<ScaledDotProductAttention>() &&
          recipe.values[node.inputs[0]].data_type() == DataType::kFloat16) {
        const auto query_length = static_cast<std::size_t>(
            recipe.values[node.inputs[0]].dimensions()[1]);
        const auto key_length = static_cast<std::size_t>(
            recipe.values[node.inputs[1]].dimensions()[1]);
        if (query_length > SIZE_MAX / key_length)
          return ProviderPrepareOutcome::error(
              Status::OutOfRange("attention mask size overflows"));
        auto &mask = attention_masks[node_index];
        mask.resize(query_length * key_length);
        const auto past = key_length - query_length;
        for (std::size_t query = 0; query < query_length; ++query)
          for (std::size_t key = 0; key < key_length; ++key)
            mask[query * key_length + key] = ggml_fp32_to_fp16(
                key <= past + query ? 0.0F : -INFINITY);
      }
    }
    void *storage = allocator.allocate(sizeof(Program), alignof(Program));
    if (!storage)
      return ProviderPrepareOutcome::resource_declined(allocation_failure());
    auto *program = ::new (storage) Program{
        static_cast<Configuration *>(context)->options, std::move(recipe),
        std::move(hadamards), std::move(attention_masks),
        std::move(index_checks)};
    *output = {.vtable = &kPrepared,
               .state = program,
               .workspace = {},
               .runtime = runtime(context),
               .persistent_bytes = std::nullopt,
               .invocation_private_bytes = std::nullopt};
    return ProviderPrepareOutcome::prepared();
  } catch (...) {
    return ProviderPrepareOutcome::error(exception_status());
  }
}
Provider with_extension(Options options, const Extension *extension) noexcept {
  return detail::ProviderAccess::make(&kProvider,
                                      Configuration{options, extension});
}
} // namespace internal
Provider provider(Options options) noexcept {
  return internal::with_extension(options, nullptr);
}
} // namespace hyle::compute::ggml_cpu
