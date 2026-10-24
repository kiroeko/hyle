#include "tensor_program.h"
#include "gated_delta_scale.h"

#include <algorithm>
#include <array>
#include <climits>

#include "numerical_contract.h"
#include "sink_mask.h"
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
  case GgufQuantizedType::kNvfp4:
    return GGML_TYPE_NVFP4;
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
          *type == GgufQuantizedType::kQ8_0 ||
          *type == GgufQuantizedType::kIq2Xxs ||
          *type == GgufQuantizedType::kIq2Xs ||
          *type == GgufQuantizedType::kIq2S ||
          *type == GgufQuantizedType::kIq1S ||
          *type == GgufQuantizedType::kIq1M ||
          *type == GgufQuantizedType::kIq3Xxs ||
          *type == GgufQuantizedType::kIq3S ||
          *type == GgufQuantizedType::kIq4Nl ||
          *type == GgufQuantizedType::kIq4Xs ||
          *type == GgufQuantizedType::kMxfp4 ||
          *type == GgufQuantizedType::kNvfp4) &&
         description.data_type() == DataType::kFloat32 &&
         description.rank() == 2 && description.block_axis() == 1 &&
         // These 32-value formats still use a 256-value CUDA get_rows loop.
         ((*type != GgufQuantizedType::kIq4Nl &&
           *type != GgufQuantizedType::kMxfp4) ||
          description.dimensions()[1] % 256 == 0);
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
bool plain_matmul_rhs(const TensorDesc &description) noexcept {
  // Admit an ordinary matrix or exactly the last-two-axis view of a dense
  // weight bank. Lowering transposes that view back to native contiguous
  // rows; padded, sliced and other permuted layouts remain outside this path.
  return description.is_plain() &&
         (description.is_dense() || native_matmul_rhs(description));
}
bool representable(const TensorDesc &description,
                   bool integer_storage = false,
                   bool native_unexpanded_source = false) noexcept {
  const auto count = description.element_count();
  const auto bytes = description.required_bytes();
  // Native kernels use int element counts and unsigned 32-bit byte strides.
  // Leave room for rounded launch sizes and FP32 conversion intermediates.
  const auto element_limit = native_unexpanded_source ? std::size_t(INT32_MAX - 1024)
      : std::size_t(INT32_MAX - 1024) / sizeof(float);
  // Qualified selected-row decode and direct BF16 matvec never materialize
  // the full readonly source as F32. Keep original element/uint32 byte bounds
  // for these proven paths, beyond the generic FP32 scratch bound.
  const auto byte_limit = native_unexpanded_source ? std::size_t(UINT32_MAX)
                                                  : std::size_t(INT32_MAX);
  if (!description.valid() || !count || !bytes || *count == 0 ||
      *count > element_limit || *bytes > byte_limit ||
      !(floating(description.data_type()) ||
        (integer_storage && description.is_dense() &&
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
// A single native batched matvec reads original floating W and F32 A directly.
// Mixed F16/BF16 activations are widened exactly before this native carrier;
// only the small activation is converted, never the full weight allocation.
// [B,M,K] is contiguous [K,1,M,B] in ggml; W is [K,N,1,B] and the
// output [N,1,M,B] is exactly row-major [B,M,N]. Rank2 has B=1.
// No matrix/cublas fallback is
// permitted here because it can expand the complete half/bfloat bank to F32.
bool native_dense_linear(const TensorDesc &a, const TensorDesc &w,
                              const TensorDesc &output) noexcept {
  const auto rank=a.rank();
  if (!a.valid() || !w.valid() || !output.valid() || (rank != 2 && rank != 3) ||
      w.rank() != rank || output.rank() != rank ||
      !a.is_plain() || !w.is_plain() || !output.is_plain() ||
      !a.is_dense() || !w.is_dense() || !output.is_dense() ||
      !floating(a.data_type()) || !floating(w.data_type()) ||
      output.data_type() != DataType::kFloat32)
    return false;
  const auto ad=a.dimensions(),wd=w.dimensions(),od=output.dimensions();
  const auto batches=rank==3?ad[0]:1,rows=ad[rank-2],width=ad[rank-1],columns=wd[rank-2];
  // MMVF dispatch is grid(N,M,B), with paired source and F32 activation reads.
  // The fixed NVIDIA
  // selector also requires N<=512 when the original W has multiple batches.
  return batches>0 && batches<=65535 && rows>0 && rows<=65535 && width>0 && width%2==0 &&
      columns>0 && (batches==1 || columns<=512) && wd[rank-1]==width &&
      od[rank-2]==rows && od[rank-1]==columns &&
      (rank==2 || (wd[0]==batches && od[0]==batches)) &&
      representable(a) && representable(output) && representable(w,false,true);
}

// A canonical packed bank can retain its original storage when the actual
// native device selects MMVQ, or selects MMQ with at least one legal tile.
// Both paths quantize only A; a cuBLAS/full-W decode fallback is declined.
bool native_packed_small_batch_rhs(const TensorDesc &activation,
                                   const TensorDesc &weight,
                                   const TensorDesc &output,
                                   int native_device) noexcept {
  if (!activation.valid() || !activation.is_plain() || !activation.is_dense() ||
      activation.data_type() != DataType::kFloat32 || activation.rank() != 2 ||
      activation.dimensions()[0] < 1 || activation.dimensions()[0] > 4 ||
      !weight.valid() || !native_quantized(weight) || weight.rank() != 2 ||
      weight.data_type() != DataType::kFloat32 || weight.block_axis() != 0 ||
      !output.valid() || !output.is_plain() || !output.is_dense() ||
      output.data_type() != DataType::kFloat32 || output.rank() != 2 ||
      output.dimensions()[0] != activation.dimensions()[0])
    return false;
  const auto reduction = weight.dimensions()[0];
  const auto output_columns = weight.dimensions()[1];
  constexpr auto limit = std::int64_t{INT32_MAX - 1024};
  // K%512 avoids native MATRIX_ROW_PADDING expansion/clear on the readonly
  // bank, so bad_padding_clear cannot divert this F32 operation to cuBLAS.
  if (reduction <= 0 || reduction % 512 != 0 || reduction > limit ||
      output_columns <= 0 || output_columns > limit / reduction ||
      activation.dimensions()[1] != reduction ||
      output.dimensions()[1] != output_columns)
    return false;
  const auto bytes = weight.required_bytes();
  const auto block = ggml_blck_size(native_quantized_type(weight));
  const auto type = gguf_quantized_type(weight.storage_format());
  if (!bytes || *bytes > std::size_t{UINT32_MAX} || block <= 0 ||
      reduction % block != 0 || output_columns > limit / (reduction / block) ||
      !type)
    return false;
  // MMVQ indexes W through typed quantization-block pointers, not signed
  // byte offsets. Q8_0 scale overhead can put its original encoded extent
  // above INT32_MAX while all retained signed logical/block/stride bounds
  // still fit. Do not grant any ordinary input/output or F32 scratch exception.
  const auto canonical = gguf_weight_description(
      *type, DataType::kFloat32, reduction, output_columns);
  if (!canonical || *canonical != weight || !native_matmul_rhs(weight) ||
      !std::ranges::all_of(weight.byte_strides(), [](auto stride) {
        return stride > 0 && stride <= INT32_MAX;
      }))
    return false;
  try {
    return packed_matmul_route(native_device, native_quantized_type(weight),
        reduction, output_columns, activation.dimensions()[0]) != PackedMatmulRoute::kDeclined;
  } catch (...) {
    // A native cold-initialization failure cannot qualify an allocation.
    return false;
  }
}
bool grouped_activation(const TensorDesc &description) noexcept {
  if (!description.is_plain() || description.rank() != 3 ||
      description.data_type() != DataType::kFloat32)
    return false;
  if (description.is_dense())
    return true;
  const auto shape = description.dimensions();
  const auto stride = description.byte_strides();
  return stride[2] == sizeof(float) && stride[1] == 0 &&
         (shape[0] == 1 ||
          (stride[0] >= shape[2] * std::int64_t{sizeof(float)} &&
           stride[0] % (shape[2] * std::int64_t{sizeof(float)}) == 0));
}
// A broadcast slot is metadata, not S physical activation rows. This narrow
// port qualification keeps the original native bounds on its folded [K,1,T]
// carrier and exact backing; ordinary per-slot tensors retain the generic gate.
bool shared_grouped_activation(const TensorDesc &description) noexcept {
  const auto bytes = description.required_bytes();
  if (!description.valid() || !grouped_activation(description) || !bytes ||
      description.byte_strides()[1] != 0 || *bytes > INT32_MAX)
    return false;
  const auto shape = description.dimensions();
  constexpr auto limit = std::size_t{(INT32_MAX - 1024) / sizeof(float)};
  return shape[0] > 0 && shape[2] > 0 &&
         static_cast<std::size_t>(shape[2]) <= limit &&
         static_cast<std::size_t>(shape[0]) <= limit / shape[2] &&
         std::ranges::all_of(description.byte_strides(), [](auto stride) {
           return stride <= INT32_MAX && stride % sizeof(float) == 0;
         });
}

ggml_tensor *grouped_activation(ggml_context *context, ggml_tensor *source) {
  if (source->nb[1] != 0)
    return source;
  return ggml_view_3d(context, source, source->ne[0], 1, source->ne[2],
      std::size_t(source->ne[0]) * sizeof(float),
      source->ne[2] == 1 ? std::size_t(source->ne[0]) * sizeof(float) : source->nb[2], 0);
}
bool native_grouped_bank(const TensorDesc &description) noexcept {
  const auto type = gguf_quantized_type(description.storage_format());
  if (!type || !native_quantized(description) || description.rank() != 3 ||
      description.data_type() != DataType::kFloat32)
    return false;
  const auto shape = description.dimensions();
  const auto canonical = gguf_grouped_weight_description(
      *type, description.data_type(), shape[0], shape[1], shape[2]);
  // MMVQ/MMQ index the encoded bank in quantization blocks, not bytes or
  // decoded F32 elements. Preserve signed block offsets, launch headroom and
  // all existing per-expert/stride bounds; no full-bank decode is introduced.
  return canonical && *canonical == description && shape[0] <= INT32_MAX &&
         shape[1] > 0 && shape[2] > 0 &&
         shape[1] <= std::int64_t{(INT32_MAX - 1024) / sizeof(float)} / shape[2] &&
         *description.element_count() /
                 ggml_blck_size(native_quantized_type(description)) <=
             INT32_MAX - 1024 &&
         std::ranges::all_of(description.byte_strides(), [](auto stride) {
           return stride > 0 && stride <= INT32_MAX;
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
  const auto matches = [&](const ggml_tensor *tensor) {
    return ggml_is_contiguous(tensor) &&
           tensor->ne[0] == columns(description) &&
           tensor->ne[1] == rows(description) && tensor->ne[2] == 1 &&
           tensor->ne[3] == 1;
  };
  if (matches(value))
    return value;
  // as_output restores the public rank. Its immediately inverse flattening
  // must reuse the mathematical producer, preserving direct native edges.
  // Fold only this zero-offset reshape; never walk or bypass other operations.
  if (value->op == GGML_OP_RESHAPE && value->view_offs == 0 &&
      value->src[0] && value->type == value->src[0]->type &&
      ggml_is_contiguous(value) &&
      ggml_nelements(value) == ggml_nelements(value->src[0]) &&
      matches(value->src[0]))
    return value->src[0];
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
  if (std::equal(shape.begin(), shape.end(), value->ne))
    return value;
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

// binbcast's first input is read through its actual strides. A readonly
// scalar view supplies the output shape without a repeated tensor payload.
// Construct the original one-element view before expanding its metadata;
// ggml's dense view constructors would otherwise overstate its byte range.
ggml_tensor *scalar_broadcast_view(
    ggml_context *context, ggml_tensor *scalar,
    std::span<const std::int64_t, GGML_MAX_DIMS> shape) {
  auto *view = ggml_view_1d(context, scalar, 1, 0);
  for (std::size_t i = 0; i < GGML_MAX_DIMS; ++i) {
    view->ne[i] = shape[i];
    view->nb[i] = 0;
  }
  return view;
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

std::size_t TensorBuildFrame::ShapeHash::operator()(
    const ShapeKey &key) const noexcept {
  auto hash = std::hash<ggml_tensor *>{}(key.source);
  const auto combine = [&](std::size_t part) {
    hash ^= part + std::size_t{0x9e3779b9} + (hash << 6) + (hash >> 2);
  };
  combine(key.rank);
  for (const auto extent : key.shape)
    combine(std::hash<std::int64_t>{}(extent));
  return hash;
}

TensorBuildFrame::TensorBuildFrame(const Recipe &recipe,
                                  mem::AllocatorRef allocator)
    : views_(0, ShapeHash{}, std::equal_to<ShapeKey>{},
             mem::StdAllocator<ViewEntry>{allocator}),
      grouped_(recipe.values.size(), nullptr,
               mem::StdAllocator<ggml_tensor *>{allocator}),
      widen_sources_(recipe.values.size(), WidenSource{},
                     mem::StdAllocator<WidenSource>{allocator}) {
  // These are recipe-local use facts, not addresses or retained permissions.
  // Count input occurrences (including repeated ports), saturating at two.
  for (const auto &node : recipe.nodes)
    for (const auto id : node.inputs) {
      auto &readers = widen_sources_[id].readers;
      if (readers < 2) ++readers;
    }
  for (const auto id : recipe.outputs)
    widen_sources_[id].published = true;
  for (const auto &node : recipe.nodes) {
    if (!node.operation.get_if<Cast>() || node.inputs.size() != 1 ||
        node.outputs.size() != 1)
      continue;
    const auto source = node.inputs[0], output = node.outputs[0];
    auto &fact = widen_sources_[output];
    if (fact.readers == 1 && !fact.published &&
        recipe.values[source].data_type() == DataType::kBFloat16 &&
        recipe.values[output].data_type() == DataType::kFloat32)
      fact.source = source;
  }
}

ggml_tensor *TensorBuildFrame::private_bfloat_widen_source(
    std::uint32_t value, std::span<ggml_tensor *const> tensors) const noexcept {
  const auto source = widen_sources_[value].source;
  return source == UINT32_MAX ? nullptr : tensors[source];
}

ggml_tensor *TensorBuildFrame::matrix_form(ggml_context *context,
                                          ggml_tensor *source,
                                          const TensorDesc &description) {
  const ShapeKey key{source, {columns(description), rows(description), 1}, 2};
  if (const auto found = views_.find(key); found != views_.end())
    return found->second;
  auto *result = matrix(context, source, description);
  views_.emplace(key, result);
  return result;
}

bool tensor_operation_supports(Operation operation,
                               std::span<const TensorDesc> inputs,
                               std::span<const TensorDesc> outputs,
                               int native_device) noexcept {
  // Concat and GatherElements move I32 storage without numerical conversion.
  const bool integer_storage =
      (operation.get_if<Concat>() || operation.custom_as<model_ops::GatherElements>()) &&
      outputs.size() == 1 && outputs[0].data_type() == DataType::kInt32;
  if (inputs.empty() || outputs.empty() ||
      !std::ranges::all_of(outputs, [&](const auto &description) {
        return description.is_dense() && representable(description, integer_storage);
      }))
    return false;
  const bool direct_dense_rhs = operation.custom_as<model_ops::DenseLinearF32>() &&
      inputs.size() == 2 && outputs.size() == 1 &&
      (inputs[0].data_type() == DataType::kFloat32 || inputs[0].data_type() != inputs[1].data_type()) &&
      native_dense_linear(inputs[0], inputs[1], outputs[0]);
  const auto *packed_matmul = operation.get_if<MatMul>();
  const bool packed_small_batch_rhs =
      packed_matmul && packed_matmul->accumulation_type == DataType::kFloat32 &&
      inputs.size() == 2 && outputs.size() == 1 &&
      native_packed_small_batch_rhs(inputs[0], inputs[1], outputs[0],
                                    native_device);
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (!representable(inputs[i], integer_storage,
            i == 0 && operation.get_if<Gather>() && native_embedding_table(inputs[i])) &&
        !(i == 1 && (packed_small_batch_rhs || direct_dense_rhs)) &&
        !(i == 0 && operation.custom_as<model_ops::GroupedMatMul>() &&
          native_grouped_bank(inputs[i])) &&
        !(i == 1 && operation.custom_as<model_ops::GroupedMatMul>() &&
          shared_grouped_activation(inputs[i])) &&
        !(((operation.get_if<RoPE>() && i == 1) ||
           (operation.get_if<Gather>() && i == 1) ||
           (operation.custom_as<model_ops::GatherElements>() && i == 1) ||
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
    return ggml_runtime::sink_score_elements(inputs[0].dimensions(), inputs[1].dimensions()) <=
           (INT32_MAX - 1024) / sizeof(float);
  if (operation.custom_as<model_ops::Clamp>())
    return true; // Public inference has already validated the entire domain.
  if (operation.custom_as<model_ops::CausalConv1d>())
    return outputs.size() == 1 && inputs.size() == 2 &&
           inputs[0].data_type() == DataType::kFloat32 &&
           inputs[1].data_type() == DataType::kFloat32;
  if (const auto gather = operation.custom_as<model_ops::GatherElements>())
    return inputs.size() == 2 && outputs.size() == 1 &&
           inputs[0].rank() == 2 && inputs[1].rank() == 2 &&
           (gather->axis == 1 || gather->axis == -1);
  if (operation.custom_as<model_ops::GroupedMatMul>())
    return outputs.size() == 1 && inputs.size() == 3 &&
           (!native_quantized(inputs[0]) ||
            (inputs[0].rank() == 3 && inputs[0].block_axis() == 2)) &&
           inputs[0].is_dense() && grouped_activation(inputs[1]) &&
           inputs[2].is_dense() &&
           (inputs[0].data_type() == DataType::kFloat32 ||
            (inputs[0].is_plain() &&
             inputs[0].data_type() == DataType::kBFloat16)) &&
           inputs[1].data_type() == DataType::kFloat32 &&
           inputs[2].data_type() == DataType::kInt32 &&
           outputs[0].data_type() == DataType::kFloat32;
  if (operation.custom_as<model_ops::DenseLinearF32>())
    return outputs.size() == 1 && inputs.size() == 2 &&
           output.data_type() == DataType::kFloat32 &&
           (inputs[0].data_type() == DataType::kFloat32 || inputs[0].data_type() != inputs[1].data_type()
                ? direct_dense_rhs : inputs[0].rank() == 2);
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
    const auto *gdn = operation.get_if<GatedDeltaNet>();
    const auto scale = gdn ? gdn->output_scale
        : operation.custom_as<model_ops::RecurrentGatedDeltaNet>()->output_scale;
    if (!ggml_runtime::valid_gdn_output_scale(scale)) return false;
    if (ggml_hyle_cuda_gdn_output_scale_contract() !=
        ggml_runtime::kGdnOutputScaleContract) return false;
    if (outputs.size() != 2 || inputs.size() != (recurrent ? 8U : 6U) ||
        !std::ranges::all_of(inputs, [](const auto &description) {
          return description.is_dense() &&
                 description.data_type() == DataType::kFloat32;
        }))
      return false;
    const auto q = inputs[0].dimensions();
    const auto v = inputs[2].dimensions();
    const auto scores = *outputs[0].element_count();
    const auto states = *outputs[1].element_count();
    // The native launch uses grid.y=B and only instantiates these
    // square-state depths. Its supports_op alone does not check them.
    return q[0] <= 65535 &&
           (q[3] == 16 || q[3] == 32 || q[3] == 64 || q[3] == 128) &&
           v[2] % q[2] == 0 && q[3] == v[3] &&
           scores <= std::size_t{INT32_MAX} &&
           states <= std::size_t{INT32_MAX} - scores;
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
    case UnaryKind::kSilu:
      return output.data_type() == DataType::kFloat32 ||
             output.data_type() == DataType::kBFloat16;
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
    if (inputs.size() != 2)
      return false;
    const bool quantized_rhs = native_quantized(inputs[1]);
    if (inputs[0].rank() < 2 || inputs[0].rank() > 4 ||
        inputs[1].rank() < 2 || inputs[1].rank() > 4 ||
        !inputs[0].is_dense() ||
        (!quantized_rhs && !plain_matmul_rhs(inputs[1])) ||
        (quantized_rhs &&
         (inputs[1].rank() != 2 || inputs[1].block_axis() != 0 ||
          !native_matmul_rhs(inputs[1]) || native_quantized(inputs[0]))) ||
        matmul->accumulation_type != DataType::kFloat32 ||
        !floating(inputs[0].data_type()) ||
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
    return inputs.size() == 3 &&
           ((inputs[0].rank() == 4 &&
             (rope->sequence_axis == 1 || rope->sequence_axis == -3)) ||
            (inputs[0].rank() == 3 &&
             (rope->sequence_axis == 1 || rope->sequence_axis == -2))) &&
           std::ranges::all_of(inputs[0].byte_strides(),
                              [](auto stride) { return stride >= 0; }) &&
           (inputs[0].data_type() == DataType::kFloat16 ||
            inputs[0].data_type() == DataType::kBFloat16 ||
            inputs[0].data_type() == DataType::kFloat32) &&
           inputs[1].rank() <= 2 && inputs[1].is_dense() &&
           inputs[1].data_type() == DataType::kInt32 && inputs[2].rank() <= 2 &&
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
                       TensorBuildFrame &forms,
                       std::vector<ScalarConstant> &constants,
                       std::vector<DenseConstant> &dense_constants,
                       std::span<const float> hadamard,
                       std::span<const std::int32_t> snapshot_order) {
  const auto &output = recipe.values[node.outputs[0]];
  const auto input = [&](std::size_t index) {
    const auto id = node.inputs[index];
    return forms.matrix_form(context, values[id], recipe.values[id]);
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
    // Native typed elementwise kernels widen on load and narrow on store.
    // Rsqrt remains the original F32 multi-operation composition.
    auto *a = output.data_type() == DataType::kBFloat16 &&
                      unary->kind != UnaryKind::kRsqrt
                  ? input(0) : fp(0);
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
      result = ggml_div(context,
                        scalar_broadcast_view(context, one,
                            std::span<const std::int64_t, GGML_MAX_DIMS>(
                                a->ne, GGML_MAX_DIMS)),
                        ggml_sqrt(context, a));
      break;
    }
    case UnaryKind::kTanh:
      result = ggml_tanh(context, a);
      break;
    case UnaryKind::kSigmoid:
      result = ggml_sigmoid(context, a);
      break;
    case UnaryKind::kSilu:
      result = ggml_silu(context, a);
      result->op_params[kUnaryContractParam] = kStableSiluContract;
      break;
    default:
      break; // Stable tails and non-finite domains need another mature carrier.
    }
  } else if (const auto *binary = node.operation.get_if<Binary>()) {
    const auto operand = [&](std::size_t index) {
      return output.data_type() == DataType::kBFloat16 ? input(index) : fp(index);
    };
    auto *left = operand(0);
    const std::array<std::int64_t, GGML_MAX_DIMS> shape{
        columns(output), rows(output), 1, 1};
    // Keep other operations' existing fusion/layout domains. Division has
    // an original stride-aware carrier for this F32 scalar-left case.
    auto *a = binary->kind == BinaryKind::kDivide &&
                      left->type == GGML_TYPE_F32 && ggml_nelements(left) == 1
                  ? scalar_broadcast_view(context, left, shape)
                  : expand_broadcast(context, left,
                                     recipe.values[node.inputs[0]], output);
    auto *b =
        expand_broadcast(context, operand(1), recipe.values[node.inputs[1]], output);
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
    if (gate->inputs == GatedInputKind::kPackedHalves) {
      auto *a = input(0);
      if (a->type == GGML_TYPE_BF16)
        a = as_float(context, a);
      result = ggml_swiglu(context, a);
    } else {
      // Preserve the grouped rank-3 producer edges before requesting matrix
      // forms. Public readers retain their original shape and ownership.
      auto *raw_a = forms.grouped(node.inputs[0]);
      auto *raw_b = forms.grouped(node.inputs[1]);
      const auto grouped_order = [&](ggml_tensor *raw, std::uint32_t id) {
        return raw && raw->op == GGML_OP_MUL_MAT_ID &&
               raw->type == GGML_TYPE_F32 &&
               recipe.values[id].data_type() == DataType::kFloat32 &&
               ggml_is_contiguous(raw) && raw->ne[3] == 1 &&
               raw->ne[0] == columns(recipe.values[id]) &&
               static_cast<std::size_t>(ggml_nelements(raw)) ==
                   elements(recipe.values[id]);
      };
      if (grouped_order(raw_a, node.inputs[0]) &&
          grouped_order(raw_b, node.inputs[1]) &&
          ggml_are_same_shape(raw_a, raw_b)) {
        result = ggml_swiglu_split(context, raw_a, raw_b);
      } else {
        auto *a = input(0);
        if (a->type == GGML_TYPE_BF16)
          a = as_float(context, a);
        auto *b = input(1);
        if (b->type == GGML_TYPE_BF16)
          b = as_float(context, b);
        result = ggml_swiglu_split(context, a, b);
      }
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
    std::vector<ggml_tensor *> parts;
    parts.reserve(node.inputs.size());
    for (std::size_t i = 0; i < node.inputs.size(); ++i) {
      const auto &description = recipe.values[node.inputs[i]];
      if (elements(description) == 0)
        continue;
      auto *part = ggml_reshape_3d(context, input(i), inner,
                                   description.dimensions()[selected], outer);
      parts.push_back(part);
    }
    if (parts.size() == 1) {
      // This runtime admits I32 CPY/CONCAT but explicitly declines I32 DUP.
      // Keep indices as integers, including a one-input Concat's owned copy.
      auto *part = parts.front();
      result = part->type == GGML_TYPE_I32
                   ? ggml_cpy(context, part, ggml_dup_tensor(context, part))
                   : ggml_dup(context, part);
    } else if (!parts.empty()) {
      // Retaining every growing prefix costs quadratic storage and copying.
      // A balanced tree keeps the original native dependencies and bit copies,
      // with at most one full output's worth of payload per level.
      while (parts.size() > 1) {
        const auto count = parts.size();
        std::size_t next = 0;
        for (std::size_t i = 0; i + 1 < count; i += 2)
          parts[next++] = ggml_concat(context, parts[i], parts[i + 1], 1);
        if (count % 2 != 0)
          parts[next++] = parts.back();
        parts.resize(next);
      }
      result = parts.front();
    }
  } else if (node.operation.get_if<Gather>()) {
    result =
        ggml_get_rows(context, values[node.inputs[0]], values[node.inputs[1]]);
  } else if (node.operation.custom_as<model_ops::GatherElements>()) {
    auto *data = values[node.inputs[0]];
    auto *ids = values[node.inputs[1]];
    // Each public row is a native batch, not a Cartesian Gather dimension.
    data = ggml_reshape_3d(context, data, 1, data->ne[0], data->ne[1]);
    // The original CUDA GET_ROWS implementation supports same-type F32/F16/
    // BF16/I32 destinations. Its C builder defaults to F32; retain the source
    // type here so this pure selection has no widening/narrowing step.
    result = ggml_new_tensor_3d(context, data->type, 1, ids->ne[0], ids->ne[1]);
    result->op = GGML_OP_GET_ROWS;
    result->src[0] = data;
    result->src[1] = ids;
  } else if (const auto clamp = node.operation.custom_as<model_ops::Clamp>()) {
    result = ggml_clamp(context, input(0), clamp->minimum, clamp->maximum);
  } else if (node.operation.custom_as<model_ops::CausalConv1d>()) {
    auto *wide = values[node.inputs[0]];
    auto *source = forms.private_bfloat_widen_source(node.inputs[0], values);
    // The source is the real BF16 tensor/view, including row/batch padding.
    // Native typed loads retain the original float window and MAC sequence;
    // only a truthful metadata permutation replaces the private Cast/CONT.
    const bool typed = source && source->type == GGML_TYPE_BF16 &&
        std::equal(source->ne, source->ne + GGML_MAX_DIMS, wide->ne) &&
        source->nb[0] == sizeof(std::uint16_t) &&
        source->nb[1] >= std::size_t(source->ne[0]) * sizeof(std::uint16_t) &&
        source->nb[2] >= std::size_t(source->ne[1] - 1) * source->nb[1] +
                            std::size_t(source->ne[0]) * sizeof(std::uint16_t);
    auto *samples = typed
        ? ggml_permute(context, source, 1, 0, 2, 3)
        : ggml_cont(context, ggml_permute(context, wide, 1, 0, 2, 3));
    result = ggml_ssm_conv(context, samples, values[node.inputs[1]]);
  } else if (node.operation.custom_as<model_ops::GroupedMatMul>()) {
    auto *weights = values[node.inputs[0]];
    // Retain the actual bank type. Native grouped execution preserves the
    // declared F32 activation/accumulation contract in its carrier or fallback.
    auto *activations = grouped_activation(context, values[node.inputs[1]]);
    auto *ids = values[node.inputs[2]];
    result = ggml_mul_mat_id(context, weights, activations, ids);
    if (!ggml_prec_set_acc(result, GGML_PREC_F32) ||
        !ggml_prec_set_src(result, GGML_PREC_F32, 1))
      return {};
    forms.remember_grouped(node.outputs[0], result);
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
             node.operation.custom_as<model_ops::DenseLinearF32>()) {
    auto *left = native_shape(context, values[node.inputs[0]],
                              recipe.values[node.inputs[0]]);
    auto *right = values[node.inputs[1]];
    const bool direct_dense = node.operation.custom_as<model_ops::DenseLinearF32>() &&
        (recipe.values[node.inputs[0]].data_type() == DataType::kFloat32 ||
         recipe.values[node.inputs[0]].data_type() != recipe.values[node.inputs[1]].data_type());
    if (direct_dense) {
      if (left->type != GGML_TYPE_F32) left=ggml_cast(context,left,GGML_TYPE_F32);
      const bool batched=recipe.values[node.inputs[0]].rank()==3;
      const auto batches=batched?left->ne[2]:1;
      left=ggml_reshape_4d(context,left,left->ne[0],1,left->ne[1],batches);
      if(batched)right=ggml_reshape_4d(context,right,right->ne[0],right->ne[1],1,batches);
    }
    if (node.operation.get_if<MatMul>() &&
        !native_quantized(recipe.values[node.inputs[1]])) {
      right = ggml_transpose(context, right);
      if (!ggml_is_contiguous(right))
        right = ggml_cont(context, right);
    }
    // The native vector carrier can load BF16 directly and round its F32
    // reduction once at this public MatMul boundary. Preserve the existing
    // F32-result program for matrices, batches, odd K and every other type.
    const bool bfloat_vector = node.operation.get_if<MatMul>() &&
        output.data_type() == DataType::kBFloat16 &&
        left->type == GGML_TYPE_BF16 && right->type == GGML_TYPE_BF16 &&
        left->ne[1] == 1 && left->ne[2] == 1 && left->ne[3] == 1 &&
        right->ne[2] == 1 && right->ne[3] == 1 && right->ne[0] % 2 == 0 &&
        ggml_is_contiguous(left) && ggml_is_contiguous(right);
    if (bfloat_vector) {
      result = ggml_new_tensor_2d(context, GGML_TYPE_BF16, right->ne[1], 1);
      result->op = GGML_OP_MUL_MAT;
      result->src[0] = right;
      result->src[1] = left;
    } else {
      result = ggml_mul_mat(context, right, left);
    }
    if (!ggml_prec_set_acc(result, GGML_PREC_F32))
      return {};
    // F32 accumulation alone does not forbid TF32 source arithmetic in cuBLAS.
    if ((node.operation.get_if<MatMul>() ||
         node.operation.custom_as<model_ops::DenseLinearF32>()) &&
        (direct_dense || recipe.values[node.inputs[0]].data_type() == DataType::kFloat32) &&
        !native_quantized(recipe.values[node.inputs[1]]) &&
        !ggml_prec_set_src(result, GGML_PREC_F32, 1))
      return {};
  } else if (const auto *rope = node.operation.get_if<RoPE>()) {
    // One native operation retains the original explicit-frequency F32
    // arithmetic and final output rounding. The private marker prevents the
    // standard inverse-factor RoPE kernels and fusions from interpreting it.
    auto *data = values[node.inputs[0]];
    // Model graphs may rotate a last-axis slice, for example the rotary
    // suffix of a latent KV row. Keep its layout conversion inside this
    // provider; the explicit-frequency kernel itself consumes dense rows.
    if (!ggml_is_contiguous(data))
      data = ggml_cont(context, data);
    if (recipe.values[node.inputs[0]].rank() == 3)
      // BSD has one implicit head. Only native metadata changes; as_output
      // below restores the original public rank after the same RoPE kernel.
      data = ggml_reshape_4d(context, data, data->ne[0], 1,
                              data->ne[1], data->ne[2]);
    // ggml_is_contiguous ignores singleton-axis strides. The private RoPE
    // carrier requires canonical dense metadata on every axis; a reshape
    // fixes those irrelevant strides without copying the one-row suffix.
    auto stride = ggml_type_size(data->type);
    bool canonical = true;
    for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
      canonical = canonical && data->nb[axis] == stride;
      stride *= static_cast<std::size_t>(data->ne[axis]);
    }
    if (!canonical)
      data = ggml_reshape_4d(context, data, data->ne[0], data->ne[1],
                            data->ne[2], data->ne[3]);
    result = ggml_dup_tensor(context, data);
    result->op = GGML_OP_ROPE;
    result->src[0] = data;
    result->src[1] = values[node.inputs[1]];
    result->src[2] = values[node.inputs[2]];
    std::fill_n(result->op_params,
                GGML_MAX_OP_PARAMS / sizeof(result->op_params[0]), 0);
    result->op_params[kExplicitRopeContractParam] = kExplicitRopeContract;
    result->op_params[kExplicitRopeLayoutParam] =
        rope->layout == RopeLayout::kInterleaved ? GGML_ROPE_TYPE_NORMAL
                                                : GGML_ROPE_TYPE_NEOX;
  } else if (node.operation.get_if<Hadamard>()) {
    const auto width = columns(output);
    if (width > 512) {
      const auto factors = hadamard_factors(width);
      if (hadamard.size() != factors.constant_elements())
        return {};
      auto *low =
          ggml_new_tensor_2d(context, GGML_TYPE_F32, factors.low, factors.low);
      const auto low_elements =
          static_cast<std::size_t>(factors.low * factors.low);
      dense_constants.push_back(
          {low, hadamard.data(), low_elements * sizeof(float)});
      auto *high = low;
      if (factors.high != factors.low) {
        high = ggml_new_tensor_2d(context, GGML_TYPE_F32, factors.high,
                                  factors.high);
        dense_constants.push_back(
            {high, hadamard.data() + low_elements,
             (hadamard.size() - low_elements) * sizeof(float)});
      }
      // The low six bits stay contiguous. Transform each factor in natural
      // Sylvester order, normalizing once per factor, then restore low-fast
      // order.
      auto *source = ggml_reshape_3d(context, as_float(context, input(0)),
                                     factors.low, factors.high, rows(output));
      auto *first = ggml_mul_mat(context, low, source);
      ggml_mul_mat_set_hint(first, GGML_HINT_SRC0_IS_HADAMARD);
      if (!ggml_prec_set_acc(first, GGML_PREC_F32) ||
          !ggml_prec_set_src(first, GGML_PREC_F32, 1))
        return {};
      auto *swapped =
          ggml_cont(context, ggml_permute(context, first, 1, 0, 2, 3));
      auto *second = ggml_mul_mat(context, high, swapped);
      ggml_mul_mat_set_hint(second, GGML_HINT_SRC0_IS_HADAMARD);
      if (!ggml_prec_set_acc(second, GGML_PREC_F32) ||
          !ggml_prec_set_src(second, GGML_PREC_F32, 1))
        return {};
      // Keep real small matrices even when FWHT is selected: native fused
      // matmul fallbacks and debug validation may still read their payload.
      result = ggml_cont(context, ggml_permute(context, second, 1, 0, 2, 3));
      return {as_output(context, result, output)};
    }
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
    const auto *gdn = node.operation.get_if<GatedDeltaNet>();
    const auto snapshots =
        gdn ? gdn->snapshot_count
            : node.operation.custom_as<model_ops::RecurrentGatedDeltaNet>()
                  ->snapshot_count;
    auto *packed = ggml_gated_delta_net(
        context, recurrent_input(0), recurrent_input(1), recurrent_input(2),
        decay, beta, initial, snapshots);
    ggml_runtime::set_gdn_output_scale(
        packed, gdn ? gdn->output_scale
                    : node.operation.custom_as<model_ops::RecurrentGatedDeltaNet>()->output_scale);
    // Keep public GDN construction and all F32 math unchanged. The private
    // native typed loads widen BF16 exactly where the original F32 loads ran.
    // Retain the actual tensor/view identity and strides: the graph allocator
    // must see the real source, including its storage root and other readers.
    const auto typed_source = [&](std::size_t port, bool rows) {
      auto *source = forms.private_bfloat_widen_source(node.inputs[port], values);
      auto *wide = recurrent_input(port);
      if (!source || source->type != GGML_TYPE_BF16 ||
          !std::equal(source->ne, source->ne + GGML_MAX_DIMS, wide->ne) ||
          !(rows ? ggml_is_contiguous_rows(source) : ggml_is_contiguous(source)))
        return static_cast<ggml_tensor *>(nullptr);
      return source;
    };
    if (auto *source = typed_source(2, true)) packed->src[2] = source;
    // RecurrentGatedDeltaNet computes sigmoid in F32 from raw beta. Bypassing
    // that operation would change its public math and BF16 rounding boundary.
    if (gdn)
      if (auto *source = typed_source(4, false)) packed->src[4] = source;
    const auto score_count = static_cast<std::int64_t>(
        *recipe.values[node.outputs[0]].element_count());
    const auto state_count = static_cast<std::int64_t>(
        *recipe.values[node.outputs[1]].element_count());
    auto *scores =
        ggml_cont(context, ggml_view_1d(context, packed, score_count, 0));
    auto *state = ggml_view_1d(context, packed, state_count,
                               static_cast<std::size_t>(score_count) *
                                   sizeof(float));
    if (snapshots > 1) {
      auto *indices =
          ggml_new_tensor_1d(context, GGML_TYPE_I32, snapshot_order.size());
      dense_constants.push_back({indices, snapshot_order.data(),
                                 snapshot_order.size_bytes()});
      const auto width = ggml_nelements(initial) / initial->ne[3];
      state = ggml_get_rows(
          context,
          ggml_reshape_2d(context, state, width, snapshot_order.size()), indices);
    } else
      state = ggml_cont(context, state);
    return {as_output(context, scores, recipe.values[node.outputs[0]]),
            as_output(context, state, recipe.values[node.outputs[1]])};
  }
  if (!result)
    return {};
  return {as_output(context, result, output)};
}

} // namespace hyle::compute::ggml_cuda::internal
