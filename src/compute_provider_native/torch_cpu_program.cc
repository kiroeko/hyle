#include "torch_cpu_program.h"
#include "hyle/core/environment.h"

#if HYLE_COMPILER_MSVC || HYLE_COMPILER_CLANG_CL
// c10 2.8 irange's inlined iterator produces C4702 under MSVC /O2, even with
// SYSTEM headers. Keep this SDK diagnostic local to its adapter translation unit.
#pragma warning(push)
#pragma warning(disable : 4702)
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <map>
#include <new>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

#include <ATen/core/grad_mode.h>
#include <c10/util/Exception.h>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "hyle/math/scalar.h"
#include "hyle/mem/resource.h"
#include "native_plan.h"
#include "row_cache_codec.h"
#include "../compute_batch/row_cache_table.h"
#include "torchao_cpu.h"
#include "bitsandbytes_cpu.h"
#include "awq_cpu.h"
#include "compressed_tensors_cpu.h"
#include "mlx_cpu_quant.h"

namespace hyle::compute::native::internal {
namespace {
enum class Kind { kDecode, kLinear, kEncodedMatMul, kGrouped8, kGrouped4, kRoundtrip,
                  kArtifactDecode, kMlx, kCausalConv, kRowCacheIndexer, kQuantize };
struct Shape final {
  Kind kind{};
  std::int64_t e = 1, n = 0, k = 0, tokens = 0, slots = 1, activations = 1;
  float floor = 1.0e-10F;
  model_ops::Float8ScalePolicy scale_policy = model_ops::Float8ScalePolicy::kDivide;
  at::ScalarType output = at::kFloat;
  model_ops::ArtifactWeightDecode artifact{};
  model_ops::MlxQuantized mlx{};
  detail::RowCachePagedIndexerScores paged{};
  bool is_paged=false;
};
struct Program final {
  using Run = Status (*)(const Program &, std::span<const std::byte *const>,
                         std::span<std::byte *const>) noexcept;
  Shape shape{};
  std::array<TensorDesc, 6> inputs{};
  std::array<TensorDesc, 2> outputs{};
  std::size_t input_count = 0, output_count = 0;
  Run run = nullptr;
};
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kProvider,
    .replay_class = ProviderReplayClass::kManaged,
    .blocks_host = true,
    .uses_private_threads = true,
    .supports_concurrent_invocations = true,
    .capture_compatible = false};

Status translate_error(const c10::Error &error) noexcept {
  const std::string_view message = error.what_without_backtrace();
  // Pinned c10 alloc_cpu uses a plain Error for CPU allocation failure.
  // Match its two allocation diagnostics; other ATen errors remain Internal.
  if (message.find("DefaultCPUAllocator: not enough memory:") != message.npos ||
      message.find("DefaultCPUAllocator: can't allocate memory:") != message.npos)
    return Status::ResourceExhausted(message);
  return Status::Internal(message);
}

Result<Shape> select(const Operation &op, std::span<const TensorDesc> inputs,
                     std::span<const TensorDesc> outputs) noexcept {
  if (!is_torch_cpu_operation(op, inputs))
    return std::unexpected(Status::Unimplemented("original ATen CPU low precision operation is unavailable"));
  const std::size_t output_count = operation_traits(op).outputs;
  std::array<TensorDesc, 2> inferred;
  if (output_count == 0 || output_count > inferred.size())
    return std::unexpected(Status::Unimplemented("ATen CPU output arity is unavailable"));
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(op, inputs,
                                               std::span(inferred).first(output_count)));
  if (outputs.size() != output_count)
    return std::unexpected(Status::InvalidArgument("ATen CPU output count differs from inference"));
  for (std::size_t i = 0; i < output_count; ++i) {
    if (outputs[i] != inferred[i])
      return std::unexpected(Status::InvalidArgument("ATen CPU output differs from inference"));
    if (outputs[i].required_bytes().value() > static_cast<std::uint64_t>(INT64_MAX))
      return std::unexpected(Status::OutOfRange("ATen CPU output exceeds native int64 capacity"));
  }
  for (const auto &desc : inputs)
    if (desc.required_bytes().value() > static_cast<std::uint64_t>(INT64_MAX))
      return std::unexpected(Status::OutOfRange("ATen CPU carrier exceeds native int64 capacity"));
  Shape s;
  s.output = outputs[0].data_type() == DataType::kBFloat16 ? at::kBFloat16 :
      (outputs[0].data_type() == DataType::kFloat16 ? at::kHalf : at::kFloat);
  if(const auto paged=op.custom_as<detail::RowCachePagedIndexerScores>()){
    s.kind=Kind::kRowCacheIndexer;s.is_paged=true;s.paged=*paged;
    s.tokens=inputs[0].dimensions()[0];s.n=paged->score_columns;s.k=128;
  } else if (op.custom_as<model_ops::RowCacheIndexerScores>()) {
    s.kind = Kind::kRowCacheIndexer;
    s.tokens = inputs[0].dimensions()[0];
    s.n = inputs[1].dimensions()[0];
    s.k = 128;
  } else if (op.custom_as<model_ops::CausalConv1d>()) {
    s.kind = Kind::kCausalConv;
    s.n = inputs[0].dimensions()[2]; s.k = inputs[1].dimensions()[1];
    for (const auto &desc : inputs)
      if (desc.element_count().value() > static_cast<std::uint64_t>(INT64_MAX) / sizeof(float))
        return std::unexpected(Status::OutOfRange("ATen CPU convolution F32 temporary capacity overflows"));
    if (outputs[0].element_count().value() > static_cast<std::uint64_t>(INT64_MAX) / sizeof(float))
      return std::unexpected(Status::OutOfRange("ATen CPU convolution result capacity overflows"));
  } else if (const auto mlx = op.custom_as<model_ops::MlxQuantized>()) {
    s.kind = Kind::kMlx;s.mlx = *mlx;
    const auto &w = inputs[mlx->kind == model_ops::MlxQuantizedKind::kEmbedding ? 0 : 1];
    const auto &scale = inputs[mlx->kind == model_ops::MlxQuantizedKind::kEmbedding ? 1 : 2];
    s.k = scale.dimensions().back() * mlx->group_size;
    s.n = w.dimensions()[w.rank() - 2];
    s.e = mlx->kind == model_ops::MlxQuantizedKind::kGrouped ? w.dimensions()[0] : 1;
    // ATen's current-group codes/decoded matrices require more bytes than the
    // caller's final F16/BF16 output. Check that bounded native capacity too.
    const auto count = outputs[0].element_count().value();
    const std::uint64_t factor = mlx->kind == model_ops::MlxQuantizedKind::kEmbedding
        ? 64 : mlx->orientation == model_ops::MlxMatmulOrientation::kWeightAsStored
        ? std::uint64_t{64} * 64 : std::uint64_t{64} * mlx->group_size;
    if (count > static_cast<std::uint64_t>(INT64_MAX) / factor)
      return std::unexpected(Status::OutOfRange("MLX selected-group native temporary capacity overflows"));
    // A shared slot is metadata. The CPU equation collapses it before its
    // F32 conversion, so charge the physical token rows rather than T*S*K.
    auto activation_elements = inputs[0].element_count().value();
    if (mlx->kind == model_ops::MlxQuantizedKind::kGrouped &&
        inputs[0].byte_strides()[1] == 0)
      activation_elements /= static_cast<std::uint64_t>(inputs[0].dimensions()[1]);
    if (mlx->kind != model_ops::MlxQuantizedKind::kEmbedding &&
        activation_elements > static_cast<std::uint64_t>(INT64_MAX) / sizeof(float))
      return std::unexpected(Status::OutOfRange("MLX activation F32 temporary capacity overflows"));
  } else if (const auto artifact = op.custom_as<model_ops::ArtifactWeightDecode>()) {
    s.kind = Kind::kArtifactDecode; s.artifact = *artifact;
    s.n = outputs[0].dimensions()[0]; s.k = outputs[0].dimensions()[1];
    // Original packing helpers expand codes to I64 before source-typed math.
    // Check each native allocation, including the padded output-axis zeros.
    if (outputs[0].element_count().value() >
        static_cast<std::uint64_t>(INT64_MAX) / sizeof(std::int64_t))
      return std::unexpected(Status::OutOfRange("ATen artifact I64 codes temporary capacity overflows"));
    if (artifact->format == model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4 &&
        inputs[1].element_count().value() >
            static_cast<std::uint64_t>(INT64_MAX) / (8 * sizeof(std::int64_t)))
      return std::unexpected(Status::OutOfRange("ATen artifact padded I64 zeros temporary capacity overflows"));
  } else if (op.custom_as<model_ops::Float8WeightDecode>()) {
    s.kind = Kind::kDecode;
    s.k = inputs[0].dimensions()[0]; s.n = inputs[0].dimensions()[1];
  } else if (const auto q = op.custom_as<model_ops::Float8BlockQuantize>()) {
    s.kind = Kind::kQuantize; s.scale_policy = q->scale_policy; s.floor = q->absmax_floor;
    s.tokens = inputs[0].dimensions()[0]; s.k = inputs[0].dimensions()[1];
    if (inputs[0].element_count().value() > static_cast<std::uint64_t>(INT64_MAX) / sizeof(float))
      return std::unexpected(Status::OutOfRange("ATen CPU quantization F32 temporary capacity overflows"));
  } else if (op.custom_as<model_ops::BlockQuantizationRoundtrip>()) {
    s.kind = Kind::kRoundtrip; s.scale_policy = model_ops::Float8ScalePolicy::kPowerOfTwo; s.floor = 1.0e-4F;
    s.tokens = inputs[0].dimensions()[0]; s.k = inputs[0].dimensions()[1];
  } else if (op.custom_as<model_ops::GroupedFloat8Linear>() ||
             op.custom_as<model_ops::GroupedFloat4Linear>()) {
    const auto grouped = op.custom_as<model_ops::GroupedFloat8Linear>();
    s.kind = grouped ? Kind::kGrouped8 : Kind::kGrouped4;
    s.floor = grouped ? grouped->absmax_floor : 1.0e-4F;
    s.scale_policy = grouped ? grouped->scale_policy : model_ops::Float8ScalePolicy::kPowerOfTwo;
    const auto w = inputs[0].dimensions(), a = inputs[1].dimensions();
    s.e = w[0]; s.n = w[1]; s.k = w[2]; s.tokens = a[0]; s.activations = a[1];
    s.slots = inputs[2].dimensions()[1];
  } else {
    const auto linear = op.custom_as<model_ops::Float8BlockLinear>();
    s.kind = linear ? Kind::kLinear : Kind::kEncodedMatMul;
    if (linear) { s.floor = linear->absmax_floor;
      s.scale_policy = linear->scale_policy; }
    s.tokens = inputs[0].dimensions()[0]; s.k = inputs[0].dimensions()[1];
    s.n = inputs[1].dimensions()[1];
    auto weight = float8_weight_description(s.k, s.n, Float8ScaleMode::kFloat32Block128);
    if (!weight || inputs[1] != *weight)
      return std::unexpected(Status::Unimplemented("ATen CPU requires canonical block128 weights"));
    if (!linear) {
      auto activation = float8_activation_description(s.tokens, s.k, Float8ScaleMode::kFloat32Block128);
      if (!activation || inputs[0] != *activation ||
          op.get_if<MatMul>()->accumulation_type != DataType::kFloat32)
        return std::unexpected(Status::Unimplemented("ATen encoded MatMul requires canonical block128 activation and F32 accumulation"));
    }
  }
  const auto product = [](std::int64_t a, std::int64_t b) {
    return math::try_multiply(static_cast<std::uint64_t>(a), static_cast<std::uint64_t>(b));
  };
  auto rows = product(s.tokens, s.slots);
  auto matrix = product(s.n, s.k);
  auto elements = rows && matrix ? math::try_multiply(*rows, *matrix) : std::nullopt;
  if (!matrix || (s.kind != Kind::kDecode && s.kind != Kind::kArtifactDecode && s.kind != Kind::kMlx && s.kind != Kind::kCausalConv && s.kind != Kind::kRowCacheIndexer &&
                  (!elements || *elements > INT64_MAX / 32)))
    return std::unexpected(Status::OutOfRange("ATen CPU selected tensor capacity overflows"));
  return s;
}

struct Quantized final { at::Tensor codes, scales, ids; };
Quantized quantize(const at::Tensor &input, const Shape &s) {
  auto data = input.to(at::kFloat).reshape({-1, 128});
  at::Tensor scale;
  if (s.scale_policy == model_ops::Float8ScalePolicy::kPowerOfTwo) {
    // Original TorchAO _to_mx_rceil receives the model's explicit absmax floor.
    auto maximum = data.abs().amax(-1, true).clamp_min(s.floor);
    auto exponent = torchao_cpu::f32_to_e8m0_rceil(maximum * (1.0 / 448.0));
    scale = torchao_cpu::e8m0_scale_to_reciprocal_fp32(exponent).reciprocal();
  } else if (s.scale_policy == model_ops::Float8ScalePolicy::kMultiplyReciprocal) {
    // Separate F32 source boundaries: a stored reciprocal448 literal,
    // scale, reciprocal(scale), then activation multiplication.
    auto maximum = data.abs().amax(-1, true).clamp_min(s.floor);
    scale = maximum * std::bit_cast<float>(std::uint32_t{0x3b124925});
    auto reciprocal = scale.reciprocal();
    auto codes = (data * reciprocal).clamp(-448, 448).to(at::kFloat8_e4m3fn);
    return {std::move(codes), std::move(scale), {}};
  } else scale = torchao_cpu::choose_scale_float8(data, s.floor);
  return {torchao_cpu::quantize_affine_float8(data, scale), std::move(scale), {}};
}
Quantized weight(const at::Tensor &carrier, const Shape &s, const at::Tensor &ids) {
  const bool four = s.kind == Kind::kGrouped4;
  const auto payload = s.e * s.n * (s.k / (four ? 2 : 1));
  auto codes = carrier.slice(0, 0, payload).reshape({s.e, s.n, s.k / (four ? 2 : 1)});
  auto scales = carrier.slice(0, payload).view(at::kFloat).reshape(
      {s.e, four ? s.n : (s.n + 127) / 128, s.k / (four ? 32 : 128)});
  // Retain encoded weights. The original FP4 codec creates several tensors;
  // select grouped rows only after slicing their current K block.
  if (!four && !ids.defined())
    scales = scales.repeat_interleave(128, 1).slice(1, 0, s.n);
  return {std::move(codes), std::move(scales), ids};
}
at::Tensor block_dot(Quantized a, Quantized w, const Shape &s) {
  const auto rows = s.tokens * s.slots;
  const auto group = s.kind == Kind::kGrouped4 ? 32 : 128;
  const auto blocks = s.k / group;
  auto lhs = a.codes.to(at::kFloat).reshape({rows, blocks, 1, group});
  auto scales = a.scales.reshape({rows, s.k / 128});
  if (group == 32) scales = scales.repeat_interleave(4, 1);
  auto accumulated = at::zeros({rows, 1, s.n}, at::kFloat);
  // Fixed original K-block order. Decode and matrix temporaries are bounded
  // by one original block; all codec and payload arithmetic remains in ATen.
  const auto encoded_group = s.kind == Kind::kGrouped4 ? group / 2 : group;
  for (std::int64_t block = 0; block < blocks; ++block) {
    auto encoded = w.codes.slice(2, block * encoded_group,
                                 (block + 1) * encoded_group);
    encoded = w.ids.defined() ? encoded.index_select(0, w.ids)
                              : encoded.contiguous();
    auto weight_scales = w.scales.select(2, block);
    if (w.ids.defined()) {
      weight_scales = weight_scales.index_select(0, w.ids);
      if (s.kind != Kind::kGrouped4)
        weight_scales = weight_scales.repeat_interleave(128, 1).slice(1, 0, s.n);
    }
    auto decoded = s.kind == Kind::kGrouped4
        ? torchao_cpu::f4_unpacked_to_f32(torchao_cpu::unpack_uint4(encoded))
        : encoded.view(at::kFloat8_e4m3fn).to(at::kFloat);
    auto dots = at::matmul(lhs.select(1, block), decoded.transpose(1, 2));
    auto scaled = dots * scales.select(1, block).reshape({rows, 1, 1});
    accumulated = accumulated + scaled * weight_scales.unsqueeze(1);
  }
  return accumulated.reshape({s.tokens, s.slots, s.n}).to(s.output);
}

at::Tensor grouped_block_dot(Quantized a, Quantized w, const Shape &s) {
  const auto rows = s.tokens * s.slots;
  const auto *ids = w.ids.const_data_ptr<std::int64_t>();
  std::vector<std::pair<std::int64_t, std::int64_t>> ordered;
  ordered.reserve(static_cast<std::size_t>(rows));
  for (std::int64_t row = 0; row < rows; ++row)
    ordered.emplace_back(ids[row], row);
  std::sort(ordered.begin(), ordered.end());
  const bool repeated =
      std::adjacent_find(ordered.begin(), ordered.end(),
                         [](const auto &left, const auto &right) {
                           return left.first == right.first;
                         }) != ordered.end();
  if (!repeated) {
    // Decode has no reusable expert here. Keep the original batched path,
    // including the single-token case, without constructing bucket tensors.
    if (s.activations == 1 && s.slots != 1) {
      a.codes = a.codes.expand({s.tokens, s.slots, s.k});
      a.scales = a.scales.expand({s.tokens, s.slots, s.k / 128});
    }
    return block_dot(std::move(a), std::move(w), s);
  }
  struct Bucket final {
    std::vector<std::int64_t> experts, positions, activations;
    at::Tensor expert_ids, row_ids, activation_ids;
  };
  // Equal occurrence counts share a native batched GEMM. This avoids both
  // per-token weight copies and padding every expert to the largest count.
  std::map<std::int64_t, Bucket> buckets;
  for (std::size_t first = 0; first < ordered.size();) {
    auto end = first + 1;
    while (end < ordered.size() && ordered[end].first == ordered[first].first)
      ++end;
    auto &bucket = buckets[static_cast<std::int64_t>(end - first)];
    bucket.experts.push_back(ordered[first].first);
    for (auto item = first; item < end; ++item) {
      const auto row = ordered[item].second;
      bucket.positions.push_back(row);
      bucket.activations.push_back((row / s.slots) * s.activations +
                                   (s.activations == 1 ? 0 : row % s.slots));
    }
    first = end;
  }
  for (auto &[count, bucket] : buckets) {
    (void)count;
    bucket.expert_ids = at::tensor(bucket.experts, at::kLong);
    bucket.row_ids = at::tensor(bucket.positions, at::kLong);
    bucket.activation_ids = at::tensor(bucket.activations, at::kLong);
  }
  const auto group = s.kind == Kind::kGrouped4 ? 32 : 128;
  const auto encoded_group = s.kind == Kind::kGrouped4 ? group / 2 : group;
  const auto codes = a.codes.reshape({s.tokens * s.activations, s.k});
  const auto scales = a.scales.reshape({s.tokens * s.activations, s.k / 128});
  auto accumulated = at::zeros({rows, s.n}, at::kFloat);
  // Preserve the original K-block accumulation/scale order. Each referenced
  // expert is decoded once per block; shared activations stay unexpanded.
  for (std::int64_t block = 0; block < s.k / group; ++block) {
    const auto activation =
        codes.slice(1, block * group, (block + 1) * group).to(at::kFloat);
    const auto activation_scales =
        scales.select(1, group == 32 ? block / 4 : block);
    const auto selected_codes =
        w.codes.slice(2, block * encoded_group, (block + 1) * encoded_group);
    const auto selected_scales = w.scales.select(2, block);
    for (const auto &[count, bucket] : buckets) {
      const auto experts = static_cast<std::int64_t>(bucket.experts.size());
      const auto encoded = selected_codes.index_select(0, bucket.expert_ids);
      const auto decoded =
          s.kind == Kind::kGrouped4
              ? torchao_cpu::f4_unpacked_to_f32(
                    torchao_cpu::unpack_uint4(encoded))
              : encoded.view(at::kFloat8_e4m3fn).to(at::kFloat);
      auto weight_scales = selected_scales.index_select(0, bucket.expert_ids);
      if (s.kind != Kind::kGrouped4)
        weight_scales =
            weight_scales.repeat_interleave(128, 1).slice(1, 0, s.n);
      const auto lhs = activation.index_select(0, bucket.activation_ids)
                           .reshape({experts, count, group});
      const auto dots = at::matmul(lhs, decoded.transpose(1, 2));
      const auto scaled =
          dots * activation_scales.index_select(0, bucket.activation_ids)
                     .reshape({experts, count, 1});
      const auto prior = accumulated.index_select(0, bucket.row_ids)
                             .reshape({experts, count, s.n});
      const auto next = prior + scaled * weight_scales.unsqueeze(1);
      accumulated.index_copy_(0, bucket.row_ids,
                              next.reshape({experts * count, s.n}));
    }
  }
  return accumulated.reshape({s.tokens, s.slots, s.n}).to(s.output);
}
Result<at::Tensor> indexer_scores(const std::array<at::Tensor, 6> &input,
                                  const Shape &shape) {
  const auto *ranges = input[3].const_data_ptr<std::int32_t>();
  std::int64_t first_visible = shape.n, last_visible = 0;
  // Validate every query before allocating results or decoding a bank row.
  for (std::int64_t row = 0; row < shape.tokens; ++row) {
    const auto begin = ranges[row * 2], end = ranges[row * 2 + 1];
    if (begin < 0 || end < begin || std::int64_t(end) > shape.n)
      return std::unexpected(
          Status::OutOfRange("row cache indexer range is outside its bank"));
    if (begin != end) {
      first_visible = (std::min)(first_visible, std::int64_t(begin));
      last_visible = (std::max)(last_visible, std::int64_t(end));
    }
  }
  auto result = at::full({shape.tokens, shape.n},
                         -std::numeric_limits<float>::infinity(), at::kFloat);
  if (first_visible >= last_visible)
    return result;
  constexpr std::int64_t kKeyTileRows = 256;
  const auto *bank = input[1].const_data_ptr<std::uint8_t>();
  // One fixed BF16 tile, shared by all query rows. The history is never
  // decoded or widened in full; ATen owns its normal matmul/reduction scratch.
  auto decoded = at::empty({(std::min)(shape.n, kKeyTileRows), 128}, at::kBFloat16);
  auto *decoded_bytes = reinterpret_cast<std::byte *>(decoded.data_ptr());
  for (std::int64_t first = first_visible; first < last_visible;) {
    const auto rows = (std::min)(last_visible - first, kKeyTileRows);
    for (std::int64_t row = 0; row < rows; ++row) {
      const auto key = first + row;
      bool referenced = false;
      for (std::int64_t query = 0; query < shape.tokens && !referenced; ++query)
        referenced = ranges[query * 2] <= key && key < ranges[query * 2 + 1];
      if (referenced)
        decode_row_cache_float4(
            reinterpret_cast<const std::byte *>(bank + key * 68),
            decoded_bytes + row * 128 * sizeof(c10::BFloat16));
    }
    for (std::int64_t query = 0; query < shape.tokens; ++query) {
      const auto begin = (std::max)(first, std::int64_t(ranges[query * 2]));
      const auto end = (std::min)(first + rows, std::int64_t(ranges[query * 2 + 1]));
      if (begin >= end) continue;
      const auto q = input[0].select(0, query);
      const auto weights = input[2].select(0, query).unsqueeze(1);
      const auto keys = decoded.slice(0, begin - first, end - first);
      // Original BF16 storage boundaries: F32-accumulating BF16 matmul,
      // BF16 ReLU/product, F32-accumulating BF16 sum, then lossless widening.
      auto dots = at::matmul(q, keys.transpose(0, 1));
      auto weighted = dots.relu() * weights;
      auto scores = weighted.sum(0).to(at::kFloat);
      result.select(0, query).slice(0, begin, end).copy_(scores);
    }
    const auto tile_end = first + rows;
    first = last_visible;
    // Skip empty gaps between request ranges instead of walking the bank
    // span. A range still covering tile_end makes the next tile adjacent.
    for (std::int64_t query = 0; query < shape.tokens; ++query)
      if (ranges[query * 2 + 1] > tile_end)
        first = (std::min)(first, (std::max)(
            tile_end, std::int64_t(ranges[query * 2])));
  }
  return result;
}

Result<at::Tensor> paged_indexer_scores(const std::array<at::Tensor,6>&input,const Shape&s){
  const auto*ranges=input[3].const_data_ptr<std::int32_t>();
  const auto*bank=input[1].const_data_ptr<std::uint8_t>();
  HYLE_ASSIGN_OR_RETURN(auto table,detail::RowCacheTable::create(s.paged.mapping,
    reinterpret_cast<const std::byte*>(input[4].const_data_ptr<std::uint8_t>()),std::size_t(s.tokens)));
  // Validate every request and query before allocating/writing the private
  // result or reading a packed row. Columns remain request-local, not slots.
  for(std::size_t r=0;r<table.requests;++r){
    if(table.length(r)>std::uint64_t(s.n))return std::unexpected(Status::OutOfRange("paged indexer history exceeds score columns"));
    for(auto q=table.query(r);q<table.query(r+1);++q)
      if(ranges[q*2]<0 || ranges[q*2+1]<ranges[q*2] || std::uint64_t(ranges[q*2+1])>table.length(r))
        return std::unexpected(Status::OutOfRange("paged indexer range is outside its request history"));
  }
  auto result=at::full({s.tokens,s.n},-std::numeric_limits<float>::infinity(),at::kFloat);
  if(!s.tokens || !s.n)return result;
  constexpr std::int64_t tile_rows=256;
  auto decoded=at::empty({(std::min)(tile_rows,s.n),128},at::kBFloat16);
  auto*decoded_bytes=reinterpret_cast<std::byte*>(decoded.data_ptr<c10::BFloat16>());
  for(std::size_t request=0;request<table.requests;++request){
    const auto qbegin=table.query(request),qend=table.query(request+1);
    std::int64_t first=s.n,last=0;
    for(auto q=qbegin;q<qend;++q)if(ranges[q*2]<ranges[q*2+1]){
      first=(std::min)(first,std::int64_t(ranges[q*2]));last=(std::max)(last,std::int64_t(ranges[q*2+1]));}
    while(first<last){const auto rows=(std::min)(tile_rows,last-first);
      for(std::int64_t row=0;row<rows;++row){const auto key=first+row;bool referenced=false;
        for(auto q=qbegin;q<qend && !referenced;++q)referenced=ranges[q*2]<=key && key<ranges[q*2+1];
        if(referenced)decode_row_cache_float4(reinterpret_cast<const std::byte*>(bank+std::size_t(table.physical(request,std::uint64_t(key)))*68),
          decoded_bytes+row*128*sizeof(c10::BFloat16));}
      for(auto q=qbegin;q<qend;++q){const auto begin=(std::max)(first,std::int64_t(ranges[q*2]));
        const auto end=(std::min)(first+rows,std::int64_t(ranges[q*2+1]));if(begin>=end)continue;
        auto dots=at::matmul(input[0].select(0,q),decoded.slice(0,begin-first,end-first).transpose(0,1));
        auto weighted=dots.relu()*input[2].select(0,q).unsqueeze(1);
        result.select(0,q).slice(0,begin,end).copy_(weighted.sum(0).to(at::kFloat));}
      const auto tile_end=first+rows;first=last;
      for(auto q=qbegin;q<qend;++q)if(ranges[q*2+1]>tile_end)
        first=(std::min)(first,(std::max)(tile_end,std::int64_t(ranges[q*2])));
    }
  }
  return result;
}

template <Kind Selected>
Status execute(const Program &p, std::span<const std::byte *const> pointers,
               std::span<std::byte *const> destinations) noexcept {
  if constexpr (Selected == Kind::kMlx) {
    // No selected rows means no reads, writes, or K-sized unpack indices.
    if (p.shape.mlx.kind == model_ops::MlxQuantizedKind::kEmbedding &&
        p.outputs[0].element_count().value() == 0)
      return Status::Ok();
  }
  try {
    at::NoGradGuard no_grad;
    std::array<at::Tensor, 6> input;
    for (std::size_t i = 0; i < p.input_count; ++i) {
      const auto &desc = p.inputs[i];
      if (!desc.is_plain()) {
        input[i] = at::from_blob(const_cast<std::byte *>(pointers[i]),
            {static_cast<std::int64_t>(desc.required_bytes().value())}, at::kByte);
      } else {
        const auto type = desc.data_type() == DataType::kUInt32 ? at::kUInt32
            : (desc.data_type() == DataType::kInt32 ? at::kInt
            : (desc.data_type() == DataType::kUInt8 ? at::kByte
            : (desc.data_type() == DataType::kFloat16 ? at::kHalf
            : (desc.data_type() == DataType::kBFloat16 ? at::kBFloat16 : at::kFloat))));
        if constexpr (Selected == Kind::kMlx) {
          if (i == 0 && p.shape.mlx.kind == model_ops::MlxQuantizedKind::kGrouped) {
            std::array<std::int64_t, 3> strides{};
            const auto width = static_cast<std::int64_t>(data_type_size(desc.data_type()));
            for (std::size_t axis = 0; axis < strides.size(); ++axis)
              strides[axis] = desc.byte_strides()[axis] / width;
            input[i] = at::from_blob(const_cast<std::byte *>(pointers[i]),
                at::IntArrayRef(desc.dimensions().data(), desc.rank()),
                at::IntArrayRef(strides.data(), strides.size()), type);
            continue;
          }
        }
        input[i] = at::from_blob(const_cast<std::byte *>(pointers[i]),
            at::IntArrayRef(desc.dimensions().data(), desc.rank()), type);
      }
    }
    const auto &s = p.shape;
    constexpr auto activation_index = (Selected == Kind::kGrouped8 || Selected == Kind::kGrouped4) ? 1u : 0u;
    if (Selected != Kind::kDecode && Selected != Kind::kArtifactDecode && Selected != Kind::kMlx &&
        Selected != Kind::kEncodedMatMul && Selected != Kind::kCausalConv && Selected != Kind::kRowCacheIndexer &&
        !at::isfinite(input[activation_index]).all().item<bool>())
      return Status::InvalidArgument("original ATen CPU quantization requires finite activation");
    at::Tensor result;
    if constexpr (Selected == Kind::kRowCacheIndexer) {
      if(s.is_paged){HYLE_ASSIGN_OR_RETURN(result,paged_indexer_scores(input,s));}
      else{HYLE_ASSIGN_OR_RETURN(result, indexer_scores(input, s));}
    } else if constexpr (Selected == Kind::kCausalConv) {
      // Keep the declared F32 accumulator and one final storage rounding.
      // Layout/casts are original ATen operations; no HYLE numerical kernel.
      auto samples = input[0].permute({0, 2, 1}).to(at::kFloat).contiguous();
      auto weights = input[1].to(at::kFloat).reshape({s.n, 1, s.k});
      const std::array<std::int64_t,1> stride{1}, padding{0}, dilation{1};
      result = at::conv1d(samples, weights, std::nullopt, stride, padding, dilation, s.n)
                   .permute({0, 2, 1}).to(s.output).contiguous();
    } else if constexpr (Selected == Kind::kMlx) {
      const bool embedding = s.mlx.kind == model_ops::MlxQuantizedKind::kEmbedding;
      const bool grouped = s.mlx.kind == model_ops::MlxQuantizedKind::kGrouped;
      const bool affine = s.mlx.format == model_ops::MlxWeightFormat::kAffine;
      at::Tensor ids;
      if (embedding || grouped) {
        ids = input[p.input_count - 1].reshape({-1}).to(at::kLong);
        if (at::logical_or(ids < 0, ids >= (embedding ? s.n : s.e)).any().item<bool>())
          return Status::OutOfRange("MLX original selected row/expert id is outside its source");
      }
      if (embedding) {
        result = borrowed::mlx_cpu_quant::embedding(input[0], input[1], input[2],
                                                    ids, static_cast<int>(s.mlx.bits), s.mlx.group_size);
      } else {
        using Format = borrowed::mlx_cpu_quant::Format;
        const auto format = affine ? Format::affine :
            (s.mlx.format == model_ops::MlxWeightFormat::kMxfp4 ? Format::mxfp4 : Format::mxfp8);
        result = borrowed::mlx_cpu_quant::matmul(input[0], input[1], input[2],
            affine ? input[3] : at::Tensor{}, ids, format, static_cast<int>(s.mlx.bits),
            s.mlx.group_size, s.output,
            s.mlx.orientation == model_ops::MlxMatmulOrientation::kWeightAsStored);
      }
    } else if constexpr (Selected == Kind::kArtifactDecode) {
      switch (s.artifact.format) {
      case model_ops::WeightArtifactFormat::kNf4:
        result = borrowed::bitsandbytes_cpu::decode_nf4(input[0], input[1],
            input[2], s.artifact.group_size, s.n, s.k, s.output); break;
      case model_ops::WeightArtifactFormat::kNf4DoubleQuant: {
        auto scales = borrowed::bitsandbytes_cpu::nested_scales(input[1],
            input[2], input[3], input[4]);
        result = borrowed::bitsandbytes_cpu::decode_nf4(input[0], scales,
            input[5], s.artifact.group_size, s.n, s.k, s.output); break;
      }
      case model_ops::WeightArtifactFormat::kAwqGemm4:
        result = borrowed::awq_cpu::decode_gemm4(input[0], input[1], input[2],
                                               s.artifact.group_size).to(s.output); break;
      case model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4:
        result = borrowed::compressed_tensors_cpu::decode_symmetric4(
            input[0], input[1], s.artifact.group_size).to(s.output); break;
      case model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4:
        result = borrowed::compressed_tensors_cpu::decode_asymmetric4(
            input[0], input[1], input[2], s.artifact.group_size).to(s.output); break;
      case model_ops::WeightArtifactFormat::kGptqV1Int4: {
        const auto *indices = input[3].const_data_ptr<std::int32_t>();
        const auto groups = s.k / s.artifact.group_size;
        // Check all current indices before indexing even the first scale or
        // zero row. g_idx maps original K to a group; it does not permute Q/A.
        for (std::int64_t k = 0; k < s.k; ++k)
          if (indices[k] < 0 || indices[k] >= groups)
            return Status::OutOfRange("GPTQ group index is outside the original bank");
        const auto shifts = at::arange(0, 32, 4, at::kLong);
        auto codes = at::bitwise_and(at::bitwise_right_shift(
            input[0].to(at::kLong).unsqueeze(1), shifts.reshape({1,8,1})), 15)
            .reshape({s.k,s.n}).to(at::kChar);
        auto zeros = at::bitwise_and(at::bitwise_right_shift(
            input[1].to(at::kLong).unsqueeze(-1), shifts), 15)
            .reshape({groups,s.n}).to(at::kChar) + 1;
        const auto ids = input[3].to(at::kLong);
        // Both integer operands are I8 and their difference [-16,14] is exact.
        // Preserve the stored F16 scale's product before requested conversion.
        result = (input[2].index_select(0,ids) *
            (codes - zeros.index_select(0,ids))).transpose(0,1).to(s.output).contiguous();
        break;
      }
      }
    } else if constexpr (Selected == Kind::kDecode) {
      auto w = weight(input[0], s, {});
      result = (w.codes.view(at::kFloat8_e4m3fn).to(at::kFloat)
                    .reshape({1, s.n, s.k / 128, 128}) * w.scales.unsqueeze(-1))
          .reshape({s.n, s.k}).to(s.output);
    } else if constexpr (Selected == Kind::kQuantize) {
      auto q = quantize(input[0], s);
      // Copy the actual E4 codes and F32 scales independently. The quantizer's
      // flattened group order is exactly row-major [M,K/128]. No QDQ or host
      // reconstruction is inserted between the typed outputs and their users.
      auto codes = at::from_blob(destinations[0], {s.tokens, s.k}, at::kFloat8_e4m3fn);
      auto scales = at::from_blob(destinations[1], {s.tokens, s.k / 128}, at::kFloat);
      codes.copy_(q.codes.reshape({s.tokens, s.k}));
      scales.copy_(q.scales.reshape({s.tokens, s.k / 128}));
      return Status::Ok();
    } else if constexpr (Selected == Kind::kRoundtrip) {
      auto q = quantize(input[0], s);
      result = (q.codes.to(at::kFloat) * q.scales).reshape({s.tokens, s.k}).to(s.output);
    } else if constexpr (Selected == Kind::kGrouped8 || Selected == Kind::kGrouped4) {
      auto ids = input[2].reshape({-1}).to(at::kLong);
      if (at::logical_or(ids < 0, ids >= s.e).any().item<bool>())
        return Status::OutOfRange("ATen CPU selected expert ID is out of range");
      auto w = weight(input[0], s, ids);
      auto q = quantize(input[1], s);
      q.codes = q.codes.reshape({s.tokens, s.activations, s.k});
      q.scales = q.scales.reshape({s.tokens, s.activations, s.k / 128});
      result = grouped_block_dot(std::move(q), std::move(w), s);
    } else {
      auto w = weight(input[1], s, {});
      // Decode one shared matrix block. Matmul and scale multiplication
      // broadcast its batch1 payload across tokens without duplicating it.
      Quantized q;
      if constexpr (Selected == Kind::kLinear) q = quantize(input[0], s);
      else {
        const auto bytes = s.tokens * s.k;
        q.codes = input[0].slice(0, 0, bytes).view(at::kFloat8_e4m3fn);
        q.scales = input[0].slice(0, bytes).view(at::kFloat);
      }
      result = block_dot(std::move(q), std::move(w), s).reshape({s.tokens, s.n});
    }
    auto output = at::from_blob(destinations[0],
        at::IntArrayRef(p.outputs[0].dimensions().data(), p.outputs[0].rank()), s.output);
    output.copy_(result);
    return Status::Ok();
  } catch (const c10::OutOfMemoryError &e) {
    return Status::ResourceExhausted(e.what_without_backtrace());
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("ATen CPU private allocation failed");
  } catch (const c10::Error &e) {
    return translate_error(e);
  } catch (const std::exception &e) {
    return Status::Internal(e.what());
  }
}

void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  auto *p = static_cast<Program *>(state);
  if (!p) return;
  p->~Program(); allocator.deallocate(p, sizeof(Program), alignof(Program));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView, void **out) noexcept {
  if (!out) return Status::InvalidArgument("ATen CPU invocation output is null");
  *out = nullptr; return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}
detail::EnqueueResult submit(void *state, void *, detail::QueueSequence &,
                            std::span<const ConstTensorView> inputs,
                            std::span<const TensorView> outputs) noexcept {
  const auto &p = *static_cast<const Program *>(state);
  if (inputs.size() != p.input_count || outputs.size() != p.output_count)
    return completed(Status::Internal("ATen CPU binding count differs"));
  std::array<mem::HostMapping, 6> mappings;
  std::array<const std::byte *, 6> pointers{};
  std::array<mem::HostMapping, 2> mapped_outputs;
  std::array<std::byte *, 2> destinations{};
  Status status = Status::Ok();
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto buffer = inputs[i].storage();
    if (!buffer.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent))
      status = buffer.invalidate();
    if (!status.ok()) break;
    auto mapping = buffer.map_host();
    if (!mapping) { status = std::move(mapping).error(); break; }
    mappings[i] = std::move(*mapping); pointers[i] = mappings[i].bytes().data();
  }
  for (std::size_t i = 0; status.ok() && i < outputs.size(); ++i) {
    auto mapping = outputs[i].storage().map_host(mem::HostAccess::kWrite);
    if (!mapping) { status = std::move(mapping).error(); break; }
    mapped_outputs[i] = std::move(*mapping);
    auto writable = mapped_outputs[i].writable_bytes();
    if (!writable) { status = std::move(writable).error(); break; }
    destinations[i] = writable->data();
  }
  if (status.ok())
    status = p.run(p, std::span(pointers).first(p.input_count),
                   std::span(destinations).first(p.output_count));
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (destinations[i] && !outputs[i].storage().resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent))
      preserve_first_error(status, mapped_outputs[i].flush());
    preserve_first_error(status, mapped_outputs[i].close());
  }
  for (auto &mapping : mappings) preserve_first_error(status, mapping.close());
  return completed(std::move(status));
}
const detail::ProviderPreparedVTable kVTable{
    .destroy_prepared = &destroy, .alias_support = &alias,
    .create_invocation = &create_invocation, .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
detail::ResourceBindingRequirement host(const TensorDesc &desc) noexcept {
  return {.required_properties = mem::MemoryProperty::kHostAccessible,
          .minimum_alignment = desc.required_alignment()};
}
} // namespace

bool inspect_torch_cpu_access(const detail::ProviderPrepared &prepared,
                             detail::OperationPreparedAccess &access) noexcept {
  if (prepared.vtable != &kVTable || !prepared.state) return false;
  const auto &program = *static_cast<const Program *>(prepared.state);
  if (program.run != &execute<Kind::kRowCacheIndexer> || !program.shape.is_paged)
    return false;
  access = {.projected_input = 1, .allows_empty_footprint = true};
  return true;
}

bool is_torch_cpu_operation(const Operation &operation,
                           std::span<const TensorDesc> inputs) noexcept {
  if (operation.custom_as<model_ops::RowCacheIndexerScores>() || operation.custom_as<detail::RowCachePagedIndexerScores>())
    return true;
  if (operation.custom_as<model_ops::CausalConv1d>())
    return inputs.size() == 2 &&
        (inputs[0].data_type() == DataType::kFloat16 || inputs[0].data_type() == DataType::kBFloat16);
  if (const auto q = operation.custom_as<model_ops::BlockQuantizationRoundtrip>())
    return q->format == model_ops::BlockQuantizationFormat::kFloat8E4M3Block128;
  return operation.custom_as<model_ops::MlxQuantized>() ||
      operation.custom_as<model_ops::ArtifactWeightDecode>() ||
      operation.custom_as<model_ops::Float8WeightDecode>() ||
      operation.custom_as<model_ops::Float8BlockLinear>() ||
      operation.custom_as<model_ops::Float8BlockQuantize>() ||
      operation.custom_as<model_ops::GroupedFloat8Linear>() ||
      operation.custom_as<model_ops::GroupedFloat4Linear>() ||
      (operation.get_if<MatMul>() && inputs.size() == 2 &&
       inputs[1].storage_format().id() == kFloat8WeightFloat32Block128);
}
Status propose_torch_cpu(const detail::ProviderOperationRequest &request,
    std::span<detail::ResourceBindingRequirement> in,
    std::span<detail::ResourceBindingRequirement> out,
    detail::ProviderOperationProposal *proposal) noexcept {
  if (!request.shape.exact()) return Status::Unimplemented("ATen CPU requires exact shapes");
  HYLE_ASSIGN_OR_RETURN(auto shape, select(request.operation, request.inputs, request.outputs));
  for (std::size_t i = 0; i < in.size(); ++i) in[i] = host(request.inputs[i]);
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = host(request.outputs[i]);
  *proposal = {.route = static_cast<std::uint64_t>(request.operation.kind()),
      .runtime = kRuntime, .workspace_resource = host(request.outputs[0]),
      .cost = {.workspace = WorkspaceRequirement{}, .persistent_bytes = sizeof(Program),
          .copy_bytes = std::nullopt, .repack_bytes = std::nullopt,
          .synchronization_count = 0, .dispatch_count = 1},
      .implementation = ProviderImplementationId{0x544f524348000001ULL + static_cast<std::uint64_t>(shape.kind)}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare_torch_cpu(const detail::ProviderPrepareRequest &request,
    mem::AllocatorRef allocator, detail::ProviderPrepared *out) noexcept {
  auto shape = select(request.operation, request.inputs, request.outputs);
  if (!shape) return shape.error().IsUnimplemented()
      ? ProviderPrepareOutcome::unsupported(std::move(shape).error())
      : ProviderPrepareOutcome::error(std::move(shape).error());
  // Initialize the library's persistent CPU workers/dispatch on isolated data.
  // No user payload, model state or provider competition is evaluated here.
  mark_thread_runtime_used();
  static std::once_flag initialized;
  try {
    std::call_once(initialized, [] {
      at::NoGradGuard no_grad;
      auto a = at::zeros({64, 128}, at::kFloat);
      auto b = at::zeros({128, 128}, at::kFloat);
      (void)at::matmul(a, b);
      (void)torchao_cpu::quantize_affine_float8(a, at::ones({64, 1}, at::kFloat));
      (void)torchao_cpu::f4_unpacked_to_f32(at::zeros({512}, at::kByte));
    });
  } catch (const std::bad_alloc &) {
    return ProviderPrepareOutcome::error(Status::ResourceExhausted("ATen CPU isolated initialization allocation failed"));
  } catch (const c10::Error &e) {
    auto status = translate_error(e);
    return status.IsResourceExhausted()
        ? ProviderPrepareOutcome::resource_declined(std::move(status))
        : ProviderPrepareOutcome::error(std::move(status));
  } catch (const std::exception &e) {
    return ProviderPrepareOutcome::error(Status::Internal(e.what()));
  }
  auto *storage = allocator.allocate(sizeof(Program), alignof(Program));
  if (!storage) return ProviderPrepareOutcome::error(Status::ResourceExhausted("ATen CPU program allocation failed"));
  auto *p = ::new (storage) Program{.shape = *shape,
      .input_count = request.inputs.size(), .output_count = request.outputs.size()};
  switch (shape->kind) {
  case Kind::kDecode: p->run = &execute<Kind::kDecode>; break;
  case Kind::kLinear: p->run = &execute<Kind::kLinear>; break;
  case Kind::kEncodedMatMul: p->run = &execute<Kind::kEncodedMatMul>; break;
  case Kind::kGrouped8: p->run = &execute<Kind::kGrouped8>; break;
  case Kind::kGrouped4: p->run = &execute<Kind::kGrouped4>; break;
  case Kind::kRoundtrip: p->run = &execute<Kind::kRoundtrip>; break;
  case Kind::kArtifactDecode: p->run = &execute<Kind::kArtifactDecode>; break;
  case Kind::kMlx: p->run = &execute<Kind::kMlx>; break;
  case Kind::kCausalConv: p->run = &execute<Kind::kCausalConv>; break;
  case Kind::kRowCacheIndexer: p->run = &execute<Kind::kRowCacheIndexer>; break;
  case Kind::kQuantize: p->run = &execute<Kind::kQuantize>; break;
  }
  for (std::size_t i = 0; i < p->input_count; ++i) p->inputs[i] = request.inputs[i];
  for (std::size_t i = 0; i < p->output_count; ++i) p->outputs[i] = request.outputs[i];
  *out = {.vtable = &kVTable, .state = p, .workspace = {}, .runtime = kRuntime,
          .persistent_bytes = sizeof(Program), .invocation_private_bytes = std::nullopt};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::native::internal

#if HYLE_COMPILER_MSVC || HYLE_COMPILER_CLANG_CL
#pragma warning(pop)
#endif
