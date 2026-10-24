#pragma once

#include "d5_glm_remaining_fragments.h"
#include <numeric>
#include <stdexcept>

// Logical GLM Flash index-cache reader component, not vendor page/footer ABI.
// Fixed SGLang 06992c92ad4494fe9f9636e75be3e35040512447:
// kpool_fp8_index.py:833-851,1139-1151: pooled F32 -> BF16 -> normalized
// Hadamard128 -> BF16 -> E4+one F32 scale per pool. Clamp absmax by floor1e-4,
// form the F32 reciprocal448 product, then round scale upward to a power of two.
// dsa_indexer.py:193-206 and dsa_indexer_kpool.py:170-175,593:
// rotate BF16 query, quantize128, fold qscale into the F32 head gate.
// deepseek_v2.py:1829 selects scale_fmt="ue8m0"; both key and query use that
// default rounded-scale reader. Stored logical scales below are F32 powers of
// two, not an assertion about the vendor's page/footer exponent-byte ABI.
// tilelang_kernel.py:1431-1487: code-domain dot, ReLU, head weighting/sum,
// then key scale. Ordinary typed ops below implement that logical math;
// they do not claim vendor tensor-core bit order or throughput.
namespace glm_official_index_cache {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace glm = d5_glm_remaining_fragments;
using Value = sem::ValueId;
inline constexpr std::int64_t kDepth = 128, kHeads = 32, kPool = 4;
inline constexpr std::int64_t kCapacity = 514, kTopPools = 512;
inline constexpr std::int64_t kLatent = 512, kMlaHeads = 64;
inline constexpr std::int64_t kSelected = kTopPools * kPool;
inline constexpr float kFloor = 1.0e-4F;

inline Value cast(sem::Fragment &f, Value x, c::DataType type) {
  return f.one(c::Operation(c::Cast{type}), {x});
}
inline Value bfloat_boundary(sem::Fragment &f, Value x) {
  return cast(f, cast(f, x, c::DataType::kBFloat16), c::DataType::kFloat32);
}
struct Quantized final { Value codes, scale, rotated, transformed, input; };
inline Quantized quantize(sem::Fragment &f, Value x, std::int64_t rows) {
  const auto &desc = f.description(x);
  if (desc.rank() != 2 || desc.dimensions()[0] != rows || desc.dimensions()[1] != kDepth)
    throw std::runtime_error("GLM quantizer requires the actual whole row extent");
  const auto input = cast(f, x, c::DataType::kFloat32);
  const auto transformed = f.one(c::Operation(c::Hadamard{}), {input});
  x = bfloat_boundary(f, transformed);
  const auto encoded = f.append(c::model_ops::float8_block_quantize(
      c::model_ops::Float8ScalePolicy::kPowerOfTwo, kFloor), {x}, 2);
  return {encoded[0], encoded[1], x, transformed, input};
}
inline std::vector<float> ape() {
  std::vector<float> result(kPool * kDepth);
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = float(int(i * 7 % 19) - 9) / 32.0F;
  return result;
}
struct Producer final {
  sem::Fragment fragment;
  std::int64_t pools;
  static constexpr std::size_t codes = 0, scales = 1, cursor = 2;
  static constexpr std::size_t new_codes = 3, new_scales = 4, pooled = 5, rotated = 6;
  static constexpr std::size_t transformed = 7;
  static constexpr std::size_t transform_input = 8;
};
inline Producer producer(std::int64_t pools) {
  if (pools <= 0 || pools > kTopPools) throw std::runtime_error("GLM cache producer pool extent");
  Producer r; r.pools = pools;
  auto &f = r.fragment;
  f.profile = &glm::core::profiles::glm53_flash_official;
  f.region = 0x474c4d4341435052ULL; f.quantized = false;
  const auto old_codes = f.input(sem::dense({kCapacity, kDepth}, c::DataType::kFloat8E4M3FN));
  const auto old_scales = f.input(sem::dense({kCapacity, 1}));
  const auto keys = f.input(sem::dense({pools, kPool, kDepth}, c::DataType::kBFloat16));
  const auto gates = f.input(sem::dense({pools, kPool, kDepth}));
  const auto start = f.input(sem::dense({1}, c::DataType::kInt32));
  const auto bias = f.constant(sem::dense({1, kPool, kDepth}), sem::bytes(ape()));
  auto logits = glm::binary(f, c::BinaryKind::kAdd, gates, bias);
  logits = f.one(c::Operation(c::Copy{}), {glm::permute(f, logits, {0, 2, 1})});
  auto probabilities = f.one(c::Operation(c::Softmax{}),
      {glm::reshape(f, logits, {pools * kDepth, kPool})});
  probabilities = f.one(c::Operation(c::Copy{}), {glm::permute(f,
      glm::reshape(f, probabilities, {pools, kDepth, kPool}), {0, 2, 1})});
  const auto pooled = glm::reduce(f, glm::binary(f, c::BinaryKind::kMultiply,
      probabilities, cast(f, keys, c::DataType::kFloat32)), 1);
  const auto encoded = quantize(f, bfloat_boundary(f, pooled), pools);
  const auto ones = f.constant(sem::dense({pools}, c::DataType::kInt32),
      sem::bytes(std::vector<std::int32_t>(std::size_t(pools), 1)));
  const auto positions = f.one(c::Operation(c::Scan{.axis = 0}),
      {f.one(c::Operation(c::Concat{.axis = 0}), {start, ones})});
  std::vector<std::int32_t> offsets(std::size_t(pools), 0);
  std::iota(offsets.begin(), offsets.end(), 0);
  const auto indices = f.one(c::Operation(c::Gather{}), {positions,
      f.constant(sem::dense({pools}, c::DataType::kInt32), sem::bytes(offsets))});
  const auto next_cursor = f.one(c::Operation(c::Gather{}), {positions,
      f.constant(sem::dense({1}, c::DataType::kInt32), sem::bytes(std::vector<std::int32_t>{std::int32_t(pools)}))});
  const auto codes = f.one(c::Operation(c::Scatter{.axis = 0}), {old_codes, indices, encoded.codes});
  const auto scales = f.one(c::Operation(c::Scatter{.axis = 0}), {old_scales, indices, encoded.scale});
  f.outputs = {codes, scales, next_cursor, encoded.codes, encoded.scale, pooled, encoded.rotated, encoded.transformed,
      f.one(c::Operation(c::Copy{}), {encoded.input})};
  return r;
}

struct Consumer final {
  sem::Fragment fragment;
  std::int64_t visible;
  static constexpr std::size_t ids = 0, attention = 1, scores = 2;
  static constexpr std::size_t query_codes = 3, query_scales = 4, query_rotated = 5, pool_ids = 6;
  static constexpr std::size_t query_transformed = 7;
  static constexpr std::size_t query_transform_input = 8;
};
// Raw F32 head weights are supplied before either 1/sqrt(heads) factor or
// quantization scale folding. Keep the original ordered F32 products shared
// by the standalone cache component and a complete NextN consumer.
inline Value score_encoded(sem::Fragment &f, Value bank, Value scales,
                           const Quantized &query, Value raw_weights,
                           std::int64_t visible) {
  std::vector<std::int32_t> prefix(std::size_t(visible), 0);
  std::iota(prefix.begin(), prefix.end(), 0);
  const auto valid = f.constant(sem::dense({visible}, c::DataType::kInt32), sem::bytes(prefix));
  const auto key = cast(f, f.one(c::Operation(c::Gather{}), {bank, valid}), c::DataType::kFloat32);
  const auto key_scale = glm::reshape(f, f.one(c::Operation(c::Gather{}), {scales, valid}), {1, visible});
  auto dot = f.one(c::Operation(c::MatMul{}),
      {cast(f, query.codes, c::DataType::kFloat32), glm::permute(f, key, {1, 0})});
  dot = glm::unary(f, c::UnaryKind::kRelu, dot);
  auto gate = glm::binary(f, c::BinaryKind::kMultiply, raw_weights,
      glm::filled(f, sem::dense({1}), 1.0F / std::sqrt(float(kHeads))));
  gate = glm::binary(f, c::BinaryKind::kMultiply,
      glm::reshape(f, gate, {kHeads, 1}), query.scale);
  gate = glm::binary(f, c::BinaryKind::kMultiply, gate,
      glm::filled(f, sem::dense({1}), 1.0F / std::sqrt(float(kDepth))));
  auto score = glm::reshape(f, glm::reduce(f,
      glm::binary(f, c::BinaryKind::kMultiply, dot, gate), 0), {1, visible});
  return glm::binary(f, c::BinaryKind::kMultiply, score, key_scale);
}
inline Consumer consumer(std::int64_t visible) {
  if (visible < kTopPools || visible > kCapacity) throw std::runtime_error("GLM cache consumer visible extent");
  Consumer r; r.visible = visible;
  auto &f = r.fragment;
  f.profile = &glm::core::profiles::glm53_flash_official;
  f.region = 0x474c4d434143434fULL; f.quantized = false;
  const auto bank = f.input(sem::dense({kCapacity, kDepth}, c::DataType::kFloat8E4M3FN));
  const auto scales = f.input(sem::dense({kCapacity, 1}));
  const auto query = f.input(sem::dense({kHeads, kDepth}, c::DataType::kBFloat16));
  const auto weights = f.input(sem::dense({kHeads}));
  const auto latent = f.input(sem::dense({kCapacity * kPool, kLatent}));
  const auto mla_query = f.input(sem::dense({1, 1, kMlaHeads, kLatent}));
  const auto q = quantize(f, query, kHeads);
  // Only logical E4+F32scale banks persist. These decoded tiles are temporary
  // scorer operands, not a second mutable F32 cache hidden behind Q/DQ.
  const auto score = score_encoded(f, bank, scales, q, weights, visible);
  const auto selected = f.append(c::Operation(c::TopK{.count = kTopPools}), {score}, 2);
  std::vector<std::int32_t> token_ids(std::size_t(kCapacity * kPool));
  std::iota(token_ids.begin(), token_ids.end(), 0);
  const auto expanded = f.one(c::Operation(c::Gather{}),
      {f.constant(sem::dense({kCapacity, kPool}, c::DataType::kInt32), sem::bytes(token_ids)), selected[1]});
  const auto ids = f.one(c::Operation(c::Copy{}), {glm::reshape(f, expanded, {kSelected})});
  const auto chosen = f.one(c::Operation(c::Gather{}), {latent, ids});
  const auto key_value = glm::reshape(f, chosen, {1, kSelected, 1, kLatent});
  const auto attention = f.one(c::Operation(c::ScaledDotProductAttention{
      .scale = 1.0F / 16.0F, .causal = true, .accumulation_type = c::DataType::kFloat32}),
      {mla_query, key_value, key_value});
  f.outputs = {ids, attention, score, q.codes, q.scale, q.rotated, selected[1], q.transformed,
      f.one(c::Operation(c::Copy{}), {q.input})};
  return r;
}

// Independent finite BF16/E4 nearest-even reference. No Hyle/device codec is
// called. E4 values are exactly representable as FP32; enumerate all127 codes.
inline std::uint16_t bfloat(float x) {
  const auto u = std::bit_cast<std::uint32_t>(x);
  return std::uint16_t((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
}
inline float wide(std::uint16_t x) { return std::bit_cast<float>(std::uint32_t(x) << 16); }
inline float code_value(unsigned code) {
  const unsigned magnitude = code & 127U, exponent = magnitude >> 3, mantissa = magnitude & 7U;
  const float value = exponent ? std::ldexp(float(8 + mantissa), int(exponent) - 10)
                               : std::ldexp(float(mantissa), -9);
  return code & 128U ? -value : value;
}
inline std::uint8_t code(float x) {
  const double value = std::min(double(std::abs(x)), 448.0);
  double distance = INFINITY; unsigned best = 0;
  for (unsigned candidate = 0; candidate < 127; ++candidate) {
    const double difference = std::abs(double(code_value(candidate)) - value);
    if (difference < distance || (difference == distance && !(candidate & 1U))) {
      distance = difference; best = candidate;
    }
  }
  return std::uint8_t(best | (std::signbit(x) ? 128U : 0U));
}
struct Encoded final {
  std::vector<std::uint8_t> codes;
  std::vector<float> scales, rotated, pooled;
};
// Scalar mathematical power-of-two ceiling; the native quantizer's exponent
// bit ladder is not reused. Keep the raw reciprocal448 product explicitly F32.
inline float power2_scale(float maximum) {
  maximum = std::max(maximum, kFloor);
  volatile float raw = float(double(maximum) * double(1.0F / 448.0F));
  int exponent = 0;
  const float fraction = std::frexp(float(raw), &exponent);
  return std::ldexp(1.0F, fraction == 0.5F ? exponent - 1 : exponent);
}
// Exact quantizer reference for its actual, separately checked BF16-rounded
// input. No transform or provider codec is reused here. This preserves the
// sign of zero; tiny legal F32 transform residuals are never snapped to zero.
inline Encoded encode_rotated(std::span<const float> rows, std::int64_t count) {
  if (rows.size() != std::size_t(count * kDepth)) throw std::runtime_error("rotated quantizer reference extent");
  Encoded r; r.codes.resize(rows.size()); r.rotated.assign(rows.begin(), rows.end());
  r.scales.resize(std::size_t(count));
  for (std::int64_t row = 0; row < count; ++row) {
    float maximum = kFloor;
    for (std::int64_t d = 0; d < kDepth; ++d)
      maximum = std::max(maximum, std::abs(rows[std::size_t(row * kDepth + d)]));
    const float scale = power2_scale(maximum);
    r.scales[std::size_t(row)] = scale;
    for (std::int64_t d = 0; d < kDepth; ++d)
      r.codes[std::size_t(row * kDepth + d)] = code(float(double(rows[std::size_t(row * kDepth + d)]) / scale));
  }
  return r;
}
inline Encoded encode_rows(std::span<const float> rows, std::int64_t count) {
  Encoded r; r.codes.resize(std::size_t(count * kDepth));
  r.rotated.resize(r.codes.size()); r.scales.resize(std::size_t(count));
  for (std::int64_t row = 0; row < count; ++row) {
    std::array<float, kDepth> x;
    for (std::size_t d = 0; d < x.size(); ++d) x[d] = wide(bfloat(rows[std::size_t(row * kDepth) + d]));
    for (std::size_t stride = 1; stride < x.size(); stride *= 2)
      for (std::size_t base = 0; base < x.size(); base += 2 * stride)
        for (std::size_t d = 0; d < stride; ++d) {
          const auto a = x[base + d], b = x[base + stride + d];
          x[base + d] = float(double(a) + b);
          x[base + stride + d] = float(double(a) - b);
        }
    float maximum = kFloor;
    for (std::size_t d = 0; d < x.size(); ++d) {
      x[d] = wide(bfloat(float(double(x[d]) * float(1.0 / std::sqrt(128.0)))));
      maximum = std::max(maximum, std::abs(x[d]));
      r.rotated[std::size_t(row * kDepth) + d] = x[d];
    }
    const float scale = power2_scale(maximum);
    r.scales[std::size_t(row)] = scale;
    for (std::size_t d = 0; d < x.size(); ++d)
      r.codes[std::size_t(row * kDepth) + d] = code(float(double(x[d]) / scale));
  }
  return r;
}
inline Encoded pool_reference(std::span<const std::uint16_t> keys,
                              std::span<const float> gates, std::int64_t pools) {
  if (keys.size() != std::size_t(pools * kPool * kDepth) || gates.size() != keys.size())
    throw std::runtime_error("GLM pooled cache reference extent");
  const auto bias = ape(); std::vector<float> pooled(std::size_t(pools * kDepth));
  for (std::int64_t row = 0; row < pools; ++row)
    for (std::int64_t d = 0; d < kDepth; ++d) {
      std::array<double, kPool> logits;
      for (std::int64_t slot = 0; slot < kPool; ++slot)
        logits[std::size_t(slot)] = double(gates[std::size_t((row * kPool + slot) * kDepth + d)]) + bias[std::size_t(slot * kDepth + d)];
      const double maximum = *std::max_element(logits.begin(), logits.end());
      double denominator = 0, numerator = 0;
      for (std::int64_t slot = 0; slot < kPool; ++slot) {
        const double weight = std::exp(logits[std::size_t(slot)] - maximum);
        denominator += weight;
        numerator += weight * wide(keys[std::size_t((row * kPool + slot) * kDepth + d)]);
      }
      pooled[std::size_t(row * kDepth + d)] = float(numerator / denominator);
    }
  auto r = encode_rows(pooled, pools); r.pooled = std::move(pooled); return r;
}
inline std::vector<float> score_reference(std::span<const std::uint8_t> keys,
    std::span<const float> scales, const Encoded &query,
    std::span<const float> weights, std::int64_t visible) {
  std::vector<float> result(std::size_t(visible), 0);
  for (std::int64_t row = 0; row < visible; ++row) {
    double sum = 0;
    for (std::int64_t head = 0; head < kHeads; ++head) {
      double dot = 0;
      for (std::int64_t d = 0; d < kDepth; ++d)
        dot += double(code_value(query.codes[std::size_t(head * kDepth + d)])) * code_value(keys[std::size_t(row * kDepth + d)]);
      float gate = float(double(weights[std::size_t(head)]) * (1.0F / std::sqrt(float(kHeads))));
      gate = float(double(gate) * query.scales[std::size_t(head)]);
      gate = float(double(gate) * (1.0F / std::sqrt(float(kDepth))));
      sum += float(double(std::max(float(dot), 0.0F)) * gate);
    }
    result[std::size_t(row)] = float(double(float(sum)) * scales[std::size_t(row)]);
  }
  return result;
}
inline std::vector<float> mla_reference(std::span<const std::int32_t> ids,
    std::span<const float> latent, std::span<const float> query) {
  std::vector<float> result(std::size_t(kMlaHeads * kLatent));
  for (std::int64_t head = 0; head < kMlaHeads; ++head) {
    std::vector<double> logits(ids.size()); double maximum = -INFINITY;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] < 0 || ids[i] >= kCapacity * kPool) throw std::runtime_error("GLM MLA reference invalid I32 id");
      double dot = 0;
      for (std::int64_t d = 0; d < kLatent; ++d)
        dot += double(query[std::size_t(head * kLatent + d)]) * latent[std::size_t(ids[i] * kLatent + d)];
      maximum = std::max(maximum, logits[i] = dot / 16.0);
    }
    double total = 0;
    for (auto &v : logits) total += v = std::exp(v - maximum);
    for (std::int64_t d = 0; d < kLatent; ++d) {
      double sum = 0;
      for (std::size_t i = 0; i < ids.size(); ++i) sum += logits[i] / total * latent[std::size_t(ids[i] * kLatent + d)];
      result[std::size_t(head * kLatent + d)] = float(sum);
    }
  }
  return result;
}
} // namespace glm_official_index_cache
