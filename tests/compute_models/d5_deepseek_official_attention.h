#pragma once

#include "d5_deepseek_official_fragments.h"
#include "d5_deepseek_remaining_fragments.h"
#include "d5_glm_remaining_fragments.h"

// Fixed official model.py consumers. All arithmetic here is either a public
// operation description or an independent test oracle; never a product kernel.
namespace d5_deepseek_official_attention {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace moe = d5_moe_execution_fragments;
namespace core = d5_moe_semantic_fragments;
namespace linear = d5_kda_full_fragments;
namespace fp8 = d5_dynamic_fp8_fragments;
namespace quant = d5_quantization_roundtrip;
namespace official = d5_deepseek_official_fragments;
namespace old = d5_deepseek_remaining_fragments;
namespace glm = d5_glm_remaining_fragments;
using sem::Fragment;
using sem::ValueId;
using sem::bytes;
using sem::dense;
using core::reshape;
using core::binary;
using core::filled;
using core::unary;
using old::slice;
using old::permute;
using old::concat;
using old::float_values;
using old::join_compressor;
using old::CompressorOptions;
using official::bfloat_boundary;

inline ValueId norm(Fragment &f, ValueId input, std::vector<float> &scales) {
  auto weight = linear::parameter(f, dense({std::int64_t(scales.size())}), scales, true);
  return bfloat_boundary(f, f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
                                 {input, weight}));
}
inline std::vector<float> norm_reference(std::span<const float> input,
                                         std::span<const float> scales) {
  return bfloat_boundary(sem::oracle::rms_norm_rows(input, scales, scales.size(), 1.0e-6F));
}
inline ValueId mean(Fragment &f, ValueId x) {
  const std::int32_t axis = std::int32_t(f.description(x).rank() - 1);
  return f.one(c::Operation(c::Reduce{.kind = c::ReductionKind::kMean,
      .axes = c::Axes::from(std::span(&axis, 1)).value(), .keep_dimensions = true,
      .accumulation_type = c::DataType::kFloat32}), {x});
}
inline ValueId query_norm(Fragment &f, ValueId x) {
  auto square = bfloat_boundary(f, binary(f, c::BinaryKind::kMultiply, x, x));
  auto variance = bfloat_boundary(f, mean(f, square));
  variance = bfloat_boundary(f, binary(f, c::BinaryKind::kAdd, variance,
                                      filled(f, dense({1}), 1.0e-6F)));
  auto inverse = bfloat_boundary(f, unary(f, c::UnaryKind::kRsqrt, variance));
  return bfloat_boundary(f, binary(f, c::BinaryKind::kMultiply, x, inverse));
}
inline std::vector<float> query_norm_reference(std::vector<float> x, std::int64_t depth) {
  for (std::size_t row = 0; row < x.size(); row += depth) {
    double squares = 0;
    for (std::int64_t i = 0; i < depth; ++i) squares += fp8::bfloat(x[row+i]*x[row+i]);
    const auto variance = fp8::bfloat(fp8::bfloat(float(squares/depth)) + 1.0e-6F);
    const auto inverse = fp8::bfloat(1.0F/std::sqrt(variance));
    for (std::int64_t i = 0; i < depth; ++i) x[row+i] = fp8::bfloat(x[row+i]*inverse);
  }
  return x;
}
inline std::vector<float> rope_frequencies(bool compressed, bool inverse) {
  const float base = compressed ? 160000.0F : 10000.0F;
  const auto correction = [&](double rotations) {
    return 64*std::log(65536/(rotations*2*std::acos(-1.0)))/(2*std::log(double(base)));
  };
  const auto low = std::max(0.0, std::floor(correction(32)));
  const auto high = std::min(63.0, std::ceil(correction(1)));
  std::vector<float> result(32);
  for (std::size_t i = 0; i < result.size(); ++i) {
    // Original torch F32 power/reciprocal and YaRN expression, not the
    // iterative frequency recurrence used by the separate GGUF consumer.
    auto frequency = 1.0F/std::pow(base, float(i)/32.0F);
    if (compressed) {
      const auto smooth = 1.0F-std::clamp((float(i)-float(low))/float(high == low ? 0.001 : high-low), 0.0F, 1.0F);
      frequency = frequency/16.0F*(1.0F-smooth)+frequency*smooth;
    }
    result[i] = inverse ? -frequency : frequency;
  }
  return result;
}
inline ValueId rope(Fragment &f, ValueId x, ValueId positions, bool compressed, bool inverse = false) {
  const auto depth = f.description(x).dimensions().back();
  auto prefix = slice(f, x, 3, 0, depth-64), tail = slice(f, x, 3, depth-64, 64);
  auto frequencies = f.constant(dense({32}), bytes(rope_frequencies(compressed, inverse)));
  auto rotated = f.one(c::Operation(c::RoPE{.rotary_dimensions = 64,
      .sequence_axis = 1, .layout = c::RopeLayout::kInterleaved}), {tail, positions, frequencies});
  return bfloat_boundary(f, concat(f, prefix, rotated, 3));
}
inline std::vector<float> rope_reference(std::span<const float> x, std::int64_t depth,
                                         std::int32_t position, bool compressed, bool inverse = false) {
  auto frequencies = rope_frequencies(compressed, inverse);
  std::vector<float> result(x.begin(), x.end());
  for (std::size_t row = 0; row < result.size(); row += depth)
    for (std::size_t pair = 0; pair < 32; ++pair) {
      const auto offset = row+depth-64+2*pair;
      const float angle = float(position)*frequencies[pair];
      const float cosine = std::cos(angle), sine = std::sin(angle);
      result[offset] = x[offset]*cosine-x[offset+1]*sine;
      result[offset+1] = x[offset]*sine+x[offset+1]*cosine;
    }
  return bfloat_boundary(std::move(result));
}
inline ValueId cache_quant(Fragment &f, ValueId x, bool indexer) {
  const auto width = f.description(x).dimensions()[1];
  if (indexer) {
    auto rotated = bfloat_boundary(f, f.one(c::Operation(c::Hadamard{}), {x}));
    return official::roundtrip(f, rotated, quant::Format::kFloat4E2M1Block32);
  }
  auto prefix = slice(f, x, 1, 0, width - 64);
  auto tail = slice(f, x, 1, width - 64, 64);
  return concat(f, official::roundtrip(f, prefix, quant::Format::kFloat8E4M3Block64), tail, 1);
}
inline std::vector<float> cache_reference(std::vector<float> x, std::int64_t depth, bool indexer) {
  if (indexer)
    return quant::reference(bfloat_boundary(old::hadamard_reference(x, depth)), true);
  for (std::size_t row = 0; row < x.size(); row += depth) {
    const auto prefix = quant::reference(std::vector<float>(x.begin()+row, x.begin()+row+depth-64), false);
    std::copy(prefix.begin(), prefix.end(), x.begin()+row);
  }
  return x;
}

// Original checkpoint BF16 compressor weights are loaded into F32 parameters.
// The fixture performs that lossless widening once, before either entry runs.
inline linear::Projection compressor_projection(Fragment &f, ValueId x,
                                                 std::int64_t n, std::int64_t k, int seed) {
  auto w = sem::weight(n, k, c::GgufQuantizedType::kQ8_0, true, seed);
  // Dense zero bytes already represent both BF16->F32 zeros. Convert the
  // fixture's nonzero source terms once; do not round billions of host zeros
  // while constructing full-vocabulary shards. The uploaded bytes are equal.
  for (std::size_t row = 0; row < w.decoded.size(); ++row)
    for (auto &[column, value] : w.decoded[row]) {
      value = fp8::bfloat(value);
      std::memcpy(w.storage.data()+(row*k+column)*sizeof(float), &value, sizeof(float));
    }
  auto stored = f.constant(w.description, std::move(w.storage));
  auto output = f.one(c::Operation(c::MatMul{}), {x, stored});
  return {std::move(w), linear::Storage::kF32, output};
}

inline ValueId compress(Fragment &f, ValueId previous_kv, ValueId previous_score,
                         ValueId current_kv, ValueId current_score,
                         ValueId weight, ValueId position, std::int64_t ratio,
                         std::int64_t depth, bool indexer) {
  if (ratio == 4) {
    previous_kv = slice(f, previous_kv, 1, 0, depth);
    previous_score = slice(f, previous_score, 1, 0, depth);
    current_kv = slice(f, current_kv, 1, depth, depth);
    current_score = slice(f, current_score, 1, depth, depth);
  }
  auto values = permute(f, concat(f, previous_kv, current_kv, 0), {1, 0});
  auto scores = permute(f, concat(f, previous_score, current_score, 0), {1, 0});
  auto weights = f.one(c::Operation(c::Softmax{.axis = -1}), {scores});
  auto pooled = bfloat_boundary(f, reshape(f,
      moe::sum(f, binary(f, c::BinaryKind::kMultiply, values, weights), -1), {1, depth}));
  auto normalized = bfloat_boundary(f, f.one(c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
                                              {pooled, weight}));
  auto rotated = reshape(f, rope(f, reshape(f, normalized, {1, 1, 1, depth}), position, true), {1, depth});
  return cache_quant(f, rotated, indexer);
}

inline std::vector<float> compress_reference(
    std::span<const float> previous_kv, std::span<const float> previous_score,
    std::span<const float> current_kv, std::span<const float> current_score,
    std::span<const float> scales, std::int64_t ratio, std::int32_t position, bool indexer) {
  const auto depth = std::int64_t(scales.size()), channels = ratio == 4 ? 2*depth : depth;
  const auto rows = ratio == 4 ? 4 : 64;
  std::vector<float> reduced(depth);
  for (std::int64_t channel = 0; channel < depth; ++channel) {
    double maximum = -std::numeric_limits<double>::infinity();
    const auto other = channel + (ratio == 4 ? depth : 0);
    for (std::int64_t row = 0; row < rows; ++row) {
      maximum = std::max(maximum, double(previous_score[row*channels+channel]));
      maximum = std::max(maximum, double(current_score[row*channels+other]));
    }
    double numerator = 0, denominator = 0;
    for (std::int64_t row = 0; row < rows; ++row) {
      const auto old_index = row*channels+channel, new_index = row*channels+other;
      const auto a = std::exp(double(previous_score[old_index])-maximum);
      const auto b = std::exp(double(current_score[new_index])-maximum);
      numerator += a*previous_kv[old_index]+b*current_kv[new_index];
      denominator += a+b;
    }
    reduced[channel] = fp8::bfloat(float(numerator/denominator));
  }
  return cache_reference(rope_reference(norm_reference(reduced, scales), depth, position, true), depth, indexer);
}

// Batch recipes below retain actual output-to-input state feedback.
inline Fragment compressor(const moe::Case &model, std::int64_t ratio,
                             bool indexer = false, CompressorOptions options = {}) {
  if ((ratio != 4 && ratio != 128) || (indexer && ratio != 4))
    throw std::runtime_error("invalid DeepSeek compressor role");
  if (model.gguf) throw std::runtime_error("official compressor profile required");
  const std::int64_t depth = indexer ? 128 : 512;
  const auto channels = ratio == 4 ? 2 * depth : depth;
  const std::int64_t chunk = ratio == 4 ? 4 : 64;
  Fragment f{.profile = model.profile,
             .region = std::uint64_t(indexer ? 0x080a3403 :
                                    ratio == 4 ? 0x080a3401 : 0x080a3402),
             .bfloat_chain_outputs = {0}};
  std::optional<linear::Projection> kv_projection, gate_projection;
  ValueId new_kv, new_gate;
  auto hidden = f.input(dense({chunk, model.hidden}));
  kv_projection = compressor_projection(f, bfloat_boundary(f, hidden), channels, model.hidden, 251);
  gate_projection = compressor_projection(f, bfloat_boundary(f, hidden), channels, model.hidden, 257);
  new_kv = kv_projection->output;
  new_gate = gate_projection->output;
  const auto past_kv_port = f.inputs.size();
  auto past_kv = f.input(dense({chunk, channels}));
  const auto past_score_port = f.inputs.size();
  auto past_score = f.input(dense({chunk, channels}));
  const auto cache_port = f.inputs.size();
  auto cache = f.input(dense({1, depth}));
  auto positions = f.input(dense({1}, c::DataType::kInt32));
  auto ape_rows = f.input(dense({chunk}, c::DataType::kInt32));
  auto publish = f.input(dense({1}, c::DataType::kBool));
  std::vector<float> ape(ratio * channels), norm(depth);
  for (std::size_t i = 0; i < ape.size(); ++i)
    ape[i] = float(std::int64_t((i * 7 + 3) % 23) - 11) / 16.0F;
  for (std::int64_t i = 0; i < depth; ++i)
    norm[i] = 0.75F + float(i % 7) / 16.0F;
  auto stored_ape = f.constant(dense({ratio, channels}), bytes(ape));
  auto selected_ape = f.one(c::Operation(c::Gather{}), {stored_ape, ape_rows});
  auto new_score = binary(f, c::BinaryKind::kAdd, new_gate, selected_ape);
  auto compressed = compress(f, past_kv, past_score, new_kv, new_score,
                              f.constant(dense({depth}), bytes(norm)), positions,
                              ratio, depth, indexer);
  auto next_cache = f.one(c::Operation(c::Select{}), {publish, compressed, cache});
  f.outputs = {next_cache, f.one(c::Operation(c::Copy{}), {new_kv}), new_score};
  f.state_ports = {{past_kv_port, 1}, {past_score_port, 2}, {cache_port, 0}};

  std::vector<float> old_kv(chunk * channels), old_score(chunk * channels), old_cache(depth);
  for (int step = 0; step < 6; ++step) {
    const bool reset = step % 3 == 0;
    const bool update = ratio == 4 || step % 3 != 1;
    if (reset) {
      for (std::size_t i = 0; i < old_kv.size(); ++i) {
        old_kv[i] = ratio == 4 && step == 0 && options.first_block == 0 ? 0.0F
            : float(std::int64_t((i * 3 + step) % 31) - 15) / 16.0F;
        old_score[i] = ratio == 4 && step == 0 && options.first_block == 0
            ? -std::numeric_limits<float>::infinity()
            : float(std::int64_t((i * 5 + step) % 29) - 14) / 8.0F;
      }
      for (std::size_t i = 0; i < old_cache.size(); ++i)
        old_cache[i] = float(std::int64_t((i + step) % 17) - 8) / 16.0F;
    }
    sem::Replay replay{};
    replay.reset_state = reset;
    std::vector<float> kv, score;
    auto hidden_values = d5_gguf_test::activation(chunk * model.hidden, 11 + 13 * step);
    if (options.positive_hidden)
      for (auto &value : hidden_values) value = std::abs(value) + 0.5F;
    hidden_values = bfloat_boundary(std::move(hidden_values));
    replay.inputs.push_back(bytes(hidden_values));
    kv = old::project_rows(*kv_projection, hidden_values, model.hidden);
    score = old::project_rows(*gate_projection, hidden_values, model.hidden);
    // Only reset inputs carry initial state. A missing actual-output feedback
    // must fail conspicuously, rather than silently using the oracle's state.
    replay.inputs.push_back(bytes(reset ? old_kv : std::vector<float>(old_kv.size(), 23)));
    replay.inputs.push_back(bytes(reset ? old_score : std::vector<float>(old_score.size(), -19)));
    replay.inputs.push_back(bytes(reset ? old_cache : std::vector<float>(old_cache.size(), 31)));
    const auto position = std::int32_t((step >= 3 ? options.reset_position_offset : 0) +
        options.first_block * ratio +
        (ratio == 4 ? 4 * (step % 3 + 1) - 1 : 127 + 64 * (step % 3)));
    std::vector<std::int32_t> indices(chunk);
    for (std::int64_t row = 0; row < chunk; ++row) {
      indices[row] = std::int32_t((position - chunk + 1 + row) % ratio);
      for (std::int64_t channel = 0; channel < channels; ++channel)
        score[row * channels + channel] += ape[indices[row] * channels + channel];
    }
    // The pinned cache planner rotates at the compressed block's START,
    // whereas APE indexes each original token's position modulo the ratio.
    const auto compressed_position = std::int32_t((position / ratio) * ratio);
    replay.inputs.push_back(bytes(std::vector<std::int32_t>{compressed_position}));
    replay.inputs.push_back(bytes(indices));
    replay.inputs.push_back(bytes(std::vector<std::uint8_t>{std::uint8_t(update)}));
    if (update)
      old_cache = compress_reference(old_kv, old_score, kv, score, norm, ratio,
                                      compressed_position, indexer);
    replay.outputs = {old_cache, kv, score};
    f.replays.push_back(std::move(replay));
    old_kv = std::move(kv);
    old_score = std::move(score);
  }
  return f;
}

struct PreparedOutputWeight final { sem::Weight weight; ValueId value; };
inline PreparedOutputWeight prepare_output_weight(Fragment &consumer, std::int64_t groups) {
  constexpr std::int64_t n_per_group = 1024, k = 4096;
  const auto n = groups*n_per_group;
  auto cold = std::make_shared<Fragment>(Fragment{.profile = consumer.profile, .region = 0x0e0a3401});
  auto encoded = c::float8_weight_description(k, n, c::Float8ScaleMode::kFloat32Block128).value();
  std::vector<std::byte> storage(encoded.required_bytes().value());
  sem::Weight weight{dense({groups, n_per_group, k}, c::DataType::kBFloat16), {},
      std::vector<std::vector<std::pair<std::size_t, float>>>(n)};
  std::vector<float> expected(n*k);
  const d5_official_fp8_fragments::Case shape{consumer.profile, n, k, true, "wo_a"};
  for (std::int64_t row = 0; row < n/128; ++row)
    for (std::int64_t block = 0; block < k/128; ++block) {
      const auto scale = d5_official_fp8_fragments::scale(shape, row, block, 283);
      std::memcpy(storage.data()+n*k+(row*(k/128)+block)*4, &scale, 4);
    }
  for (std::int64_t row = 0; row < n; ++row)
    for (int term = 0; term < 2; ++term) {
      const auto col = (row*17+283+term*2069)%k;
      const auto code = std::size_t((row+term*3)%8);
      storage[row*k+col] = std::byte{d5_official_fp8_fragments::kCodes[code]};
      const auto value = fp8::bfloat(d5_official_fp8_fragments::kValues[code]*
          d5_official_fp8_fragments::scale(shape, row/128, col/128, 283));
      expected[row*k+col] = value;
      weight.decoded[row].emplace_back(std::size_t(col), value);
    }
  auto stored = cold->constant(encoded, std::move(storage));
  auto decoded = cold->one(c::model_ops::float8_weight_decode(), {stored});
  cold->outputs = {cold->one(c::Operation(c::Copy{}), {reshape(*cold, decoded, {groups, n_per_group, k})})};
  cold->replays.push_back({{}, {std::move(expected)}});
  consumer.preparation = std::move(cold);
  consumer.prepared_constants = {consumer.constants.size()};
  auto value = consumer.constant(weight.description, {});
  return {std::move(weight), value};
}

using WindowReference = std::function<moe::ReferenceOutputs(
    std::span<const float>, std::span<const float>, std::int32_t)>;

inline Fragment attention(const moe::Case &model, std::int64_t ratio, WindowReference *reference = nullptr) {
  if (model.gguf || (ratio != 0 && ratio != 4 && ratio != 128) || (reference && ratio != 0))
    throw std::runtime_error("official attention profile or reference mismatch");
  const std::int64_t heads = model.experts == 384 ? 128 : 64;
  const std::int64_t rank = model.experts == 384 ? 1536 : 1024;
  const std::int64_t top = model.experts == 384 ? 1024 : 512;
  const std::int64_t groups = heads / 8;
  constexpr std::int64_t depth = 512, group_width = 4096, out_rank = 1024;
  const std::int64_t chunk = ratio == 4 ? 4 : ratio == 128 ? 64 : 1;
  const std::int64_t capacity = ratio == 4 ? top + 4 : ratio == 128 ? 20 : 0;
  const auto initial_blocks = capacity ? capacity - 3 : 0;
  const CompressorOptions options{std::int32_t(initial_blocks), 0, true};
  Fragment f{.profile = model.profile, .region = 0x080a3404,
             .quantized = true};
  if (ratio) {
    f = d5_deepseek_official_attention::compressor(model, ratio, false, options);
    if (ratio == 4) join_compressor(f, d5_deepseek_official_attention::compressor(model, ratio, true, options));
    f.region = ratio == 4 ? 0x080a3405 : 0x080a3406;
  } else {
    (void)f.input(dense({1, model.hidden}));
    for (int step = 0; step < 6; ++step) {
      auto hidden = d5_gguf_test::activation(model.hidden, 11 + 13 * step);
      for (auto &value : hidden) value = std::abs(value) + 0.5F;
      hidden = bfloat_boundary(std::move(hidden));
      sem::Replay replay{{bytes(hidden)}, {}};
      replay.reset_state = step % 3 == 0;
      f.replays.push_back(std::move(replay));
    }
  }
  // Assigning the compressor recipe must not discard this complete chain's
  // predeclared quantized accuracy class. Standalone F32 compressor outputs
  // continue to use their original tight gate in compressor().
  f.quantized = true;
  const auto raw_port = f.inputs.size();
  auto raw_past = f.input(dense({127, depth}));
  auto positions = f.input(dense({chunk}, c::DataType::kInt32));
  auto query_position = slice(f, positions, 0, chunk - 1, 1);
  auto hidden = slice(f, f.inputs[0], 0, chunk - 1, 1);
  auto qa = fp8::model_linear(f, hidden, rank, model.hidden, 263, true);
  std::vector<float> qnorm(rank), knorm(depth), sinks(heads);
  for (std::int64_t i = 0; i < rank; ++i) qnorm[i] = 0.75F + float(i % 7) / 32.0F;
  for (std::int64_t i = 0; i < depth; ++i) knorm[i] = 0.875F + float(i % 5) / 32.0F;
  for (std::int64_t i = 0; i < heads; ++i) sinks[i] = 0.75F + float(i % 11) / 16.0F;
  auto qr = norm(f, qa.output, qnorm);
  auto qb = fp8::model_linear(f, qr, heads * depth, rank, 269, true);
  auto q = query_norm(f, reshape(f, qb.output, {1, 1, heads, depth}));
  q = rope(f, q, query_position, ratio != 0);
  auto kv = fp8::model_linear(f, f.inputs[0], depth, model.hidden, 271, true);
  auto key = norm(f, kv.output, knorm);
  key = reshape(f, rope(f, reshape(f, key, {1, chunk, 1, depth}),
                             positions, ratio != 0), {chunk, depth});
  key = cache_quant(f, key, false);
  auto raw_all = concat(f, raw_past, key, 0);
  auto raw_window = slice(f, raw_all, 0, chunk - 1, 128);
  auto next_raw = slice(f, raw_all, 0, chunk, 127);
  const auto raw_output = f.outputs.size();
  f.outputs.push_back(next_raw);
  f.state_ports.push_back({raw_port, raw_output});
  auto keys = raw_window;
  auto mask = core::filled(f, dense({1, 128}), 0);
  std::size_t comp_port = 0, comp_output = 0, index_port = 0, index_output = 0;
  std::size_t scores_output = 0, ids_output = 0;
  std::optional<fp8::ModelLinear> index_q;
  std::optional<linear::Projection> index_weight;
  if (ratio) {
    comp_port = f.inputs.size();
    auto old_compressed = f.input(dense({capacity, depth}));
    auto write_mask = f.input(dense({capacity, 1}, c::DataType::kBool));
    auto visibility = f.input(dense({1, capacity}));
    auto compressed = f.one(c::Operation(c::Select{}),
                             {write_mask, f.outputs[0], old_compressed});
    comp_output = f.outputs.size();
    f.outputs.push_back(compressed);
    f.state_ports.push_back({comp_port, comp_output});
    auto compressed_mask = visibility;
    if (ratio == 4) {
      index_port = f.inputs.size();
      auto old_index = f.input(dense({capacity, 128}));
      auto index_keys = f.one(c::Operation(c::Select{}),
                               {write_mask, f.outputs[3], old_index});
      index_output = f.outputs.size();
      f.outputs.push_back(index_keys);
      f.state_ports.push_back({index_port, index_output});
      index_q = fp8::model_linear(f, qr, 64 * 128, rank, 277, true);
      auto iq = rope(f, reshape(f, index_q->output, {1, 1, 64, 128}),
                          query_position, true);
      iq = cache_quant(f, reshape(f, iq, {64, 128}), true);
      index_weight = linear::projection(f, hidden, 64, model.hidden, linear::Storage::kBFloat16, 281);
      auto weight = bfloat_boundary(f, binary(f, c::BinaryKind::kMultiply,
          bfloat_boundary(f, index_weight->output), filled(f, dense({1}), 1.0F / std::sqrt(8192.0F))));
      auto dot = bfloat_boundary(f, f.one(c::Operation(c::MatMul{}), {iq, permute(f, index_keys, {1, 0})}));
      auto positive = f.one(c::model_ops::clamp(0, std::numeric_limits<float>::max()), {dot});
      auto weighted = bfloat_boundary(f, binary(f, c::BinaryKind::kMultiply, positive, reshape(f, weight, {64, 1})));
      auto scores = bfloat_boundary(f, reshape(f, moe::sum(f, weighted, 0), {1, capacity}));
      auto selected = f.append(c::Operation(c::TopK{.count = std::size_t(top)}),
          {binary(f, c::BinaryKind::kAdd, scores, visibility)}, 2);
      scores_output = f.outputs.size();
      f.outputs.push_back(f.one(c::Operation(c::Copy{}), {scores}));
      ids_output = f.outputs.size();
      f.outputs.push_back(selected[1]);
      f.unordered_ids = Fragment::UnorderedIds{ids_output, std::size_t(top), capacity, true};
      compressed = reshape(f, f.one(c::Operation(c::Gather{}), {compressed, selected[1]}), {top, depth});
      compressed_mask = reshape(f, f.one(c::Operation(c::Gather{}),
          {reshape(f, visibility, {capacity, 1}), selected[1]}), {1, top});
    }
    keys = concat(f, keys, compressed, 0);
    mask = concat(f, mask, compressed_mask, 1);
  }
  const auto key_count = 128 + (ratio == 4 ? top : capacity);
  keys = reshape(f, keys, {1, key_count, 1, depth});
  const float scale = 1.0F / std::sqrt(float(depth));
  auto context = bfloat_boundary(f, f.one(c::model_ops::sink_attention(scale),
      {q, keys, keys, mask, f.constant(dense({heads}), bytes(sinks))}));
  auto derotated = rope(f, context, query_position, ratio != 0, true);
  auto grouped = reshape(f, derotated, {1, groups, group_width});
  auto prepared = prepare_output_weight(f, groups);
  auto wa = std::move(prepared.weight);
  std::vector<std::int32_t> group_ids(groups);
  std::iota(group_ids.begin(), group_ids.end(), 0);
  auto ids = f.constant(dense({1, groups}, c::DataType::kInt32), bytes(group_ids));
  auto oa = bfloat_boundary(f, f.one(c::model_ops::grouped_matmul(),
      {prepared.value, grouped, ids}));
  auto ob = fp8::model_linear(f, reshape(f, oa, {1, groups * out_rank}),
                                 model.hidden, groups * out_rank, 293, true);
  const auto context_output = f.outputs.size();
  f.outputs.push_back(context);
  f.outputs.push_back(derotated);
  f.outputs.push_back(oa);
  f.outputs.push_back(ob.output);
  if (reference) {
    *reference = [qa, qb, kv, ob, wa, qnorm, knorm, sinks, group_ids]
        (std::span<const float> input, std::span<const float> past, std::int32_t position) -> moe::ReferenceOutputs {
      auto x = bfloat_boundary(std::vector<float>(input.begin(), input.end()));
      auto qr = norm_reference(qa.reference(x), qnorm);
      auto q = rope_reference(query_norm_reference(qb.reference(qr), 512), 512, position, false);
      auto k = cache_reference(rope_reference(norm_reference(kv.reference(x), knorm), 512, position, false), 512, false);
      std::vector<float> keys(past.begin(), past.end());
      keys.insert(keys.end(), k.begin(), k.end());
      auto context = bfloat_boundary(old::sink_reference(q, keys, keys, std::vector<float>(128), sinks, 1, 128, 1.0F/std::sqrt(512.0F)));
      auto inverse = rope_reference(context, 512, position, false, true);
      auto oa = bfloat_boundary(moe::project(inverse, wa, group_ids, 1024, 4096));
      auto out = ob.reference(oa);
      keys.erase(keys.begin(), keys.begin()+512);
      return {std::move(keys), std::move(context), std::move(inverse), std::move(oa), std::move(out)};
    };
    return f;
  }

  std::vector<float> raw_history(127 * depth), comp_history(capacity * depth);
  std::vector<float> index_history(ratio == 4 ? capacity * 128 : 0);
  std::int64_t visible = initial_blocks;
  for (std::size_t step = 0; step < f.replays.size(); ++step) {
    auto &replay = f.replays[step];
    if (replay.reset_state) {
      visible = initial_blocks;
      for (std::size_t i = 0; i < raw_history.size(); ++i)
        raw_history[i] = float(std::sin(double(i + step * 17) * 0.017) * 0.5);
      for (std::int64_t row = 0; row < capacity; ++row)
        for (std::int64_t d = 0; d < depth; ++d)
          comp_history[row * depth + d] = row < initial_blocks
              ? float(std::sin(double(row * 19 + d + step * 11) * 0.021) * 0.75) : 13.0F;
      for (std::int64_t row = 0; row < (ratio == 4 ? capacity : 0); ++row)
        for (std::int64_t d = 0; d < 128; ++d)
          index_history[row * 128 + d] = row < initial_blocks
              ? (d == 0 ? float(row + 1) / 64.0F : 0.0F) : 32.0F;
    }
    if (replay.reset_state) {
      raw_history = cache_reference(bfloat_boundary(std::move(raw_history)), depth, false);
      if (capacity) comp_history = cache_reference(bfloat_boundary(std::move(comp_history)), depth, false);
      if (ratio == 4) index_history = quant::reference(bfloat_boundary(std::move(index_history)), true);
    }
    auto x = bfloat_boundary(float_values(replay.inputs[0]));
    auto last = std::span<const float>(x).last(model.hidden);
    const auto position = std::int32_t(ratio
        ? initial_blocks * ratio + (ratio == 4 ? 4 * (step % 3 + 1) - 1 : 127 + 64 * (step % 3))
        : 127 + step % 3);
    std::vector<std::int32_t> token_positions(chunk);
    for (std::int64_t t = 0; t < chunk; ++t) token_positions[t] = std::int32_t(position - chunk + 1 + t);
    replay.inputs.push_back(bytes(replay.reset_state ? raw_history : std::vector<float>(raw_history.size(), 23)));
    replay.inputs.push_back(bytes(token_positions));
    auto qr_values = norm_reference(qa.reference(last), qnorm);
    auto query = query_norm_reference(qb.reference(qr_values), depth);
    query = rope_reference(query, depth, position, ratio != 0);
    auto new_keys = norm_reference(kv.reference(x), knorm);
    for (std::int64_t t = 0; t < chunk; ++t) {
      auto row = rope_reference(std::span<const float>(new_keys).subspan(t * depth, depth),
                                depth, token_positions[t], ratio != 0);
      std::copy(row.begin(), row.end(), new_keys.begin() + t * depth);
    }
    new_keys = cache_reference(std::move(new_keys), depth, false);
    auto full_raw = raw_history;
    full_raw.insert(full_raw.end(), new_keys.begin(), new_keys.end());
    std::vector<float> attended_keys(full_raw.end() - 128 * depth, full_raw.end());
    raw_history.assign(full_raw.end() - 127 * depth, full_raw.end());
    replay.outputs.resize(f.outputs.size());
    replay.outputs[raw_output] = raw_history;
    std::vector<float> attention_mask(128, 0);
    if (ratio) {
      replay.inputs.push_back(bytes(replay.reset_state ? comp_history : std::vector<float>(comp_history.size(), -23)));
      const bool update = ratio == 4 || step % 3 != 1;
      std::vector<std::uint8_t> writing(capacity);
      if (update) {
        writing[visible] = 1;
        std::copy(replay.outputs[0].begin(), replay.outputs[0].end(), comp_history.begin() + visible * depth);
        ++visible;
      }
      std::vector<float> visibility(capacity, -std::numeric_limits<float>::infinity());
      std::fill_n(visibility.begin(), visible, 0.0F);
      replay.inputs.push_back(bytes(writing));
      replay.inputs.push_back(bytes(visibility));
      replay.outputs[comp_output] = comp_history;
      std::vector<std::int32_t> selected;
      if (ratio == 4) {
        replay.inputs.push_back(bytes(replay.reset_state ? index_history : std::vector<float>(index_history.size(), 29)));
        std::copy(replay.outputs[3].begin(), replay.outputs[3].end(), index_history.begin() + (visible - 1) * 128);
        replay.outputs[index_output] = index_history;
        auto iq = cache_reference(rope_reference(index_q->reference(qr_values), 128, position, true), 128, true);
        auto iw = bfloat_boundary(index_weight->reference(last));
        for (auto &value : iw) value = fp8::bfloat(value * (1.0F / std::sqrt(8192.0F)));
        std::vector<float> scores(capacity);
        for (std::int64_t row = 0; row < capacity; ++row) {
          double sum = 0;
          for (std::int64_t h = 0; h < 64; ++h) {
            double dot = 0;
            for (std::int64_t d = 0; d < 128; ++d)
              dot += double(iq[h * 128 + d]) * index_history[row * 128 + d];
            sum += fp8::bfloat(std::max(0.0F, fp8::bfloat(float(dot))) * iw[h]);
          }
          scores[row] = fp8::bfloat(float(sum));
        }
        selected.resize(visible);
        std::iota(selected.begin(), selected.end(), 0);
        std::stable_sort(selected.begin(), selected.end(), [&](auto a, auto b) { return scores[a] > scores[b]; });
        selected.resize(top);
        replay.outputs[scores_output] = scores;
        for (auto id : selected) replay.outputs[ids_output].push_back(static_cast<float>(id));
      } else {
        selected.resize(capacity);
        std::iota(selected.begin(), selected.end(), 0);
      }
      for (auto row : selected) {
        attended_keys.insert(attended_keys.end(), comp_history.begin() + row * depth,
                              comp_history.begin() + (row + 1) * depth);
        attention_mask.push_back(visibility[row]);
      }
    }
    auto context_values = bfloat_boundary(old::sink_reference(query, attended_keys, attended_keys, attention_mask,
                                          sinks, 1, key_count, scale));
    auto derotated_values = rope_reference(context_values, depth, position, ratio != 0, true);
    auto grouped_values = bfloat_boundary(moe::project(derotated_values, wa, group_ids, out_rank, group_width));
    replay.outputs[context_output] = std::move(context_values);
    replay.outputs[context_output + 1] = std::move(derotated_values);
    replay.outputs[context_output + 2] = grouped_values;
    replay.outputs[context_output + 3] = ob.reference(grouped_values);
  }
  return f;
}
struct Hyper final {
  ValueId collapsed, post, combination;
  linear::Projection fn;
  std::int64_t hidden;
  std::vector<float> scales{0.125F, 0.25F, 0.5F}, bases = std::vector<float>(24);
  moe::ReferenceOutputs reference(std::span<const float> state, std::span<const float> branch) const {
    auto mixes = fn.reference(state);
    double square = 0;
    for (auto x : state) square += double(x)*x;
    const float inverse = 1.0F/std::sqrt(float(square/state.size())+1.0e-6F);
    for (auto &x : mixes) x *= inverse;
    auto result = core::hyper_reference(mixes, state, branch, hidden, scales, bases);
    result[1] = bfloat_boundary(std::move(result[1]));
    result[2] = bfloat_boundary(std::move(result[2]));
    return result;
  }
};
inline ValueId inverse_rms(Fragment &f, ValueId input) {
  auto square = binary(f, c::BinaryKind::kMultiply, input, input);
  return unary(f, c::UnaryKind::kRsqrt, binary(f, c::BinaryKind::kAdd,
      mean(f, square), filled(f, dense({1}), 1.0e-6F)));
}
inline Hyper hyper_pre(Fragment &f, ValueId state, std::int64_t hidden, int seed) {
  auto flat = reshape(f, state, {1, 4*hidden});
  auto fn = linear::projection(f, flat, 24, 4*hidden, linear::Storage::kF32, seed);
  auto mixes = binary(f, c::BinaryKind::kMultiply, fn.output, inverse_rms(f, flat));
  Hyper result{{}, {}, {}, std::move(fn), hidden};
  for (int i = 0; i < 24; ++i) result.bases[i] = float(i%7-3)/8.0F;
  auto scale = f.constant(dense({3}), bytes(result.scales));
  auto base = f.constant(dense({24}), bytes(result.bases));
  result.combination = f.one(c::model_ops::hyper_connection_combine(1.0e-6F, 20), {mixes, scale, base});
  const auto gate = [&](int start, float multiplier, float offset) {
    auto part = slice(f, mixes, 1, start, 4);
    auto affine = binary(f, c::BinaryKind::kAdd,
        binary(f, c::BinaryKind::kMultiply, part, filled(f, dense({1}), result.scales[start/4])),
        f.constant(dense({1, 4}), bytes(std::vector<float>(result.bases.begin()+start, result.bases.begin()+start+4))));
    auto sigmoid = unary(f, c::UnaryKind::kSigmoid, affine);
    return binary(f, c::BinaryKind::kAdd,
        binary(f, c::BinaryKind::kMultiply, sigmoid, filled(f, dense({1}), multiplier)),
        filled(f, dense({1}), offset));
  };
  result.post = gate(4, 2, 0);
  result.collapsed = bfloat_boundary(f, f.one(c::model_ops::hyper_connection_pre(), {state, gate(0, 1, 1.0e-6F)}));
  return result;
}
inline ValueId hyper_post(Fragment &f, const Hyper &mix, ValueId state, ValueId branch) {
  return bfloat_boundary(f, f.one(c::model_ops::hyper_connection_post(), {branch, state, mix.post, mix.combination}));
}

// The fixed official config appends ratio0 for mtp.0 on both Pro and Flash.
// MTP has two distinct FP8 projections and a BF16 sum, then a complete mHC
// attention + MoE decoder; it is not GLM's concatenated EH projection.
inline Fragment mtp(const moe::Case &model, bool native_routing = true) {
  if (model.gguf) throw std::runtime_error("official MTP profile required");
  Fragment f{.profile = model.profile, .region = 0x0d0a3401, .quantized = true};
  auto hidden = f.input(dense({1, 4, model.hidden}));
  auto embedding = f.input(dense({1, model.hidden}));
  auto raw = f.input(dense({127, 512}));
  auto position = f.input(dense({1}, c::DataType::kInt32));
  auto hs = sem::oracle::scale(model.hidden), es = hs, ans = hs, fns = hs, final_scale = hs;
  std::reverse(es.begin(), es.end());
  auto hn = norm(f, bfloat_boundary(f, hidden), hs);
  auto en = norm(f, bfloat_boundary(f, embedding), es);
  auto hp = fp8::model_linear(f, reshape(f, hn, {4, model.hidden}), model.hidden, model.hidden, 317, true);
  auto ep = fp8::model_linear(f, en, model.hidden, model.hidden, 331, true);
  auto mixed = bfloat_boundary(f, binary(f, c::BinaryKind::kAdd,
      reshape(f, hp.output, {1, 4, model.hidden}), reshape(f, ep.output, {1, 1, model.hidden})));
  auto a = hyper_pre(f, mixed, model.hidden, 307);
  auto attn_input = norm(f, a.collapsed, ans);
  WindowReference attention_reference;
  auto recipe = attention(model, 0, &attention_reference);
  auto attn = glm::attach(f, std::move(recipe), {attn_input, raw, position});
  auto after_attention = hyper_post(f, a, mixed, attn.back());
  auto m = hyper_pre(f, after_attention, model.hidden, 313);
  auto moe_input = norm(f, m.collapsed, fns);
  moe::MoeReference moe_reference;
  auto moe_recipe = official::moe_chain(model, false, &moe_reference, native_routing);
  auto moe_outputs = glm::attach(f, std::move(moe_recipe), {moe_input});
  auto after_moe = hyper_post(f, m, after_attention, moe_outputs[5]);
  auto flat = reshape(f, after_moe, {1, 4*model.hidden});
  auto head_fn = linear::projection(f, flat, 4, 4*model.hidden, linear::Storage::kF32, 337);
  const std::vector<float> head_base{-0.25F, 0.125F, 0.375F, -0.5F};
  auto head_mix = binary(f, c::BinaryKind::kMultiply, head_fn.output, inverse_rms(f, flat));
  auto gate = binary(f, c::BinaryKind::kAdd,
      unary(f, c::UnaryKind::kSigmoid, binary(f, c::BinaryKind::kAdd,
          binary(f, c::BinaryKind::kMultiply, head_mix, filled(f, dense({1}), 0.125F)),
          f.constant(dense({4}), bytes(head_base)))), filled(f, dense({1}), 1.0e-6F));
  auto head_hidden = bfloat_boundary(f, f.one(c::model_ops::hyper_connection_pre(), {after_moe, gate}));
  auto normalized = norm(f, head_hidden, final_scale);
  f.outputs = {after_moe, attn[0], mixed, attn_input, attn.back(), moe_input, moe_outputs[5], head_hidden, normalized};
  f.state_ports = {{0, 0}, {2, 1}};
  std::vector<float> state, history;
  for (int step = 0; step < 6; ++step) {
    const bool reset = step%3 == 0;
    if (reset) {
      state = bfloat_boundary(d5_gguf_test::values(4, model.hidden, 31+step));
      history.resize(127*512);
      for (std::size_t i = 0; i < history.size(); ++i) history[i] = float(std::sin(double(i+step*17)*0.017)*0.5);
      history = cache_reference(bfloat_boundary(std::move(history)), 512, false);
    }
    auto e = bfloat_boundary(d5_gguf_test::activation(model.hidden, 41+step*13));
    const std::int32_t pos = (step >= 3 ? 65536 : 127)+step%3;
    sem::Replay replay{{bytes(reset ? state : std::vector<float>(state.size(), 37)), bytes(e),
                       bytes(reset ? history : std::vector<float>(history.size(), -19)),
                       bytes(std::vector<std::int32_t>{pos})}, {}};
    replay.reset_state = reset;
    auto mixed_values = hp.reference(norm_reference(state, hs));
    auto embedding_values = ep.reference(norm_reference(e, es));
    for (std::size_t i = 0; i < mixed_values.size(); ++i)
      mixed_values[i] = fp8::bfloat(mixed_values[i]+embedding_values[i%model.hidden]);
    auto pre_a = a.reference(mixed_values, std::vector<float>(model.hidden));
    auto av = norm_reference(pre_a[1], ans);
    auto attn_values = attention_reference(av, history, pos);
    auto after_a = a.reference(mixed_values, attn_values.back())[2];
    auto pre_m = m.reference(after_a, std::vector<float>(model.hidden));
    auto mv = norm_reference(pre_m[1], fns);
    auto moe_values = moe_reference(mv);
    auto after_m = m.reference(after_a, moe_values[5])[2];
    auto head_mix_values = head_fn.reference(after_m);
    double square = 0;
    for (auto x : after_m) square += double(x)*x;
    const float inverse = 1.0F/std::sqrt(float(square/after_m.size())+1.0e-6F);
    std::vector<float> collapsed(model.hidden);
    for (std::int64_t d = 0; d < model.hidden; ++d) {
      double sum = 0;
      for (int stream = 0; stream < 4; ++stream)
        sum += after_m[stream*model.hidden+d]*(core::sigmoid(head_mix_values[stream]*inverse*0.125F+head_base[stream])+1.0e-6);
      collapsed[d] = fp8::bfloat(float(sum));
    }
    auto final = norm_reference(collapsed, final_scale);
    replay.outputs = {after_m, attn_values[0], mixed_values, av, attn_values.back(), mv, moe_values[5], collapsed, final};
    f.replays.push_back(std::move(replay));
    state = std::move(after_m);
    history = std::move(attn_values[0]);
  }
  return f;
}

template <class Run> void check_mtp_head(const moe::Case &model, Run run,
                                       bool native_routing = true) {
  auto recipe = mtp(model, native_routing);
  std::vector<std::vector<float>> observed, reference;
  for (const auto &replay : recipe.replays) reference.push_back(replay.outputs.back());
  recipe.observe = [&](std::size_t port, std::span<const float> actual) {
    if (port == 8) observed.emplace_back(actual.begin(), actual.end());
  };
  run(std::move(recipe));
  ASSERT_FALSE(::testing::Test::HasFailure());
  ASSERT_EQ(observed.size(), reference.size());
  // Original lm_head: BF16 checkpoint -> F32 parameter, BF16 normalized input
  // widened to F32, and F32 logits. Keep all 129280 rows, bounded resident shards.
  constexpr std::int64_t vocabulary = 129280, shard = 4096;
  std::int64_t checked = 0;
  for (std::int64_t first = 0; first < vocabulary; first += shard) {
    const auto count = std::min(shard, vocabulary-first);
    Fragment head{.profile = model.profile, .region = 0x010a3401, .quantized = true};
    auto input = head.input(dense({1, model.hidden}));
    auto projection = compressor_projection(head, input, count, model.hidden, int(347+first));
    head.outputs = {projection.output};
    std::vector<std::vector<float>> local;
    for (std::size_t step = 0; step < observed.size(); ++step) {
      head.replays.push_back({{bytes(observed[step])}, {projection.reference(reference[step])}});
      local.push_back(projection.reference(observed[step]));
    }
    head.observe = [local = std::move(local), step = std::size_t{0}]
        (std::size_t, std::span<const float> actual) mutable {
      sem::oracle::expect_close(actual, local.at(step++), 8.0e-5F, 4.0e-5F);
    };
    run(std::move(head));
    ASSERT_FALSE(::testing::Test::HasFailure());
    checked += count;
  }
  ASSERT_EQ(checked, vocabulary);
}
} // namespace d5_deepseek_official_attention
