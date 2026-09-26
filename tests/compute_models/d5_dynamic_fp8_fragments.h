#pragma once

#include <bit>
#include <memory>

#include "d5_official_fp8_fragments.h"

namespace d5_dynamic_fp8_fragments {
namespace fp8 = d5_official_fp8_fragments;
namespace c = fp8::c;
namespace sem = fp8::sem;
using Rounding = c::model_ops::Float8ScaleRounding;

inline float bfloat(float x) { return float(sem::q::decode(sem::q::encode(x))); }
inline float fp8_round(float x) {
  float best = 0, error = std::numeric_limits<float>::infinity();
  unsigned best_code = 0;
  for (unsigned code = 0; code < 127; ++code) {
    const auto exponent = code >> 3, mantissa = code & 7;
    const float value = exponent == 0 ? std::ldexp(float(mantissa), -9)
        : std::ldexp(1.0F + float(mantissa) / 8, int(exponent) - 7);
    const float distance = std::abs(std::abs(x) - value);
    if (distance < error || (distance == error && (code & 1) == 0 && (best_code & 1))) {
      best = value; best_code = code; error = distance;
    }
  }
  return std::copysign(best, x);
}

inline std::vector<double> quantized(std::span<const float> input,
                                    std::int64_t k, bool power2) {
  std::vector<double> result(input.size());
  for (std::size_t begin = 0; begin < input.size(); begin += 128) {
    float maximum = power2 ? 1.0e-4F : 1.0e-10F;
    for (std::size_t i = 0; i < 128; ++i)
      maximum = std::max(maximum, std::abs(input[begin + i]));
    float scale = maximum / 448.0F;
    if (power2) {
      const auto bits = std::bit_cast<std::uint32_t>(scale);
      const auto exponent = (bits >> 23) + ((bits & 0x7fffff) != 0);
      scale = std::bit_cast<float>(exponent << 23);
    }
    for (std::size_t i = 0; i < 128; ++i)
      result[begin + i] = double(fp8_round(input[begin + i] / scale)) * scale;
  }
  (void)k;
  return result;
}

inline std::vector<float> reference(const fp8::Case &shape,
                                    std::span<const float> input,
                                    bool bf16, bool power2) {
  auto encoded = quantized(input, shape.k, power2);
  const auto m = std::int64_t(input.size()) / shape.k;
  std::vector<float> result(m * shape.n);
  for (std::int64_t row = 0; row < m; ++row)
    for (std::int64_t out = 0; out < shape.n; ++out) {
      double sum = 0;
      for (int term = 0; term < 2; ++term) {
        const auto col = (out * 17 + 31 + term * (shape.k / 2 + 13)) % shape.k;
        const double weight = double(fp8::kValues[(out + term * 3) % 8]) *
                              fp8::scale(shape, out / 128, col / 128, 13);
        sum += double(encoded[row * shape.k + col]) * weight;
      }
      result[row * shape.n + out] = bf16 ? bfloat(float(sum)) : float(sum);
    }
  return result;
}

inline std::vector<float> input_values(std::int64_t m, std::int64_t k, int seed, bool bf16) {
  std::vector<float> result(m * k);
  for (std::int64_t row = 0; row < m; ++row)
    for (std::int64_t col = 0; col < k; ++col) {
      const float scale = std::ldexp(1.0F, int((row + col / 128 + seed) % 4) - 7);
      float value = fp8::kValues[(row * 3 + col + seed) % 8] * scale;
      if (col % 128 == 0) value = 448 * scale;
      if (col < 128) value = 0;
      else if (col < 256) value = 1.0e-5F;
      result[row * k + col] = bf16 ? bfloat(value) : value;
    }
  return result;
}

inline std::vector<std::byte> input_bytes(std::span<const float> values, bool bf16) {
  if (!bf16) return sem::bytes(values);
  std::vector<hyle::simd::bfloat16_t> stored;
  for (float value : values) stored.push_back(sem::q::encode(value));
  return sem::bytes(stored);
}
inline sem::ValueId weight(sem::Fragment &fragment, const fp8::Case &shape) {
  auto fixture = fp8::projection(shape, 1);
  return fragment.constant(fp8::descriptions(shape, 1)[1],
                            std::move(fixture.constant_bytes[0]));
}
inline c::Operation operation(bool power2) {
  return c::model_ops::float8_block_linear(power2 ? Rounding::kPowerOfTwo : Rounding::kNone,
                                          power2 ? 1.0e-4F : 1.0e-10F);
}
inline sem::ValueId visible(sem::Fragment &f, sem::ValueId value, bool bf16) {
  return bf16 ? f.one(c::Operation(c::Cast{.output_type = c::DataType::kFloat32}), {value}) : value;
}

inline void strict_output_check(sem::Fragment &f, bool bf16, int ulps = 1) {
  std::vector<std::vector<std::vector<float>>> reference;
  for (const auto &replay : f.replays) reference.push_back(replay.outputs);
  // This local checker is tighter than the shared full-fragment BF16 gate.
  f.observe = [expected = std::move(reference), iteration = std::size_t{0}, bf16, ulps]
      (std::size_t port, std::span<const float> actual) mutable {
    const auto &target = expected.at(iteration).at(port);
    ASSERT_EQ(actual.size(), target.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
      if (bf16 && actual[i] != target[i]) {
        const auto a = std::bit_cast<std::uint32_t>(actual[i]) >> 16;
        const auto b = std::bit_cast<std::uint32_t>(target[i]) >> 16;
        EXPECT_LE(std::abs(std::int64_t(a) - b), ulps) << i << " output=" << port;
      } else if (!bf16) {
        EXPECT_NEAR(actual[i], target[i], 2.0e-6F + 2.0e-6F * std::abs(target[i])) << i;
      }
    }
    if (port + 1 == expected[iteration].size()) ++iteration;
  };
  if (bf16)
    for (std::size_t i = 0; i < f.outputs.size(); ++i) f.bfloat_chain_outputs.push_back(i);
  else f.absolute_tolerance = 2.0e-6F;
}

inline sem::Fragment projection(const fp8::Case &shape, std::int64_t m,
                                 bool bf16, bool power2) {
  sem::Fragment f{.profile = shape.profile, .region = 0x0b0604};
  const auto type = bf16 ? c::DataType::kBFloat16 : c::DataType::kFloat32;
  auto input = f.input(sem::dense({m, shape.k}, type));
  auto output = f.one(operation(power2), {input, weight(f, shape)});
  f.outputs = {visible(f, output, bf16)};
  for (int seed : {0, 1}) {
    const auto values = input_values(m, shape.k, seed, bf16);
    f.replays.push_back({{input_bytes(values, bf16)}, {reference(shape, values, bf16, power2)}});
  }
  strict_output_check(f, bf16);
  return f;
}

// Test consumer for arbitrary official FP8 linear roles. The encoded bytes
// stay on the selected route; the independent host math is an oracle only.
struct ModelLinear final {
  sem::ValueId output;
  std::int64_t n, k;
  int seed;
  std::vector<float> reference(std::span<const float> input,
                               bool declared_f32_accumulation = true) const {
    std::vector<float> rounded(input.begin(), input.end());
    for (auto &value : rounded) value = bfloat(value);
    const auto quant = quantized(rounded, k, false);
    std::vector<float> scales(input.size() / 128, 1.0e-10F);
    for (std::size_t i = 0; i < rounded.size(); ++i)
      scales[i / 128] = std::max(scales[i / 128], std::abs(rounded[i]));
    for (auto &scale : scales) scale /= 448.0F;
    std::vector<float> result(input.size() / k * n);
    const fp8::Case shape{nullptr, n, k, false, "official linear"};
    for (std::size_t token = 0; token < input.size() / k; ++token)
      for (std::int64_t row = 0; row < n; ++row) {
        double sum = 0;
        std::array<std::pair<std::int64_t, double>, 2> terms;
        for (int term = 0; term < 2; ++term) {
          const auto col = (row * 17 + seed + term * (k / 2 + 13)) % k;
          const auto coefficient = fp8::kValues[(row + seed + term) % 4];
          sum += quant[token * k + col] * coefficient *
                 fp8::scale(shape, row / 128, col / 128, seed);
          // The decoded code*scale product is exact in double. Recovering
          // its code keeps the independent dot separate from F32 scaling.
          terms[term] = {col / 128,
              quant[token * k + col] / scales[(token * k + col) / 128] * coefficient};
        }
        if (declared_f32_accumulation) {
          // Match the declared F32 accumulator and the mature block128
          // program's rounding boundaries, not an unrounded F64 matrix.
          // Codes/weights are decoded independently; no GPU result is used.
          if (terms[1].first < terms[0].first) std::swap(terms[0], terms[1]);
          float accumulated = 0;
          for (std::size_t i = 0; i < terms.size();) {
            const auto block = terms[i].first;
            double dot = terms[i++].second;
            while (i < terms.size() && terms[i].first == block) dot += terms[i++].second;
            const float scaled = float(dot * scales[token * (k / 128) + block]);
            accumulated = std::fma(scaled, fp8::scale(shape, row / 128, block, seed), accumulated);
          }
          sum = accumulated;
        }
        result[token * n + row] = bfloat(float(sum));
      }
    return result;
  }
};
inline ModelLinear model_linear(sem::Fragment &f, sem::ValueId input,
                                std::int64_t n, std::int64_t k, int seed) {
  const auto description = c::float8_weight_description(k, n, c::Float8ScaleMode::kFloat32Block128).value();
  std::vector<std::byte> storage(description.required_bytes().value());
  const fp8::Case shape{f.profile, n, k, false, "official linear"};
  for (std::int64_t row = 0; row < n; ++row)
    for (int term = 0; term < 2; ++term) {
      const auto col = (row * 17 + seed + term * (k / 2 + 13)) % k;
      storage[row * k + col] = std::byte{fp8::kCodes[(row + seed + term) % 4]};
    }
  for (std::int64_t row = 0; row < (n + 127) / 128; ++row)
    for (std::int64_t block = 0; block < k / 128; ++block) {
      const float value = fp8::scale(shape, row, block, seed);
      std::memcpy(storage.data() + n * k + (row * (k / 128) + block) * 4, &value, 4);
    }
  input = f.one(c::Operation(c::Cast{.output_type = c::DataType::kBFloat16}), {input});
  const auto weight = f.constant(description, std::move(storage));
  auto output = f.one(operation(false), {input, weight});
  return {visible(f, output, true), n, k, seed};
}

inline sem::Fragment model_accumulation_boundaries() {
  sem::Fragment f{.profile = fp8::kCases[6].profile, .region = 0x0b0606};
  auto input = f.input(sem::dense({1, 4096}));
  auto projected = model_linear(f, input, 1536, 4096, 197);
  f.outputs = {projected.output};
  for (int seed : {3, 17, 31, 45}) {
    auto x = d5_gguf_test::activation(4096, seed);
    x = sem::oracle::rms_norm_rows(x, sem::oracle::scale(4096), 4096, 1.0e-5F);
    auto expected = projected.reference(x);
    const auto unrounded = projected.reference(x, false);
    // These real consumer inputs exercise the missing rounding boundary;
    // an all-double chain is a different numerical program.
    EXPECT_FALSE(std::equal(expected.begin(), expected.end(), unrounded.begin()));
    f.replays.push_back({{sem::bytes(x)}, {std::move(expected)}});
  }
  strict_output_check(f, true, 1);
  return f;
}

inline sem::ValueId glm_gated(sem::Fragment &f, sem::ValueId packed, bool flash) {
  if (flash) {
    auto plain = visible(f, packed, true);
    const auto desc = f.description(plain);
    const auto axis = static_cast<std::int32_t>(desc.rank() - 1);
    const auto width = desc.dimensions().back() / 2;
    const auto half = [&](std::int64_t first) {
      auto view = f.one(c::Operation(c::View::slice(axis, first, width).value()), {plain});
      return f.one(c::Operation(c::Copy{}), {view});
    };
    auto gate = f.one(c::model_ops::clamp(-std::numeric_limits<float>::infinity(), 10.0F), {half(0)});
    auto up = f.one(c::model_ops::clamp(-10.0F, 10.0F), {half(width)});
    auto joined = f.one(c::Operation(c::Concat{.axis = axis}), {gate, up});
    packed = f.one(c::Operation(c::Cast{c::DataType::kBFloat16}), {joined});
  }
  return f.one(c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kPackedHalves}), {packed});
}

inline sem::Fragment glm_shared_ffn(std::int64_t m, bool flash = false) {
  const auto &up_shape = fp8::kCases[flash ? 6 : 4];
  const auto &down_shape = fp8::kCases[flash ? 7 : 5];
  const fp8::Case packed{up_shape.profile, up_shape.n * 2, up_shape.k, false, "packed gate/up"};
  sem::Fragment f{.profile = up_shape.profile, .region = 0x0b0605};
  auto input = f.input(sem::dense({m, packed.k}, c::DataType::kBFloat16));
  auto gate_up = f.one(operation(false), {input, weight(f, packed)});
  auto activated = glm_gated(f, gate_up, flash);
  auto output = f.one(operation(false), {activated, weight(f, down_shape)});
  f.outputs = {visible(f, gate_up, true), visible(f, activated, true), visible(f, output, true)};
  for (int seed : {0, 1}) {
    auto values = input_values(m, packed.k, seed, true);
    auto projected = reference(packed, values, true, false);
    std::vector<float> activation(m * up_shape.n);
    for (std::int64_t row = 0; row < m; ++row)
      for (std::int64_t col = 0; col < up_shape.n; ++col) {
        double gate = projected[row * packed.n + col];
        double up = projected[row * packed.n + up_shape.n + col];
        if (flash) { gate = std::min(gate, 10.0); up = std::clamp(up, -10.0, 10.0); }
        activation[row * up_shape.n + col] = bfloat(float(gate / (1 + std::exp(-gate)) * up));
      }
    auto result = reference(down_shape, activation, true, false);
    f.replays.push_back({{input_bytes(values, true)}, {std::move(projected), std::move(activation), std::move(result)}});
  }
  strict_output_check(f, true, 1);
  return f;
}
} // namespace d5_dynamic_fp8_fragments
