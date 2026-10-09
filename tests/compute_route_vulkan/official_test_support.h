#pragma once

#include "mlx_test_support.h"
#include "../compute_models/d5_quantization_roundtrip.h"

namespace vulkan_official_test {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace t = vulkan_tensor_test;

inline float fp8_round(float x) {
  float best = 0, distance = std::numeric_limits<float>::infinity();unsigned chosen = 0;
  for (unsigned code = 0; code < 127; ++code) {
    const auto value = vulkan_mlx_test::fp8(static_cast<std::uint8_t>(code));
    const auto delta = std::abs(std::abs(x) - value);
    if (delta < distance || (delta == distance && (code & 1) == 0 && (chosen & 1))) {
      distance = delta;best = value;chosen = code;
    }
  }
  return std::copysign(best, x);
}

struct Weight final {
  int e, n, k, group;
  bool four;
  c::TensorDesc description;
  std::vector<std::byte> bytes;
  std::vector<float> values, scales;
  Weight(int experts, int columns, int depth, bool fp4 = false, int seed = 0)
      : e(experts), n(columns), k(depth), group(fp4 ? 32 : 128), four(fp4),
        description(fp4 ? c::float4_grouped_weight_description(e, n, k).value()
                    : e > 1 ? c::float8_grouped_weight_description(e, n, k).value()
                            : c::float8_weight_description(k, n, c::Float8ScaleMode::kFloat32Block128).value()),
        bytes(*description.required_bytes()), values(e * n * k),
        scales(e * (fp4 ? n : (n + 127) / 128) * (k / group)) {
    constexpr std::array<std::uint8_t, 8> codes{0, 0x30, 0x38, 0x40, 0xb0, 0xb8, 0xc0, 0x48};
    constexpr std::array<float, 16> fp4_values{0, .5, 1, 1.5, 2, 3, 4, 6, -0.0f, -.5, -1, -1.5, -2, -3, -4, -6};
    const auto payload = e * n * k / (fp4 ? 2 : 1);
    for (std::size_t i = 0; i < scales.size(); ++i) scales[i] = .0625f * (1 + (i * 3 + seed) % 7);
    std::memcpy(bytes.data() + payload, scales.data(), scales.size() * sizeof(float));
    for (int ex = 0; ex < e; ++ex) for (int row = 0; row < n; ++row) for (int j = 0; j < k; ++j) {
      const auto index = (ex * n + row) * k + j;
      const auto code = static_cast<std::uint8_t>((index * 7 + row * 3 + seed * 5) % (fp4 ? 16 : 8));
      if (fp4) {
        const auto old = std::to_integer<unsigned>(bytes[index / 2]);
        bytes[index / 2] = std::byte(old | (code << (4 * (index % 2))));values[index] = fp4_values[code];
      } else { bytes[index] = std::byte{codes[code]};values[index] = vulkan_mlx_test::fp8(codes[code]); }
    }
  }
  float scale(int expert, int row, int block) const {
    return scales[(expert * (four ? n : (n + 127) / 128) + (four ? row : row / 128)) * (k / group) + block];
  }
};

inline std::vector<float> activation(int rows, int k, c::DataType type, int seed) {
  std::vector<float> values(rows * k);
  for (int r = 0; r < rows; ++r) for (int j = 0; j < k; ++j) {
    float value = static_cast<float>((r * 17 + j * 3 + seed * 7) % 19 - 9) / 64.f;
    if (j % 128 == 0) value = 3.5f + .25f * seed;
    if (j / 128 % 3 == 1) value = static_cast<float>(j % 7 - 3) * 1e-5f;
    values[r * k + j] = static_cast<float>(t::rounded(value, type));
  }
  return values;
}

struct Quantized final { std::vector<float> codes, scales; };
inline Quantized quantize(std::span<const float> values, int k, float floor, bool power) {
  Quantized result{std::vector<float>(values.size()), std::vector<float>(values.size() / 128)};
  for (std::size_t base = 0; base < values.size(); base += 128) {
    float maximum = floor;
    for (int i = 0; i < 128; ++i) maximum = std::max(maximum, std::abs(values[base + i]));
    const float raw = power ? maximum * (1.f / 448.f) : maximum / 448.f;
    const float scale = power ? static_cast<float>(std::exp2(std::ceil(std::log2(double(raw))))) : raw;
    result.scales[base / 128] = scale;
    for (int i = 0; i < 128; ++i) result.codes[base + i] = fp8_round(std::clamp(values[base + i] / scale, -448.f, 448.f));
  }
  (void)k;return result;
}

inline std::vector<double> reference(const Weight &w, const Quantized &a, int tokens,
    int activations, int slots, std::span<const std::int32_t> ids, c::DataType type) {
  std::vector<double> result(tokens * slots * w.n);
  for (int token = 0; token < tokens; ++token) for (int slot = 0; slot < slots; ++slot) {
    const auto row = token * slots + slot, ar = token * activations + (activations == 1 ? 0 : slot);
    const auto expert = ids.empty() ? 0 : ids[row];
    for (int out = 0; out < w.n; ++out) {
      float sum = 0;
      for (int block = 0; block < w.k / w.group; ++block) {
        float dot = 0;
        for (int j = 0; j < w.group; ++j) {
          const auto col = block * w.group + j;
          dot += a.codes[ar * w.k + col] * w.values[(expert * w.n + out) * w.k + col];
        }
        const auto a_scale = a.scales[ar * (w.k / 128) + block / (128 / w.group)];
        const float scaled = dot * a_scale;
        sum += scaled * w.scale(expert, out, block);
      }
      result[row * w.n + out] = t::rounded(sum, type);
    }
  }
  return result;
}

template <class Prepare>
void linears(Prepare prepare) {
  auto context = vulkan_test::context().value();
  for (auto type : {c::DataType::kFloat32, c::DataType::kBFloat16})
    for (bool power : {false, true}) for (int n : {5, 129}) {
      SCOPED_TRACE(::testing::Message() << "linear/" << static_cast<int>(type) << '/' << power << '/' << n);
      const int m = 3, k = 384;Weight first(1, n, k), second(1, n, k, false, 1);
      const auto ad = t::dense(type, {m, k});const auto floor = power ? 1e-4f : 1e-10f;
      const auto operation = c::model_ops::float8_block_linear(power ? c::model_ops::Float8ScaleRounding::kPowerOfTwo : c::model_ops::Float8ScaleRounding::kNone, floor);
      const std::array descriptions{ad, first.description};auto program = prepare(context, operation, descriptions);
      ASSERT_TRUE(program) << program.error().message();auto slot = program->create_invocation().value();
      for (int pass = 0; pass < 2; ++pass) {
        auto x = activation(m, k, type, pass);std::vector<double> data(x.begin(), x.end());
        const auto &w = pass ? second : first;
        const std::array inputs{t::make(context, ad, data), context.upload(w.description, w.bytes).value()};
        ASSERT_TRUE(slot.bind(inputs).ok());auto work = slot.submit().value();auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        t::expect(context, result->get()->tensors()[0], reference(w, quantize(x, k, floor, power), m, 1, 1, {}, type));
      }
    }
}

template <class Prepare>
void grouped(Prepare prepare) {
  auto context = vulkan_test::context().value();
  for (bool four : {false, true}) for (auto type : {c::DataType::kFloat32, c::DataType::kBFloat16}) {
    if (four && type != c::DataType::kBFloat16) continue;
    for (int activations : {1, 2}) {
      SCOPED_TRACE(::testing::Message() << "grouped/" << four << '/' << static_cast<int>(type) << '/' << activations);
      const int tokens = 2, slots = 2, k = 256, n = 7;Weight first(3, n, k, four), second(3, n, k, four, 1);
      const auto ad = t::dense(type, {tokens, activations, k}), id = t::dense(c::DataType::kInt32, {tokens, slots});
      const auto operation = four ? c::model_ops::grouped_float4_linear() : c::model_ops::grouped_float8_linear();
      const std::array descriptions{first.description, ad, id};auto program = prepare(context, operation, descriptions);
      ASSERT_TRUE(program) << program.error().message();auto slot = program->create_invocation().value();
      for (int pass = 0; pass < 2; ++pass) {
        const std::array<std::int32_t, 4> ids = pass ? std::array<std::int32_t, 4>{2, 2, 0, 1} : std::array<std::int32_t, 4>{0, 1, 2, 2};
        const auto x = activation(tokens * activations, k, type, pass);std::vector<double> values(x.begin(), x.end());
        const auto &w = pass ? second : first;
        const std::array inputs{context.upload(w.description, w.bytes).value(), t::make(context, ad, values), context.upload(id, std::as_bytes(std::span(ids))).value()};
        ASSERT_TRUE(slot.bind(inputs).ok());auto work = slot.submit().value();auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        t::expect(context, result->get()->tensors()[0], reference(w, quantize(x, k, four ? 1e-4f : 1e-10f, four), tokens, activations, slots, ids, type));
      }
      auto x = activation(tokens * activations, k, type, 0);std::vector<double> values(x.begin(), x.end());
      const std::array<std::int32_t, 4> bad{0, 1, -1, 2};
      const std::array inputs{context.upload(first.description, first.bytes).value(), t::make(context, ad, values), context.upload(id, std::as_bytes(std::span(bad))).value()};
      ASSERT_TRUE(slot.bind(inputs).ok());auto work = slot.submit().value();auto result = work.wait();
      ASSERT_FALSE(result);EXPECT_TRUE(result.error().IsOutOfRange());
    }
  }
}

template <class Prepare>
void decodes(Prepare prepare) {
  auto context = vulkan_test::context().value();
  for (int n : {5, 129}) for (auto type : {c::DataType::kFloat32, c::DataType::kBFloat16}) {
    SCOPED_TRACE(::testing::Message() << "decode/" << static_cast<int>(type) << '/' << n);
    Weight data(1, n, 256);auto program = prepare(context, c::model_ops::float8_weight_decode(type), std::span(&data.description, 1));
    ASSERT_TRUE(program) << program.error().message();auto slot = program->create_invocation().value();
    ASSERT_TRUE(slot.bind(context.upload(data.description, data.bytes).value()).ok());
    auto work = slot.submit().value();auto result = work.wait();ASSERT_TRUE(result) << result.error().message();
    std::vector<double> expected(n * 256);
    for (int row = 0; row < n; ++row) for (int col = 0; col < 256; ++col)
      expected[row * 256 + col] = t::rounded(data.values[row * 256 + col] * data.scale(0, row, col / 128), type);
    t::expect(context, result->get()->tensors()[0], expected, true);
  }
}

template <class Prepare>
void roundtrips(Prepare prepare) {
  auto context = vulkan_test::context().value();
  std::vector<std::uint16_t> all;
  for (unsigned bits = 0; bits < 65536; ++bits) if ((bits & 0x7f80) != 0x7f80) all.push_back(static_cast<std::uint16_t>(bits));
  for (auto format : {c::model_ops::BlockQuantizationFormat::kFloat4E2M1Block32,
                     c::model_ops::BlockQuantizationFormat::kFloat8E4M3Block64,
                     c::model_ops::BlockQuantizationFormat::kFloat8E4M3Block128}) {
    SCOPED_TRACE(::testing::Message() << "roundtrip/" << static_cast<int>(format));
    const bool four = format == c::model_ops::BlockQuantizationFormat::kFloat4E2M1Block32;
    const int group = four ? 32 : format == c::model_ops::BlockQuantizationFormat::kFloat8E4M3Block64 ? 64 : 128;
    const auto description = t::dense(c::DataType::kBFloat16, {1, static_cast<std::int64_t>(all.size())});
    auto program = prepare(context, c::model_ops::block_quantization_roundtrip(format), std::span(&description, 1));
    ASSERT_TRUE(program) << program.error().message();auto slot = program->create_invocation().value();
    auto values = all;
    for (int pass = 0; pass < 2; ++pass) {
      std::vector<float> decoded;for (auto bits : values) decoded.push_back(std::bit_cast<float>(std::uint32_t{bits} << 16));
      const auto expected = d5_quantization_roundtrip::reference(decoded, four, group);
      ASSERT_TRUE(slot.bind(context.upload(description, std::as_bytes(std::span(values))).value()).ok());
      auto work = slot.submit().value();auto result = work.wait();ASSERT_TRUE(result) << result.error().message();
      const auto actual = t::values(context, result->get()->tensors()[0]);ASSERT_EQ(actual.size(), expected.size());
      for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_EQ(actual[i], expected[i]) << i;
      std::reverse(values.begin(), values.end());
    }
    values[0] = 0x7fc0;
    ASSERT_TRUE(slot.bind(context.upload(description, std::as_bytes(std::span(values))).value()).ok());
    auto work = slot.submit().value();auto result = work.wait();ASSERT_FALSE(result);EXPECT_TRUE(result.error().IsInvalidArgument());
  }
}

template <class Prepare>
void lifecycle(Prepare prepare) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 512 * 1024 * 1024);
  {
    auto context = vulkan_test::context(&quota).value();Weight w(1, 5, 256);
    const auto ad = t::dense(c::DataType::kBFloat16, {2, 256});const std::array descriptions{ad, w.description};
    const auto operation = c::model_ops::float8_block_linear(c::model_ops::Float8ScaleRounding::kPowerOfTwo, 1e-4f);
    auto program = prepare(context, operation, descriptions).value();
    { auto reservation = quota.try_reserve_bytes(quota.limit() - quota.used());ASSERT_TRUE(reservation);
      auto denied = program.create_invocation();ASSERT_FALSE(denied);EXPECT_TRUE(denied.error().IsResourceExhausted()); }
    auto first = program.create_invocation().value(), second = program.create_invocation().value();
    const auto x = activation(2, 256, c::DataType::kBFloat16, 0), y = activation(2, 256, c::DataType::kBFloat16, 1);
    std::vector<double> xd(x.begin(), x.end()), yd(y.begin(), y.end());
    const auto bank = context.upload(w.description, w.bytes).value();
    const std::array a{t::make(context, ad, xd), bank}, b{t::make(context, ad, yd), bank};
    const auto ax = reference(w, quantize(x, 256, 1e-4f, true), 2, 1, 1, {}, c::DataType::kBFloat16),
        by = reference(w, quantize(y, 256, 1e-4f, true), 2, 1, 1, {}, c::DataType::kBFloat16);
    ASSERT_TRUE(first.bind(a).ok());ASSERT_TRUE(second.bind(b).ok());v::Tensor retained;
    { auto one = first.submit().value(), two = second.submit().value();auto r2 = two.wait(), r1 = one.wait();
      ASSERT_TRUE(r1);ASSERT_TRUE(r2);t::expect(context, r1->get()->tensors()[0], ax);t::expect(context, r2->get()->tensors()[0], by);
      retained = r1->get()->tensors()[0]; }
    EXPECT_FALSE(first.submit());
    { auto cancelled = second.submit().value();ASSERT_TRUE(cancelled.request_cancel().ok());auto result = cancelled.wait();
      if (result) t::expect(context, result->get()->tensors()[0], by);else EXPECT_TRUE(result.error().IsCancelled()); }
    { auto repeat = second.submit().value();auto result = repeat.wait();ASSERT_TRUE(result);t::expect(context, result->get()->tensors()[0], by); }
    t::expect(context, retained, ax);EXPECT_EQ(t::bytes(context, bank), w.bytes);
    auto bad = xd;bad[0] = std::numeric_limits<double>::infinity();
    ASSERT_TRUE(second.bind(std::array{t::make(context, ad, bad), bank}).ok());
    { auto work = second.submit().value();auto result = work.wait();ASSERT_FALSE(result);EXPECT_TRUE(result.error().IsInvalidArgument()); }
    EXPECT_FALSE(second.submit());
    auto recovered = program.create_invocation().value();ASSERT_TRUE(recovered.bind(a).ok());
    { auto work = recovered.submit().value();auto result = work.wait();ASSERT_TRUE(result);t::expect(context, result->get()->tensors()[0], ax); }
    EXPECT_TRUE(context.close().ok());EXPECT_FALSE(recovered.submit());
  }
  EXPECT_EQ(quota.used(), 0U);
}
} // namespace vulkan_official_test
