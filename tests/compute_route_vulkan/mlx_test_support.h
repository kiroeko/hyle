#pragma once

#include "hyle/compute/model_ops.h"
#include "tensor_test_support.h"

namespace vulkan_mlx_test {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace t = vulkan_tensor_test;
using Format = c::model_ops::MlxWeightFormat;
using Kind = c::model_ops::MlxQuantizedKind;
constexpr std::array types{c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16};

inline float fp8(std::uint8_t code) {
  const auto exponent = (code >> 3) & 15, fraction = code & 7;
  if (exponent == 15 && fraction == 7) return std::numeric_limits<float>::quiet_NaN();
  const auto value = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -9)
      : std::ldexp(1.f + fraction / 8.f, static_cast<int>(exponent) - 7);
  return (code & 128) ? -value : value;
}

// Logical reference uses original format definitions and a scalar F32 group
// accumulator, not the generated IR, GPU results or product codec.
struct Inputs final {
  Format format;
  Kind kind;
  int n, k, group, experts, rows;
  c::DataType activation_type, scale_type, dtype;
  c::Operation operation;
  std::vector<c::TensorDesc> descriptions;
  std::vector<std::vector<std::byte>> payloads;
  c::TensorDesc output;
  std::vector<std::uint8_t> codes, exponents;
  std::vector<double> activation, scales, biases, expected;
  std::vector<std::int32_t> ids;

  Inputs(Format fmt, Kind mode, c::DataType type, int g = 32, int seed = 0,
         c::DataType scale = c::DataType::kUnknown, int width = 0)
      : format(fmt), kind(mode), n(mode == Kind::kEmbedding ? 7 : 5),
        k(width ? width : 3 * g), group(g), experts(mode == Kind::kGrouped ? 3 : 1),
        rows(mode == Kind::kGrouped ? 4 : mode == Kind::kEmbedding ? 3 : 2),
        activation_type(type), scale_type(scale == c::DataType::kUnknown ? type : scale),
        dtype(mode == Kind::kEmbedding ? scale_type
              : fmt == Format::kAffine4 && type != scale_type ? c::DataType::kFloat32 : type),
        operation(c::Operation::custom(c::model_ops::MlxQuantized{fmt, mode, static_cast<std::uint32_t>(g)})) {
    const auto add = [&](c::TensorDesc desc, std::span<const std::uint64_t> raw) {
      descriptions.push_back(desc);payloads.push_back(t::payload(desc, raw));
    };
    const bool embedding = kind == Kind::kEmbedding, grouped = kind == Kind::kGrouped;
    const bool affine = format == Format::kAffine4;
    if (!embedding) {
      activation.resize(rows * k);std::vector<std::uint64_t> raw(activation.size());
      for (std::size_t i = 0; i < raw.size(); ++i) {
        activation[i] = t::rounded((static_cast<int>((i * 11 + seed * 19) % 19) - 9) / 16., type);
        raw[i] = t::encode(activation[i], type);
      }
      add(grouped ? t::dense(type, {2, 2, k}) : t::dense(type, {rows, k}), raw);
    }
    const auto bits = format == Format::kMxfp8 ? 8 : 4, pack = 32 / bits;
    codes.resize(experts * n * k);std::vector<std::uint64_t> words(codes.size() / pack, 0);
    constexpr std::array<std::uint8_t, 8> fp8_codes{0x30, 0x38, 0x40, 0xb0, 0xb8, 0xc0, 0x00, 0x80};
    for (std::size_t i = 0; i < codes.size(); ++i) {
      codes[i] = bits == 8 ? fp8_codes[(i * 3 + seed) % fp8_codes.size()]
                           : static_cast<std::uint8_t>((i * 7 + i / k * 3 + seed * 5) % 16);
      words[i / pack] |= static_cast<std::uint64_t>(codes[i]) << (bits * (i % pack));
    }
    add(grouped ? t::dense(c::DataType::kUInt32, {experts, n, k / pack})
                : t::dense(c::DataType::kUInt32, {n, k / pack}), words);
    const auto count = experts * n * (k / group);
    std::vector<std::uint64_t> scale_raw(count), bias_raw(count);
    if (affine) {
      scales.resize(count);biases.resize(count);
      for (int i = 0; i < count; ++i) {
        const double s = embedding ? .1 + ((i * 17 + seed) % 7) * .0137 : .0625 * (1 + (i + seed) % 5);
        const double b = embedding ? .0143 * (i % 5 - seed - 2) : -.015625 * (1 + (i + seed) % 3);
        scales[i] = t::rounded(s, scale_type);biases[i] = t::rounded(b, scale_type);
        scale_raw[i] = t::encode(scales[i], scale_type);bias_raw[i] = t::encode(biases[i], scale_type);
      }
      const auto shape = grouped ? t::dense(scale_type, {experts, n, k / group})
                                  : t::dense(scale_type, {n, k / group});
      add(shape, scale_raw);add(shape, bias_raw);
    } else {
      exponents.resize(count);
      for (int i = 0; i < count; ++i) scale_raw[i] = exponents[i] = static_cast<std::uint8_t>(125 + (i + seed) % 5);
      add(grouped ? t::dense(c::DataType::kUInt8, {experts, n, k / group})
                   : t::dense(c::DataType::kUInt8, {n, k / group}), scale_raw);
    }
    if (grouped || embedding) {
      ids.resize(rows);std::vector<std::uint64_t> raw(rows);
      for (int i = 0; i < rows; ++i) raw[i] = static_cast<std::uint64_t>(ids[i] = (i * 2 + seed) % (grouped ? experts : n));
      add(grouped ? t::dense(c::DataType::kInt32, {2, 2}) : t::dense(c::DataType::kInt32, {rows}), raw);
    }
    std::array<c::TensorDesc, 1> inferred;
    EXPECT_TRUE(c::infer_output_descriptions(operation, descriptions, inferred).ok());output = inferred[0];
    calculate();
  }

  void calculate() {
    constexpr std::array<float, 16> fp4{0, .5, 1, 1.5, 2, 3, 4, 6, -0.0f, -.5, -1, -1.5, -2, -3, -4, -6};
    const bool affine = format == Format::kAffine4, embedding = kind == Kind::kEmbedding;
    expected.assign(rows * (embedding ? k : n), 0);
    for (int r = 0; r < rows; ++r) {
      const auto expert = kind == Kind::kGrouped ? ids[r] : 0;
      if (embedding) {
        for (int col = 0; col < k; ++col) {
          const auto s = ids[r] * (k / group) + col / group;
          const float product = static_cast<float>(codes[ids[r] * k + col] * scales[s]);
          const float added = static_cast<float>(t::rounded(product, dtype) + biases[s]);
          expected[r * k + col] = t::rounded(added, dtype);
        }
        continue;
      }
      for (int col = 0; col < n; ++col) {
        float sum = 0;
        for (int block = 0; block < k / group; ++block) {
          const auto scale_index = (expert * n + col) * (k / group) + block;
          float dot = 0;
          for (int j = 0; j < group; ++j) {
            const auto code = codes[(expert * n + col) * k + block * group + j];
            float value;
            if (affine) {
              const float product = static_cast<float>(code * t::rounded(scales[scale_index], dtype));
              value = product + static_cast<float>(t::rounded(biases[scale_index], dtype));
            } else value = format == Format::kMxfp4 ? fp4[code] : fp8(code);
            const float a = static_cast<float>(t::rounded(activation[r * k + block * group + j], dtype));
            dot += a * value;
          }
          if (!affine) {
            const auto e = exponents[scale_index];
            const auto raw = e == 255 ? std::numeric_limits<double>::infinity()
                : std::ldexp(1., e == 0 ? -127 : static_cast<int>(e) - 127);
            dot *= static_cast<float>(t::rounded(raw, dtype));
          }
          sum += dot;
        }
        expected[r * n + col] = t::rounded(sum, dtype);
      }
    }
  }

  std::vector<v::Tensor> upload(const v::Context &context) const {
    std::vector<v::Tensor> result;
    for (std::size_t i = 0; i < descriptions.size(); ++i)
      result.push_back(context.upload(descriptions[i], payloads[i]).value());
    return result;
  }

  void check(const v::Context &context, const v::Tensor &tensor) const {
    ASSERT_EQ(tensor.description(), output);
    t::expect(context, tensor, expected, true);
  }
};

template <class Prepare>
void batch(Prepare prepare) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 384 * 1024 * 1024);
  {
    auto context = vulkan_test::context(&quota).value();
    for (auto format : {Format::kAffine4, Format::kMxfp4, Format::kMxfp8})
      for (auto kind : {Kind::kLinear, Kind::kGrouped, Kind::kEmbedding}) {
        if (kind == Kind::kEmbedding && format != Format::kAffine4) continue;
        for (auto dtype : types)
          for (int group : {32, 64, 128}) {
            if (format != Format::kAffine4 && group != 32) continue;
            SCOPED_TRACE(::testing::Message() << static_cast<int>(format) << '/' << static_cast<int>(kind)
                << '/' << static_cast<int>(dtype) << '/' << group);
            Inputs a(format, kind, dtype, group), b(format, kind, dtype, group, 1);
            auto program = prepare(context, a.operation, a.descriptions);
            ASSERT_TRUE(program) << program.error().message();
            auto first = program->create_invocation();ASSERT_TRUE(first) << first.error().message();
            auto second = program->create_invocation();ASSERT_TRUE(second) << second.error().message();
            ASSERT_TRUE(first->bind(a.upload(context)).ok());ASSERT_TRUE(second->bind(b.upload(context)).ok());
            v::Tensor retained;
            { auto w1 = first->submit().value(), w2 = second->submit().value();
              auto r2 = w2.wait(), r1 = w1.wait();
              ASSERT_TRUE(r1) << r1.error().message();ASSERT_TRUE(r2) << r2.error().message();
              a.check(context, r1->get()->tensors()[0]);b.check(context, r2->get()->tensors()[0]);
              retained = r1->get()->tensors()[0]; }
            EXPECT_FALSE(first->submit());
            for (int pass = 0; pass < 2; ++pass) {
              auto done = second->submit().value();auto result = done.wait();ASSERT_TRUE(result) << result.error().message();
              b.check(context, result->get()->tensors()[0]);a.check(context, retained);
            }
            retained = {};
            ASSERT_TRUE(first->bind(b.upload(context)).ok());
            { auto done = first->submit().value();auto result = done.wait();ASSERT_TRUE(result) << result.error().message();
              b.check(context, result->get()->tensors()[0]); }
            if (kind != Kind::kLinear) {
              auto inputs = b.upload(context);
              auto raw = b.ids;raw.back() = -1;
              inputs.back() = context.upload(b.descriptions.back(), std::as_bytes(std::span(raw))).value();
              ASSERT_TRUE(first->bind(inputs).ok());
              { auto done = first->submit().value();auto failure = done.wait();ASSERT_FALSE(failure);
                EXPECT_TRUE(failure.error().IsOutOfRange()); }
              EXPECT_TRUE(first->bind(a.upload(context)).IsOutOfRange());
              auto recovered = program->create_invocation().value();
              ASSERT_TRUE(recovered.bind(a.upload(context)).ok());
              { auto done = recovered.submit().value();auto result = done.wait();ASSERT_TRUE(result) << result.error().message();
                a.check(context, result->get()->tensors()[0]); }
            }
          }
      }
    for (auto a : types) for (auto s : types) {
      if (a == s) continue;
      Inputs data(Format::kAffine4, Kind::kGrouped, a, 64, 1, s);
      auto program = prepare(context, data.operation, data.descriptions).value();
      auto slot = program.create_invocation().value();ASSERT_TRUE(slot.bind(data.upload(context)).ok());
      auto done = slot.submit().value();auto result = done.wait();ASSERT_TRUE(result) << result.error().message();
      data.check(context, result->get()->tensors()[0]);
    }
    EXPECT_TRUE(context.close().ok());
  }
  EXPECT_EQ(quota.used(), 0U);
}

template <class Prepare>
void constants(Prepare prepare) {
  auto context = vulkan_test::context().value();
  Inputs a(Format::kAffine4, Kind::kGrouped, c::DataType::kBFloat16, 64);
  Inputs b(Format::kAffine4, Kind::kGrouped, c::DataType::kBFloat16, 64, 1);
  b.codes = a.codes;b.scales = a.scales;b.biases = a.biases;b.calculate();
  const auto resident = a.upload(context), changed = b.upload(context);
  auto program = prepare(context, a.operation, resident);
  ASSERT_TRUE(program) << program.error().message();
  auto first = program->create_invocation().value(), second = program->create_invocation().value();
  const std::array first_inputs{resident.front(), resident.back()}, second_inputs{changed.front(), changed.back()};
  ASSERT_TRUE(first.bind(first_inputs).ok());ASSERT_TRUE(second.bind(second_inputs).ok());
  v::Tensor retained;
  { auto w1 = first.submit().value(), w2 = second.submit().value();auto r1 = w1.wait(), r2 = w2.wait();
    ASSERT_TRUE(r1) << r1.error().message();ASSERT_TRUE(r2) << r2.error().message();
    a.check(context, r1->get()->tensors()[0]);b.check(context, r2->get()->tensors()[0]);
    retained = r1->get()->tensors()[0]; }
  for (int pass = 0; pass < 3; ++pass) {
    ASSERT_TRUE(second.bind(pass % 2 ? first_inputs : second_inputs).ok());
    auto work = second.submit().value();auto result = work.wait();ASSERT_TRUE(result) << result.error().message();
    (pass % 2 ? a : b).check(context, result->get()->tensors()[0]);a.check(context, retained);
    EXPECT_EQ(t::bytes(context, resident[1]), a.payloads[1]);
  }
}
} // namespace vulkan_mlx_test
