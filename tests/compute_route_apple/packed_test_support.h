#pragma once
#include "tensor_test_support.h"
#include "hyle/compute/model_ops.h"
#include "../compute_models/mlx_grouped_shared_fixture.h"
#include <gtest/gtest.h>

namespace apple_packed_test {
using namespace apple_tensor_test;
using Format = c::model_ops::MlxWeightFormat;
using Kind = c::model_ops::MlxQuantizedKind;

template <class Prepare>
void check(Format format, bool grouped, mx::Dtype dtype, Prepare prepare) {
  const int k = 128, n = 16, e = grouped ? 2 : 1;
  const bool affine = format == Format::kAffine;
  const int bits = format == Format::kMxfp8 ? 8 : 4;
  const int words_per_row = (k / 32) * bits, group = affine ? 64 : 32;
  std::vector<std::uint32_t> words(e * n * words_per_row,
      bits == 8 ? 0x4038b8c0U : 0x76543210U);
  std::vector<std::uint8_t> exponents(e * n * k / group, 126);
  std::vector<double> scales(e * n * k / group, .5), biases(scales.size(), -.25);
  for (int expert = 0; expert < e; ++expert)
    for (int i = expert * n * k / group; i < (expert + 1) * n * k / group; ++i) {
      exponents[i] = static_cast<std::uint8_t>(126 + expert);
      scales[i] = expert ? 1 : .5;
    }
  std::vector<double> values((grouped ? 2 : 1) * k, 0);
  values[0] = .25; values[1] = -.125; values[k - 1] = .5;
  if (grouped) {
    values[k] = -.5; values[k + 1] = .25; values[2 * k - 1] = .125;
  }
  const auto wshape = grouped ? mx::Shape{e, n, words_per_row}
                             : mx::Shape{n, words_per_row};
  const auto sshape = grouped ? mx::Shape{e, n, k / group} : mx::Shape{n, k / group};
  auto activation = ready(make(values, grouped ? mx::Shape{1, 2, k} : mx::Shape{1, k}, dtype));
  std::vector<mx::array> ports{
      activation, ready(mx::array(words.data(), wshape, mx::uint32)),
      affine ? ready(make(scales, sshape, dtype))
             : ready(mx::array(exponents.data(), sshape, mx::uint8))};
  if (affine) ports.push_back(ready(make(biases, sshape, dtype)));
  if (grouped) ports.push_back(ready(mx::array({std::int32_t{0}, std::int32_t{1}}, {1, 2})));
  auto context = a::Context::create().value();
  const auto operation = c::Operation::custom(c::model_ops::MlxQuantized{
      format, grouped ? Kind::kGrouped : Kind::kLinear,
      static_cast<std::uint32_t>(bits), static_cast<std::uint32_t>(group)});
  auto program = prepare(context, operation, ports);
  ASSERT_TRUE(program) << program.error().message();
  auto invocation = program->create_invocation().value();
  constexpr std::array<double, 8> fp4{0, .5, 1, 1.5, 2, 3, 4, 6};
  constexpr std::array<double, 4> fp8{-2, -1, 1, 2};
  const auto weight = [&](int col, int expert) {
    const double scale = expert ? 1 : .5;
    return affine ? (col % 8) * scale - .25 :
           bits == 8 ? fp8[col % 4] * scale : fp4[col % 8] * scale;
  };
  for (int pass = 0; pass < 2; ++pass) {
    if (pass) {
      for (auto &x : values) x *= 2;
      ports[0] = ready(make(values, grouped ? mx::Shape{1, 2, k} : mx::Shape{1, k}, dtype));
      if (grouped)
        ports.back() = ready(mx::array({std::int32_t{1}, std::int32_t{0}}, {1, 2}));
    }
    ASSERT_TRUE(invocation.bind(ports).ok());
    auto done = invocation.submit(); ASSERT_TRUE(done);
    auto result = done->wait(); ASSERT_TRUE(result) << result.error().message();
    const auto actual = apple_tensor_test::values(result->get()->arrays()[0]);
    ASSERT_EQ(actual.size(), static_cast<std::size_t>((grouped ? 2 : 1) * n));
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const auto row = i / n;
      const int expert = grouped ? (pass ? 1 - static_cast<int>(row) : static_cast<int>(row)) : 0;
      const double expected = rounded(weight(0, expert) * values[row * k] +
          weight(1, expert) * values[row * k + 1] + weight(k - 1, expert) * values[(row + 1) * k - 1], dtype);
      EXPECT_EQ(actual[i], expected);
    }
  }
  if (grouped) {
    ports.back() = ready(mx::array({std::int32_t{-1}, std::int32_t{1}}, {1, 2}));
    ASSERT_TRUE(invocation.bind(ports).ok());
    auto done = invocation.submit(); ASSERT_TRUE(done);
    auto failed = done->wait(); ASSERT_FALSE(failed);
    EXPECT_TRUE(failed.error().IsOutOfRange());
  }
}

template <class Prepare>
void formats(Prepare prepare) {
  for (auto format : {Format::kAffine, Format::kMxfp4, Format::kMxfp8})
    for (bool grouped : {false, true})
      for (auto dtype : {mx::float32, mx::bfloat16}) {
        SCOPED_TRACE(static_cast<int>(format));
        SCOPED_TRACE(grouped);
        SCOPED_TRACE(dtype);
        check(format, grouped, dtype, prepare);
      }
}

template <class Prepare>
void embedding(Prepare prepare) {
  constexpr int k = 128, rows = 3, group = 64;
  std::vector<std::uint32_t> words(rows * k / 8, 0x76543210U);
  const std::vector<double> scales{.5, .5, 1, 1, 2, 2}, bias(scales.size(), -.25);
  for (const auto dtype : {mx::float32, mx::bfloat16}) {
    auto context = a::Context::create().value();
    std::vector<mx::array> ports{
        ready(mx::array(words.data(), {rows, k / 8}, mx::uint32)),
        ready(make(scales, {rows, k / group}, dtype)),
        ready(make(bias, {rows, k / group}, dtype)),
        ready(mx::array({std::int32_t{2}, std::int32_t{0}}, {2}))};
    auto program = prepare(context, c::model_ops::mlx_quantized_embedding(4, group), ports);
    ASSERT_TRUE(program) << program.error().message();
    auto invocation = program->create_invocation().value();
    for (int pass = 0; pass < 2; ++pass) {
      if (pass)
        ports.back() = ready(mx::array({std::int32_t{1}, std::int32_t{2}}, {2}));
      ASSERT_TRUE(invocation.bind(ports).ok());
      auto done = invocation.submit(); ASSERT_TRUE(done);
      auto result = done->wait(); ASSERT_TRUE(result) << result.error().message();
      const auto actual = apple_tensor_test::values(result->get()->arrays()[0]);
      ASSERT_EQ(actual.size(), 2 * k);
      for (int i = 0; i < 2 * k; ++i) {
        const int selected = pass ? (i < k ? 1 : 2) : (i < k ? 2 : 0);
        EXPECT_EQ(actual[i], rounded((i % 8) * scales[selected * 2] - .25, dtype));
      }
    }
    ports.back() = ready(mx::array({std::int32_t{0}, std::int32_t{rows}}, {2}));
    ASSERT_TRUE(invocation.bind(ports).ok());
    auto done = invocation.submit(); ASSERT_TRUE(done);
    auto failed = done->wait(); ASSERT_FALSE(failed);
    EXPECT_TRUE(failed.error().IsOutOfRange());
  }
}

template<class Prepare>
void grouped_shared_activation(Prepare prepare) {
  auto context = a::Context::create().value();
  for (const auto type : {c::DataType::kFloat32, c::DataType::kBFloat16}) {
    const auto native = type == c::DataType::kFloat32 ? mx::float32 : mx::bfloat16;
    for (const bool zero_slot : {false, true}) {
      SCOPED_TRACE(static_cast<int>(type));
      SCOPED_TRACE(zero_slot);
      mlx_grouped_shared_test::Case data(type, zero_slot);
      std::vector<std::uint32_t> words(data.payloads[1].size() / sizeof(std::uint32_t));
      std::memcpy(words.data(), data.payloads[1].data(), data.payloads[1].size());
      const auto activation = [&](int factor) {
        std::vector<double> raw(zero_slot ? 144 : 128, -777.);
        std::fill_n(raw.begin(), 64, double(factor));
        std::fill_n(raw.begin() + (zero_slot ? 80 : 64), 64, 2. * factor);
        auto value = ready(make(raw, {static_cast<int>(raw.size())}, native));
        return zero_slot ? ready(mx::as_strided(value, {2, 3, 64}, {80, 0, 1}, 0, stream()))
                         : ready(mx::reshape(value, {2, 1, 64}, stream()));
      };
      std::vector<mx::array> ports{activation(1),
          ready(mx::array(words.data(), {3, 2, 10}, mx::uint32)),
          ready(make(std::vector<double>(12, .5), {3, 2, 2}, native)),
          ready(make(std::vector<double>(12, -.25), {3, 2, 2}, native)),
          ready(mx::array({std::int32_t{2}, std::int32_t{0}, std::int32_t{2},
                           std::int32_t{1}, std::int32_t{1}, std::int32_t{0}}, {2, 3}))};
      if (zero_slot) ASSERT_EQ(ports[0].strides()[1], 0U);
      auto prepared = prepare(context, data.operation, ports);
      ASSERT_TRUE(prepared) << prepared.error().message();
      auto slot = prepared->create_invocation().value();
      for (int pass = 0; pass < 2; ++pass) {
        if (pass) ports[0] = activation(2);
        ASSERT_TRUE(slot.bind(ports).ok());
        auto work = slot.submit().value();
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        const auto actual = apple_tensor_test::values(result->get()->arrays()[0]);
        ASSERT_EQ(actual.size(), data.expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
          EXPECT_EQ(actual[i], data.expected[i] * float(pass + 1));
      }
      ports.back() = ready(mx::array({std::int32_t{-1}, std::int32_t{0}, std::int32_t{2},
                                      std::int32_t{1}, std::int32_t{1}, std::int32_t{0}}, {2, 3}));
      ASSERT_TRUE(slot.bind(ports).ok());
      auto work = slot.submit().value();
      auto failed = work.wait();
      ASSERT_FALSE(failed);
      EXPECT_TRUE(failed.error().IsOutOfRange());
      EXPECT_FALSE(slot.submit());
    }
  }
}
} // namespace
