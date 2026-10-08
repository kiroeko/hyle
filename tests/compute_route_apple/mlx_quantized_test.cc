#include "tensor_test_support.h"
#include "hyle/compute/model_ops.h"
#include <gtest/gtest.h>

namespace {
using namespace apple_tensor_test;
using Format = c::model_ops::MlxWeightFormat;
using Kind = c::model_ops::MlxQuantizedKind;

void check(Format format, bool grouped, mx::Dtype dtype) {
  const int k = 128, n = 16, e = grouped ? 2 : 1;
  const bool affine = format == Format::kAffine4;
  const int bits = format == Format::kMxfp8 ? 8 : 4;
  const int pack = 32 / bits, group = affine ? 64 : 32;
  std::vector<std::uint32_t> words(e * n * k / pack,
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
  const auto wshape = grouped ? mx::Shape{e, n, k / pack} : mx::Shape{n, k / pack};
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
      format, grouped ? Kind::kGrouped : Kind::kLinear, static_cast<std::uint32_t>(group)});
  auto program = context.prepare(operation, ports);
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

TEST(AppleMlxQuantized, NativeFormatsUseOrdinaryEntryAndCheckedExperts) {
  for (auto format : {Format::kAffine4, Format::kMxfp4, Format::kMxfp8})
    for (bool grouped : {false, true})
      for (auto dtype : {mx::float32, mx::bfloat16}) {
        SCOPED_TRACE(static_cast<int>(format));
        SCOPED_TRACE(grouped);
        SCOPED_TRACE(dtype);
        check(format, grouped, dtype);
      }
}

TEST(AppleMlxQuantized, EmbeddingSelectsPackedRowsAndChecksChangedIds) {
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
    auto program = context.prepare(c::model_ops::mlx_quantized_embedding(group), ports);
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
} // namespace
