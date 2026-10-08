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
  const auto weight = [&](int col) {
    return affine ? (col % 8) * .5 - .25 :
           bits == 8 ? fp8[col % 4] * .5 : fp4[col % 8] * .5;
  };
  for (int pass = 0; pass < 2; ++pass) {
    if (pass) {
      for (auto &x : values) x *= 2;
      ports[0] = ready(make(values, grouped ? mx::Shape{1, 2, k} : mx::Shape{1, k}, dtype));
    }
    ASSERT_TRUE(invocation.bind(ports).ok());
    auto done = invocation.submit(); ASSERT_TRUE(done);
    auto result = done->wait(); ASSERT_TRUE(result) << result.error().message();
    const auto actual = apple_tensor_test::values(result->get()->arrays()[0]);
    ASSERT_EQ(actual.size(), static_cast<std::size_t>((grouped ? 2 : 1) * n));
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const auto row = i / n;
      const double expected = rounded(weight(0) * values[row * k] +
          weight(1) * values[row * k + 1] + weight(k - 1) * values[(row + 1) * k - 1], dtype);
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
} // namespace
