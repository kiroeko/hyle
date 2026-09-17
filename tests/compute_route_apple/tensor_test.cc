#include <gtest/gtest.h>

#include <array>
#include <limits>

#include "tensor_test_support.h"

namespace {
using namespace apple_tensor_test;
TEST(AppleTensor, FloatingUnaryBroadcastBinaryAndGatesUseIndependentFormulas) {
  auto context = a::Context::create().value();
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (auto type : {mx::float32, mx::float16, mx::bfloat16}) {
    auto host = pattern(34);
    host[0] = -inf;
    host[1] = inf;
    host[2] = nan;
    host[3] = -0.0;
    auto x = ready(mx::transpose(make(host, {2, 17}, type), stream()));
    auto logical = values(x);
    for (int mode = 0; mode <= static_cast<int>(c::UnaryKind::kSoftplus);
         ++mode) {
      const auto kind = static_cast<c::UnaryKind>(mode);
      SCOPED_TRACE(::testing::Message()
                   << static_cast<int>(apple_tensor_test::logical(type)) << '/'
                   << mode);
      auto result = execute(context, c::Unary{kind}, std::array{x});
      ASSERT_TRUE(result) << result.error().message();
      std::vector<double> expected;
      for (double value : logical)
        expected.push_back(activation(kind, value));
      expect(*result, expected);
    }
    const auto right = make(std::array{0.5, nan}, {1, 2}, type);
    const auto rhs = values(right);
    for (int mode = 0; mode <= static_cast<int>(c::BinaryKind::kMaximum);
         ++mode) {
      const auto kind = static_cast<c::BinaryKind>(mode);
      auto result = execute(context, c::Binary{kind}, std::array{x, right});
      ASSERT_TRUE(result) << result.error().message();
      std::vector<double> expected(logical.size());
      for (std::size_t i = 0; i < expected.size(); ++i) {
        const double l = logical[i], r = rhs[i % 2];
        switch (kind) {
        case c::BinaryKind::kAdd:
          expected[i] = l + r;
          break;
        case c::BinaryKind::kSubtract:
          expected[i] = l - r;
          break;
        case c::BinaryKind::kMultiply:
          expected[i] = l * r;
          break;
        case c::BinaryKind::kDivide:
          expected[i] = l / r;
          break;
        case c::BinaryKind::kMinimum:
          expected[i] = std::fmin(l, r);
          break;
        case c::BinaryKind::kMaximum:
          expected[i] = std::fmax(l, r);
          break;
        }
      }
      expect(*result, expected);
    }
    const auto gate =
        make(std::array{-inf, -2.75, -0.55, 0.333, 0.75, 1.125, 2.875, nan},
             {1, 8}, type);
    const auto up = make(
        std::array{2.0, 0.75, 1.3, 2.125, -3.5, 0.75, 1.3, 1.0}, {1, 8}, type);
    const auto packed = ready(mx::concatenate({gate, up}, -1, stream()));
    for (auto kind :
         {c::GatedActivationKind::kSilu, c::GatedActivationKind::kGelu,
          c::GatedActivationKind::kGeluTanh}) {
      const auto unary_kind =
          kind == c::GatedActivationKind::kSilu   ? c::UnaryKind::kSilu
          : kind == c::GatedActivationKind::kGelu ? c::UnaryKind::kGelu
                                                  : c::UnaryKind::kGeluTanh;
      const auto g = values(gate), u = values(up);
      std::vector<double> expected(g.size());
      for (std::size_t i = 0; i < g.size(); ++i)
        expected[i] = activation(unary_kind, g[i]) * u[i];
      for (auto layout :
           {c::GatedInputKind::kSeparate, c::GatedInputKind::kPackedHalves}) {
        const auto arguments = layout == c::GatedInputKind::kSeparate
                                   ? std::vector{gate, up}
                                   : std::vector{packed};
        auto result =
            execute(context, c::GatedActivation{kind, layout}, arguments);
        ASSERT_TRUE(result) << result.error().message();
        expect(*result, expected);
      }
    }
    for (int width : {0, 1, 127, 128, 129, 257}) {
      const auto input =
          make(pattern(static_cast<std::size_t>(width)), {width}, type);
      auto result = execute(context, c::Unary{c::UnaryKind::kSoftplus},
                            std::array{input});
      ASSERT_TRUE(result) << result.error().message();
      auto expected = values(input);
      for (auto &value : expected)
        value = activation(c::UnaryKind::kSoftplus, value);
      expect(*result, expected);
    }
  }
}

TEST(AppleTensor, ViewsCopiesFillCastSelectionAndConcatPreserveLogicalStorage) {
  auto context = a::Context::create().value();
  for (auto type : {mx::float32, mx::float16, mx::bfloat16}) {
    const auto input = make(pattern(24), {2, 12}, type);
    const auto original = bits(input);
    const auto slice = c::View::slice(-1, 1, 5, 2).value();
    auto sliced = execute(context, slice, std::array{input});
    ASSERT_TRUE(sliced) << sliced.error().message();
    std::vector<std::uint64_t> expected;
    for (std::size_t row = 0; row < 2; ++row)
      for (std::size_t col = 0; col < 5; ++col)
        expected.push_back(original[row * 12 + 1 + col * 2]);
    EXPECT_EQ(bits(*sliced), expected);
    auto copied = execute(context, c::Copy{}, std::array{*sliced});
    ASSERT_TRUE(copied);
    EXPECT_EQ(bits(*copied), expected);
    auto permuted = execute(
        context, c::View::permute(std::array<std::int32_t, 2>{1, 0}).value(),
        std::array{input});
    ASSERT_TRUE(permuted);
    expected.clear();
    for (std::size_t col = 0; col < 12; ++col)
      for (std::size_t row = 0; row < 2; ++row)
        expected.push_back(original[row * 12 + col]);
    EXPECT_EQ(bits(*permuted), expected);
    const auto eight = c::TensorShape::create(
                           std::array<std::int64_t, 8>{2, 1, 1, 2, 1, 1, 2, 3})
                           .value();
    auto reshaped =
        execute(context, c::View::reshape(eight).value(), std::array{input});
    ASSERT_TRUE(reshaped);
    EXPECT_EQ(bits(*reshaped), original);
    const auto scalar_input = make(std::array{0.375}, {}, type);
    const auto broad =
        c::TensorShape::create(std::array<std::int64_t, 2>{3, 5}).value();
    auto broadcast = execute(context, c::View::broadcast_to(broad).value(),
                             std::array{scalar_input});
    ASSERT_TRUE(broadcast);
    EXPECT_EQ(bits(*broadcast),
              std::vector<std::uint64_t>(15, bits(scalar_input)[0]));
    const auto mask = mx::array({true, false, true, false, true}, {1, 5});
    auto chosen =
        execute(context, c::Select{}, std::array{mask, *sliced, scalar_input});
    ASSERT_TRUE(chosen) << chosen.error().message();
    auto chosen_expected = bits(*sliced);
    for (std::size_t i = 0; i < chosen_expected.size(); ++i)
      if (i % 5 % 2 == 1)
        chosen_expected[i] = bits(scalar_input)[0];
    EXPECT_EQ(bits(*chosen), chosen_expected);
    const auto empty = make(std::span<const double>{}, {2, 0}, type);
    const std::array args{*sliced, empty, *sliced, *sliced, empty};
    auto joined = execute(context, c::Concat{.axis = -1}, args);
    ASSERT_TRUE(joined) << joined.error().message();
    expected.clear();
    const auto part = bits(*sliced);
    for (std::size_t row = 0; row < 2; ++row)
      for (int repeat = 0; repeat < 3; ++repeat)
        expected.insert(expected.end(), part.begin() + row * 5,
                        part.begin() + row * 5 + 5);
    EXPECT_EQ(bits(*joined), expected);
    for (auto destination : {mx::float32, mx::float16, mx::bfloat16}) {
      auto cast =
          execute(context, c::Cast{logical(destination)}, std::array{input});
      ASSERT_TRUE(cast);
      expect(*cast, values(input));
    }
    const std::uint64_t pattern_bits =
        type == mx::float32 ? 0x80000000ULL : 0x8000ULL;
    const c::Fill fill{broad, logical(type),
                       c::ScalarValue::from_bits(logical(type), pattern_bits)};
    auto result = execute(context, fill, std::span<const mx::array>{});
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(bits(*result), std::vector<std::uint64_t>(15, pattern_bits));
  }
  for (auto type : {mx::int32, mx::bool_}) {
    const auto shape =
        c::TensorShape::create(std::array<std::int64_t, 1>{17}).value();
    const auto value = type == mx::int32 ? c::ScalarValue(std::int32_t{-7})
                                         : c::ScalarValue(true);
    auto result = execute(context, c::Fill{shape, logical(type), value},
                          std::span<const mx::array>{});
    ASSERT_TRUE(result);
    EXPECT_EQ(values(*result),
              std::vector<double>(17, type == mx::int32 ? -7 : 1));
  }
}

TEST(AppleTensor, MixedParameterNormsMatchFp64AndRejectWrongPrecision) {
  auto context = a::Context::create().value();
  for (auto type : {mx::float32, mx::float16, mx::bfloat16})
    for (int width : {0, 1, 129, 2048}) {
      auto input =
          make(pattern(static_cast<std::size_t>(3 * width)), {3, width}, type);
      const auto data = values(input);
      for (bool mixed : {false, true}) {
        auto weight_values = pattern(width, 1), bias_values = pattern(width, 2);
        for (auto &x : weight_values)
          x = x * 0.03 + 1;
        for (auto &x : bias_values)
          x *= 0.05;
        auto weight = make(weight_values, {width}, mixed ? mx::float32 : type);
        auto bias = make(bias_values, {width}, mixed ? mx::float32 : type);
        const auto w = values(weight), b = values(bias);
        for (bool layer : {false, true}) {
          const auto operation =
              layer ? c::Operation(c::LayerNorm{}) : c::Operation(c::RmsNorm{});
          const auto args = layer ? std::vector{input, weight, bias}
                                  : std::vector{input, weight};
          auto result = execute(context, operation, args);
          ASSERT_TRUE(result) << result.error().message();
          std::vector<double> expected(data.size());
          for (int row = 0; row < 3 && width != 0; ++row) {
            double mean = 0, variance = 0;
            if (layer) {
              for (int col = 0; col < width; ++col)
                mean += data[row * width + col];
              mean /= width;
            }
            for (int col = 0; col < width; ++col) {
              const auto d = data[row * width + col] - mean;
              variance += d * d;
            }
            const auto inverse =
                1 / std::sqrt(variance / width + static_cast<double>(1e-5F));
            for (int col = 0; col < width; ++col)
              expected[row * width + col] =
                  (data[row * width + col] - mean) * inverse * w[col] +
                  (layer ? b[col] : 0);
          }
          expect(*result, expected);
        }
      }
    }
  const std::array specs{a::InputSpec{{2, 129}, mx::bfloat16},
                         a::InputSpec{{129}, mx::float32}};
  auto prepared = context.prepare(c::RmsNorm{}, specs);
  ASSERT_TRUE(prepared) << prepared.error().message();
  auto unsupported = context.prepare(
      c::RmsNorm{.accumulation_type = c::DataType::kFloat64}, specs);
  ASSERT_FALSE(unsupported);
  EXPECT_TRUE(unsupported.error().IsUnimplemented());
  const std::array integers{a::InputSpec{{3}, mx::int32},
                            a::InputSpec{{3}, mx::int32}};
  auto rejected = context.prepare(c::Binary{}, integers);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsUnimplemented());
  const std::array mismatch{a::InputSpec{{2, 129}, mx::bfloat16},
                            a::InputSpec{{128}, mx::float32}};
  auto bad = context.prepare(c::RmsNorm{}, mismatch);
  ASSERT_FALSE(bad);
  EXPECT_TRUE(bad.error().IsInvalidArgument());
  const std::array bfmat{a::InputSpec{{2, 2}, mx::bfloat16},
                         a::InputSpec{{2, 2}, mx::bfloat16}};
  EXPECT_FALSE(context.prepare(c::MatMul{}, bfmat));
}
} // namespace
