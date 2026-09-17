#include <gtest/gtest.h>

#include <array>
#include <numeric>

#include "reduction_test_support.h"

namespace {
using namespace apple_tensor_test;

TEST(AppleReduction, AxesLayoutsEmptyIdentitiesAndSignedZero) {
  auto context = a::Context::create().value();
  for (auto type : {mx::float32, mx::float16, mx::bfloat16}) {
    for (auto shape : {mx::Shape{3, 129}, mx::Shape{2, 17, 3},
                       mx::Shape{1, 2, 1, 3, 1, 17, 1, 1}, mx::Shape{2, 0, 3},
                       mx::Shape{0, 17}, mx::Shape{}}) {
      const auto count = std::accumulate(shape.begin(), shape.end(),
                                         std::size_t{1}, std::multiplies<>{});
      auto data = pattern(count);
      for (auto &x : data)
        x = 1.0 + x * 0.1;
      auto input = make(data, shape, type);
      if (shape == mx::Shape{2, 17, 3})
        input = ready(
            mx::transpose(make(data, {17, 2, 3}, type), {1, 0, 2}, stream()));
      data = values(input);
      std::vector<std::int32_t> selected;
      if (shape.size() == 3)
        selected = shape[1] == 0 ? std::vector<std::int32_t>{1}
                                 : std::vector<std::int32_t>{0, -1};
      else if (shape.size() == 8)
        selected = {1, 3, 5};
      else if (!shape.empty())
        selected = {-1};
      const auto axes =
          shape.empty() ? c::Axes::all() : c::Axes::from(selected).value();
      for (bool keep : {false, true}) {
        for (auto kind :
             {c::ReductionKind::kSum, c::ReductionKind::kMean,
              c::ReductionKind::kMinimum, c::ReductionKind::kMaximum}) {
          SCOPED_TRACE(::testing::Message()
                       << static_cast<int>(logical(type)) << '/' << shape.size()
                       << '/' << static_cast<int>(kind) << '/' << keep);
          auto result = execute(
              context, c::Reduce{kind, axes, keep, c::DataType::kFloat32},
              std::array{input});
          ASSERT_TRUE(result) << result.error().message();
          std::vector<double> expected(result->size(), 0.0);
          if (kind == c::ReductionKind::kMinimum)
            std::fill(expected.begin(), expected.end(),
                      std::numeric_limits<double>::infinity());
          if (kind == c::ReductionKind::kMaximum)
            std::fill(expected.begin(), expected.end(),
                      -std::numeric_limits<double>::infinity());
          std::vector<std::size_t> counts(expected.size());
          std::vector<bool> reduced(shape.size(), shape.empty());
          for (int axis : selected)
            reduced[axis < 0 ? axis + shape.size() : axis] = true;
          for (std::size_t i = 0; i < data.size(); ++i) {
            std::size_t remaining = i, destination = 0, stride = 1;
            for (std::size_t dim = shape.size(); dim-- > 0;) {
              const auto coordinate = remaining % shape[dim];
              remaining /= shape[dim];
              if (!reduced[dim]) {
                destination += coordinate * stride;
                stride *= shape[dim];
              }
            }
            ++counts[destination];
            if (kind == c::ReductionKind::kMinimum)
              expected[destination] = std::min(expected[destination], data[i]);
            else if (kind == c::ReductionKind::kMaximum)
              expected[destination] = std::max(expected[destination], data[i]);
            else
              expected[destination] += data[i];
          }
          if (kind == c::ReductionKind::kMean)
            for (std::size_t i = 0; i < expected.size(); ++i)
              expected[i] = counts[i]
                                ? expected[i] / counts[i]
                                : std::numeric_limits<double>::quiet_NaN();
          expect(*result, expected);
        }
      }
    }
    const auto inf = std::numeric_limits<double>::infinity();
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto input = make(std::array{0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.0,
                                       nan, 3.0, inf, -inf, 1.0},
                            {4, 3}, type);
    for (auto kind : {c::ReductionKind::kMinimum, c::ReductionKind::kMaximum}) {
      auto output =
          execute(context,
                  c::Reduce{.kind = kind,
                            .axes = c::Axes::from(std::array{1}).value(),
                            .accumulation_type = c::DataType::kFloat32},
                  std::array{input});
      ASSERT_TRUE(output) << output.error().message();
      const auto actual = values(*output);
      for (int i : {0, 1}) {
        EXPECT_EQ(actual[i], 0);
        EXPECT_EQ(std::signbit(actual[i]), kind == c::ReductionKind::kMinimum);
      }
      EXPECT_TRUE(std::isnan(actual[2]));
      EXPECT_EQ(actual[3], kind == c::ReductionKind::kMinimum ? -inf : inf);
    }
  }
}

TEST(AppleReduction, SoftmaxMasksScaleAxesAndNonfiniteRows) {
  auto context = a::Context::create().value();
  for (auto type : {mx::float32, mx::float16, mx::bfloat16}) {
    for (int width : {0, 1, 17, 127, 129, 2048, 8193}) {
      auto input =
          ready(mx::transpose(make(pattern(static_cast<std::size_t>(6 * width)),
                                   {width, 2, 3}, type),
                              {1, 0, 2}, stream()));
      const auto data = values(input);
      for (auto mode : {c::MaskKind::kNone, c::MaskKind::kBoolean,
                        c::MaskKind::kAdditive}) {
        std::vector<std::uint8_t> mask(width);
        std::vector<double> bias(width);
        for (int col = 0; col < width; ++col) {
          mask[col] = col % 3 != 1;
          bias[col] = mask[col] ? (col % 7) * 0.3
                                : -std::numeric_limits<double>::infinity();
        }
        auto native_mask =
            mode == c::MaskKind::kBoolean
                ? mx::array(mask.data(), mx::Shape{1, width, 1}, mx::bool_)
                : make(bias, {1, width, 1}, type);
        for (auto kind :
             {c::SoftmaxKind::kSoftmax, c::SoftmaxKind::kLogSoftmax}) {
          const c::Softmax operation{
              .axis = -2, .scale = -0.7F, .kind = kind, .mask = mode};
          std::vector<mx::array> inputs{input};
          if (mode != c::MaskKind::kNone)
            inputs.push_back(native_mask);
          auto result = execute(context, operation, inputs);
          ASSERT_TRUE(result) << result.error().message();
          expect(*result, probability_oracle(data, input.shape(), operation,
                                             values(native_mask)));
        }
      }
    }
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto input =
        make(std::array{nan,  1.0,  2.0,  -inf, 4.0,  0.0, inf, 2.0, 3.0, 4.0,
                        -inf, -inf, -inf, -inf, -inf, 0.0, 1.0, nan, 3.0, 4.0},
             {4, 5}, type);
    for (auto kind : {c::SoftmaxKind::kSoftmax, c::SoftmaxKind::kLogSoftmax}) {
      for (auto flags : {std::array<std::uint8_t, 5>{0, 1, 1, 1, 1},
                         std::array<std::uint8_t, 5>{0, 0, 0, 0, 0},
                         std::array<std::uint8_t, 5>{1, 1, 1, 1, 1}}) {
        const auto mask = mx::array(flags.data(), {5}, mx::bool_);
        const c::Softmax operation{.kind = kind, .mask = c::MaskKind::kBoolean};
        auto result = execute(context, operation, std::array{input, mask});
        ASSERT_TRUE(result) << result.error().message();
        expect(*result, probability_oracle(values(input), input.shape(),
                                           operation, values(mask)));
      }
    }
  }
  // A large shared F32 offset must not erase log(sum(exp(x-max))).
  const auto offset = make(std::array{1e8, 1e8, 1e8, 1e8}, {1, 4}, mx::float32);
  auto log = execute(context, c::Softmax{.kind = c::SoftmaxKind::kLogSoftmax},
                     std::array{offset});
  ASSERT_TRUE(log) << log.error().message();
  expect(*log, std::array{-std::log(4.0), -std::log(4.0), -std::log(4.0),
                          -std::log(4.0)});
}

TEST(AppleReduction, TopKStableTiesRawStorageAndNativeMergeTiles) {
  auto context = a::Context::create().value();
  for (auto type : {mx::float32, mx::float16, mx::bfloat16}) {
    for (const auto &[width, count] : {std::pair{1, 1},
                                      {17, 1},
                                      {17, 17},
                                      {127, 6},
                                      {129, 8},
                                      {256, 8},
                                      {897, 8},
                                      {2048, 8},
                                      {4097, 6},
                                      {8193, 1}}) {
      auto data = pattern(static_cast<std::size_t>(width * 6));
      for (std::size_t i = 0; i < data.size(); ++i) {
        if (i % 7 == 0)
          data[i] = std::numeric_limits<double>::quiet_NaN();
        if (i % 7 == 1)
          data[i] = std::numeric_limits<double>::infinity();
        if (i % 7 == 2)
          data[i] = -std::numeric_limits<double>::infinity();
        if (i % 7 == 3)
          data[i] = -0.0;
        if (i % 7 == 4)
          data[i] = 0.0;
      }
      auto input = ready(
          mx::transpose(make(data, {width, 2, 3}, type), {1, 0, 2}, stream()));
      data = values(input);
      const auto raw = bits(input);
      for (bool largest : {false, true}) {
        const c::TopK operation{.count = static_cast<std::size_t>(count),
                                .axis = -2,
                                .largest = largest,
                                .sorted = width % 2 == 0,
                                .stable = width % 3 != 0};
        auto outputs =
            execute_all(context, c::Operation(operation), std::array{input});
        ASSERT_TRUE(outputs) << outputs.error().message();
        ASSERT_EQ(outputs->size(), 2U);
        const auto actual = bits((*outputs)[0]);
        const auto indices = values((*outputs)[1]);
        EXPECT_EQ((*outputs)[1].dtype(), mx::int32);
        for (int row = 0; row < 2; ++row) {
          for (int lane = 0; lane < 3; ++lane) {
            std::vector<double> scores(width);
            for (int col = 0; col < width; ++col)
              scores[col] = data[(row * width + col) * 3 + lane];
            const auto expected = top_indices(scores, count, largest);
            for (int i = 0; i < count; ++i) {
              const auto output = (row * count + i) * 3 + lane;
              EXPECT_EQ(indices[output], expected[i]);
              EXPECT_EQ(actual[output],
                        raw[(row * width + expected[i]) * 3 + lane]);
            }
          }
        }
        EXPECT_EQ(bits(input), raw);
      }
    }
    // All-NaN and equal-zero rows preserve distinct payload/sign bits.
    std::vector<std::uint32_t> raw32{0x7fc00001U, 0xffc01234U, 0x7fc05678U,
                                     0x80000000U, 0U,          0x80000000U};
    std::vector<std::uint16_t> raw16 =
        type == mx::float16 ? std::vector<std::uint16_t>{0x7e01, 0xfe37, 0x7e28,
                                                         0x8000, 0,      0x8000}
                            : std::vector<std::uint16_t>{
                                  0x7fc1, 0xffc3, 0x7fc7, 0x8000, 0, 0x8000};
    const auto input = ready(mx::view(
        type == mx::float32 ? mx::array(raw32.data(), {2, 3}, mx::uint32)
                            : mx::array(raw16.data(), {2, 3}, mx::uint16),
        type, stream()));
    for (std::size_t count : {1U, 3U}) {
      auto out = execute_all(context, c::Operation(c::TopK{.count = count}),
                             std::array{input});
      ASSERT_TRUE(out) << out.error().message();
      const auto raw = bits(input), actual = bits((*out)[0]);
      const auto indices = values((*out)[1]);
      for (int row = 0; row < 2; ++row)
        for (std::size_t i = 0; i < count; ++i) {
          EXPECT_EQ(indices[row * count + i], i);
          EXPECT_EQ(actual[row * count + i], raw[row * 3 + i]);
        }
    }
  }
}

TEST(AppleReduction, EmptyResultsAndPreparationRejections) {
  auto context = a::Context::create().value();
  const auto input = ready(mx::zeros({0, 17}, mx::bfloat16, stream()));
  auto top = execute_all(context, c::Operation(c::TopK{.count = 8}),
                         std::array{input});
  ASSERT_TRUE(top) << top.error().message();
  EXPECT_EQ((*top)[0].shape(), (mx::Shape{0, 8}));
  EXPECT_EQ((*top)[1].shape(), (mx::Shape{0, 8}));
  const std::array spec{a::InputSpec{{2, 17}, mx::bfloat16}};
  EXPECT_FALSE(context.prepare(c::Reduce{}, spec));
  EXPECT_FALSE(context.prepare(
      c::Reduce{.accumulation_type = c::DataType::kFloat64}, spec));
  EXPECT_FALSE(context.prepare(
      c::Softmax{.accumulation_type = c::DataType::kUnknown}, spec));
  EXPECT_FALSE(context.prepare(c::TopK{.count = 0}, spec));
  EXPECT_FALSE(context.prepare(c::TopK{.count = 18}, spec));
  EXPECT_FALSE(context.prepare(c::TopK{.axis = 2}, spec));
  EXPECT_FALSE(
      context.prepare(c::TopK{}, std::array{a::InputSpec{{17}, mx::int32}}));
  EXPECT_FALSE(
      context.prepare(c::Softmax{.mask = c::MaskKind::kAdditive},
                      std::array{spec[0], a::InputSpec{{17}, mx::float32}}));
  EXPECT_FALSE(
      context.prepare(c::Softmax{.mask = c::MaskKind::kBoolean},
                      std::array{spec[0], a::InputSpec{{16}, mx::bool_}}));
  EXPECT_TRUE(context.prepare(c::Reduce{},
                              std::array{a::InputSpec{{17}, mx::float32}}));
}
} // namespace
