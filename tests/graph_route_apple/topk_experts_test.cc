#include <gtest/gtest.h>

#include <numeric>

#include "../compute_route_apple/tensor_test_support.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/apple.h"

namespace {
using namespace apple_tensor_test;
namespace g = hyle::graph;
namespace ga = g::apple;
constexpr int kExperts = 4, kTokens = 2, kSlots = 2, kColumns = 8, kWidth = 64;
enum class IdSource { kIndices, kReshape, kExternal, kDifferentExtent, kValues };

c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

g::Graph graph(IdSource mode) {
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(dense(c::DataType::kFloat32,
      {kTokens, kSlots, kWidth})).value();
  const auto weight = builder.constant(dense(c::DataType::kFloat32,
      {kExperts, kColumns, kWidth})).value();
  g::ValueId ids;
  if (mode == IdSource::kExternal) {
    ids = builder.input(dense(c::DataType::kInt32, {kTokens, kSlots})).value();
  } else {
    const auto scores = builder.input(dense(mode == IdSource::kValues
        ? c::DataType::kInt32 : c::DataType::kFloat32,
        {kTokens, mode == IdSource::kDifferentExtent ? 5 : kExperts})).value();
    const auto top = builder.add_node(c::Operation(c::TopK{.count = kSlots}),
        std::array{scores}).value();
    ids = builder.result(top, mode == IdSource::kValues ? 0 : 1).value();
    if (mode == IdSource::kReshape) {
      ids = builder.add(c::View::reshape(c::TensorShape::create(
          std::array<std::int64_t, 1>{kTokens * kSlots}).value()).value(), ids).value();
      ids = builder.add(c::View::reshape(c::TensorShape::create(
          std::array<std::int64_t, 2>{kTokens, kSlots}).value()).value(), ids).value();
    }
  }
  const auto node = builder.add_node(c::model_ops::grouped_matmul(),
      std::array{weight, input, ids}).value();
  const auto product = builder.result(node).value();
  const auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, product).value();
  EXPECT_TRUE(builder.publish(ids).ok());
  EXPECT_TRUE(builder.publish(product).ok());
  EXPECT_TRUE(builder.publish(negative).ok());
  return std::move(builder).finish().value();
}

mx::array weight() {
  std::vector<float> values(kExperts * kColumns * kWidth);
  for (int e = 0; e < kExperts; ++e)
    for (int n = 0; n < kColumns; ++n) {
      values[(e * kColumns + n) * kWidth] = float((e + 1) * (n + 1)) / 16;
      values[(e * kColumns + n) * kWidth + 17] = float(e - n) / 32;
    }
  return ready(mx::array(values.data(), {kExperts, kColumns, kWidth}, mx::float32));
}

mx::array activation(int phase) {
  std::vector<float> values(kTokens * kSlots * kWidth);
  for (int row = 0; row < kTokens * kSlots; ++row) {
    values[row * kWidth] = float(row + phase + 1) / 8;
    values[row * kWidth + 17] = float(row - phase - 2) / 16;
  }
  return ready(mx::array(values.data(), {kTokens, kSlots, kWidth}, mx::float32));
}

std::array<float, 8> scores(int phase) {
  return phase ? std::array<float, 8>{3, 0, 2, 1, 0, 3, 1, 2}
               : std::array<float, 8>{0, 2, 1, 3, 3, 0, 2, 1};
}

std::array<std::int32_t, 4> selected(int phase) {
  const auto values = scores(phase);
  std::array<std::int32_t, 4> result{};
  for (int t = 0; t < kTokens; ++t) {
    std::array<int, kExperts> order{0, 1, 2, 3};
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return values[t * kExperts + a] > values[t * kExperts + b];
    });
    for (int s = 0; s < kSlots; ++s) result[t * kSlots + s] = order[s];
  }
  return result;
}

void check(std::span<const mx::array> outputs, int phase) {
  ASSERT_EQ(outputs.size(), 3U);
  const auto ids = selected(phase);
  const auto observed_ids = bits(outputs[0]);
  const auto product = values(outputs[1]), negative = values(outputs[2]);
  for (int row = 0; row < kTokens * kSlots; ++row) {
    EXPECT_EQ(observed_ids[row], ids[row]);
    for (int n = 0; n < kColumns; ++n) {
      const double expected = double((ids[row] + 1) * (n + 1)) / 16 * (row + phase + 1) / 8
          + double(ids[row] - n) / 32 * (row - phase - 2) / 16;
      EXPECT_EQ(product[row * kColumns + n], expected);
      EXPECT_EQ(negative[row * kColumns + n], -expected);
    }
  }
}

TEST(AppleTopKExperts, IndicesAndReshapesKeepNativeResultsAcrossSlots) {
  auto context = a::Context::create().value();
  const auto bank = weight();
  for (auto mode : {IdSource::kIndices, IdSource::kReshape}) {
    SCOPED_TRACE(static_cast<int>(mode));
    const auto recipe = graph(mode);
    auto program = ga::compile(context, recipe, std::array{bank}).value();
    auto first = program.create_invocation().value();
    auto next = program.create_invocation().value();
    const auto raw = scores(0);
    ASSERT_TRUE(first.bind(std::array{activation(0), ready(mx::array(raw.data(), {2, 4}, mx::float32))}).ok());
    auto work = first.submit().value();
    auto old = work.wait().value();
    check(old.get()->arrays(), 0);
    const auto old_product = bits(old.get()->arrays()[1]);
    for (int phase : {0, 1}) {
      const auto data = scores(phase);
      ASSERT_TRUE(next.bind(std::array{activation(phase), ready(mx::array(data.data(), {2, 4}, mx::float32))}).ok());
      {
        auto next_work = next.submit().value();
        auto result = next_work.wait();
        ASSERT_TRUE(result) << result.error().message();
        check(result->get()->arrays(), phase);
      }
      EXPECT_EQ(bits(old.get()->arrays()[1]), old_product);
    }
  }
}

TEST(AppleTopKExperts, ForeignIdsValuesAndDifferentExtentsRetainBoundsChecks) {
  auto context = a::Context::create().value();
  const auto bank = weight();
  for (auto mode : {IdSource::kExternal, IdSource::kDifferentExtent, IdSource::kValues}) {
    SCOPED_TRACE(static_cast<int>(mode));
    const auto recipe = graph(mode);
    auto program = ga::compile(context, recipe, std::array{bank}).value();
    const auto input = activation(0);
    auto arguments = [&](bool invalid) {
      if (mode == IdSource::kDifferentExtent) {
        std::array<float, 10> data{3, 2, 1, 0, -1, 3, 2, 1, 0, -1};
        if (invalid) data[4] = 9;
        return std::array{input, ready(mx::array(data.data(), {2, 5}, mx::float32))};
      }
      std::array<std::int32_t, 8> data{3, 2, 1, 0, 3, 2, 1, 0};
      if (invalid) data[0] = mode == IdSource::kExternal ? -1 : 9;
      return std::array{input, ready(mx::array(data.data(),
          mode == IdSource::kExternal ? mx::Shape{2, 2} : mx::Shape{2, 4}, mx::int32))};
    };
    auto retained = program.create_invocation().value();
    ASSERT_TRUE(retained.bind(arguments(false)).ok());
    auto good_work = retained.submit().value();
    auto old = good_work.wait().value();
    const auto previous = bits(old.get()->arrays()[1]);
    auto failed = program.create_invocation().value();
    ASSERT_TRUE(failed.bind(arguments(true)).ok());
    auto work = failed.submit().value();
    auto result = work.wait();
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsOutOfRange());
    EXPECT_EQ(bits(old.get()->arrays()[1]), previous);
  }
}
} // namespace
