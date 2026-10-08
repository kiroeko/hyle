#include <gtest/gtest.h>
#include <mlx/ops.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "hyle/compute/model_ops.h"
#include "tensor_test_support.h"

namespace {
using namespace apple_tensor_test;
using Lease = hyle::compute::ResultLease<a::Results>;

Lease execute(a::Program &program, std::span<const mx::array> inputs) {
  auto slot = program.create_invocation().value();
  EXPECT_TRUE(slot.bind(inputs).ok());
  auto work = slot.submit().value();
  return work.wait().value();
}
mx::array bfloat(std::span<const float> values, mx::Shape shape) {
  std::vector<std::uint16_t> encoded;
  for (float value : values) encoded.push_back(half_test::encode<B16>(value).bits);
  return ready(mx::view(mx::array(encoded.data(), std::move(shape)), mx::bfloat16, stream()));
}
std::vector<float> input(int rows, int k, int seed) {
  std::vector<float> x(rows * k);
  for (int row = 0; row < rows; ++row) for (int col = 0; col < k; ++col)
    x[row * k + col] = float((col * 13 + row * 5 + seed) % 31 - 15) / 16;
  return x;
}
void expect_source(const Lease &output, std::span<const float> expected) {
  const auto actual = values(output.get()->arrays()[0]);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_EQ(actual[i], expected[i]) << i;
}
a::Program mlx_linear(a::Context &context, std::span<const a::InputSpec> specs) {
  return context.compile([](const auto &inputs, mx::Stream s) {
    return std::vector{mx::matmul(mx::astype(inputs[0], mx::float32, s),
        mx::transpose(mx::astype(inputs[1], mx::float32, s), s), s)};
  }, specs).value();
}
a::Program mlx_grouped(a::Context &context, std::span<const a::InputSpec> specs) {
  return context.compile([](const auto &inputs, mx::Stream s) -> hyle::Result<std::vector<mx::array>> {
    // Reuse the public checked-index service, not either mathematical arm.
    const std::array index_inputs{mx::arange(inputs[0].shape(0), mx::int32, s), inputs[2]};
    HYLE_ASSIGN_OR_RETURN(auto checked, a::lower(c::Operation(c::Gather{}), index_inputs, s));
    auto ids = mx::astype(mx::reshape(checked[0], {-1}, s), mx::uint32, s);
    const int tokens = inputs[1].shape(0), slots = inputs[1].shape(1), k = inputs[1].shape(2);
    auto activation = mx::reshape(inputs[1], {tokens * slots, 1, k}, s);
    auto weights = mx::transpose(mx::astype(inputs[0], mx::float32, s), {0, 2, 1}, s);
    auto output = mx::gather_mm(activation, weights, std::nullopt, ids, false, s);
    return std::vector{mx::reshape(output, {tokens, slots, inputs[0].shape(1)}, s)};
  }, specs).value();
}

TEST(AppleMatureOverlap, BfloatLinearRealRouterDecodeAndMultiToken) {
  constexpr int n = 768, k = 8192;
  for (int rows : {1, 16}) {
    SCOPED_TRACE(rows);
    auto metal_context = a::Context::create().value();
    auto mlx_context = a::Context::create().value();
    const std::array specs{a::InputSpec{{rows, k}, mx::bfloat16}, a::InputSpec{{n, k}, mx::bfloat16}};
    auto metal = metal_context.prepare(c::model_ops::bfloat_linear_f32(), specs).value();
    auto mlx = mlx_linear(mlx_context, specs);
    std::optional<Lease> old_metal, old_mlx;
    std::vector<float> old_expected;
    for (int seed : {3, 19}) {
      std::vector<float> weight(n * k), expected(rows * n);
      const auto x = input(rows, k, seed);
      for (int row = 0; row < n; ++row) {
        const int col0 = (row * 17 + seed) % k, col1 = (col0 + k / 2 + 7) % k;
        const float a = float(1 + row % 7) / 8, b = -float(1 + row % 5) / 16;
        weight[row * k + col0] = a; weight[row * k + col1] = b;
        for (int token = 0; token < rows; ++token)
          expected[token * n + row] = float(double(x[token * k + col0]) * a + double(x[token * k + col1]) * b);
      }
      const std::array args{bfloat(x, {rows, k}), bfloat(weight, {n, k})};
      auto gm = execute(metal, args), ml = execute(mlx, args);
      expect_source(gm, expected); expect_source(ml, expected);
      if (old_metal) {expect_source(*old_metal, old_expected); expect_source(*old_mlx, old_expected);}
      old_metal = std::move(gm); old_mlx = std::move(ml); old_expected = std::move(expected);
    }
    ASSERT_TRUE(metal_context.close().ok()); ASSERT_TRUE(mlx_context.close().ok());
    expect_source(*old_metal, old_expected); expect_source(*old_mlx, old_expected);
  }
}

TEST(AppleMatureOverlap, GroupedRealOutputProjectionF32AndBfloatWeights) {
  constexpr int e = 8, n = 1024, k = 4096, tokens = 5, slots = 8;
  for (bool bf16 : {false, true}) {
    SCOPED_TRACE(bf16);
    auto metal_context = a::Context::create().value(); auto mlx_context = a::Context::create().value();
    const std::array specs{a::InputSpec{{e, n, k}, bf16 ? mx::bfloat16 : mx::float32},
        a::InputSpec{{tokens, slots, k}, mx::float32}, a::InputSpec{{tokens, slots}, mx::int32}};
    auto metal = metal_context.prepare(c::model_ops::grouped_matmul(), specs).value();
    auto mlx = mlx_grouped(mlx_context, specs);
    for (int seed : {3, 19}) {
      std::vector<float> weight(e * n * k), expected(tokens * slots * n);
      const auto x = input(tokens * slots, k, seed); std::vector<std::int32_t> ids(tokens * slots);
      for (int slot = 0; slot < tokens * slots; ++slot) ids[slot] = (slot * 3 + seed) % e;
      for (int expert = 0; expert < e; ++expert) for (int row = 0; row < n; ++row) {
        const int first = (row * 17 + expert * 13 + seed) % k, second = (first + k / 2 + 7) % k;
        const float a = float(1 + (row + expert) % 7) / 8, b = -float(1 + (row + expert) % 5) / 16;
        weight[(expert * n + row) * k + first] = a; weight[(expert * n + row) * k + second] = b;
        for (int slot = 0; slot < tokens * slots; ++slot) if (ids[slot] == expert)
          expected[slot * n + row] = float(double(x[slot * k + first]) * a + double(x[slot * k + second]) * b);
      }
      const auto w = bf16 ? bfloat(weight, {e, n, k}) : mx::array(weight.data(), {e, n, k});
      const std::array args{w, mx::array(x.data(), {tokens, slots, k}), mx::array(ids.data(), {tokens, slots})};
      expect_source(execute(metal, args), expected); expect_source(execute(mlx, args), expected);
    }
  }
}

TEST(AppleMatureOverlap, BfloatLinearFullF32ExponentRange) {
  constexpr int rows = 16, k = 128, n = 32;
  const std::array specs{a::InputSpec{{rows, k}, mx::bfloat16}, a::InputSpec{{n, k}, mx::bfloat16}};
  std::vector<float> x(rows * k), weight(n * k), expected(rows * n);
  for (int row = 0; row < rows; ++row) {
    x[row * k] = (row % 2 ? -1.0F : 1.0F) * 65536.0F;
    x[row * k + 1] = (row % 2 ? -1.0F : 1.0F) * std::ldexp(1.0F, -30);
  }
  for (int row = 0; row < n; ++row) {
    weight[row * k + row % 2] = 1;
    for (int token = 0; token < rows; ++token) expected[token * n + row] = x[token * k + row % 2];
  }
  for (bool native_mlx : {false, true}) {
    SCOPED_TRACE(native_mlx);
    auto context = a::Context::create().value();
    auto program = native_mlx ? mlx_linear(context, specs) :
        context.prepare(c::model_ops::bfloat_linear_f32(), specs).value();
    expect_source(execute(program, std::array{bfloat(x, {rows, k}), bfloat(weight, {n, k})}), expected);
  }
}

TEST(AppleMatureOverlap, GroupedCheckedIdsPreservePriorLease) {
  constexpr int e = 3, n = 4, k = 32;
  const std::array specs{a::InputSpec{{e, n, k}, mx::float32},
      a::InputSpec{{1, 2, k}, mx::float32}, a::InputSpec{{1, 2}, mx::int32}};
  std::vector<float> weight(e * n * k), x(2 * k, 1), expected(2 * n);
  for (int expert = 0; expert < e; ++expert) for (int row = 0; row < n; ++row)
    weight[(expert * n + row) * k] = float(expert + row + 1);
  const auto w = mx::array(weight.data(), {e, n, k}), activation = mx::array(x.data(), {1, 2, k});
  for (int row = 0; row < n; ++row) {expected[row] = float(row + 1); expected[n + row] = float(row + 3);}
  for (bool native_mlx : {false, true}) {
    SCOPED_TRACE(native_mlx);
    auto context = a::Context::create().value();
    auto program = native_mlx ? mlx_grouped(context, specs) :
        context.prepare(c::model_ops::grouped_matmul(), specs).value();
    const std::array<std::int32_t, 2> good{0, 2};
    auto old = execute(program, std::array{w, activation, mx::array(good.data(), {1, 2})});
    expect_source(old, expected);
    for (int bad : {-1, e}) {
      auto slot = program.create_invocation().value();
      const std::array<std::int32_t, 2> ids{0, bad};
      ASSERT_TRUE(slot.bind(std::array{w, activation, mx::array(ids.data(), {1, 2})}).ok());
      auto work = slot.submit().value(); auto failure = work.wait();
      ASSERT_FALSE(failure); EXPECT_EQ(failure.error().code(), hyle::StatusCode::OutOfRange);
      expect_source(old, expected);
    }
    ASSERT_TRUE(context.close().ok()); expect_source(old, expected);
  }
}
} // namespace
