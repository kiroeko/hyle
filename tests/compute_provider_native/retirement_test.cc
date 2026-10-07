#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "../compute/native_context_fixture.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"

namespace {
using namespace hyle::compute;
TensorDesc dense(DataType type, std::initializer_list<std::int64_t> shape) {
  return TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

TEST(CpuNativeRetirement, FourteenRedundantKindsDeclineBeforeExecution) {
  auto context = test::native_context().value();
  const auto values = dense(DataType::kFloat32, {2, 3});
  const auto parameter = dense(DataType::kFloat32, {3});
  const auto index = dense(DataType::kInt32, {1});
  const auto update = dense(DataType::kFloat32, {1, 3});
  const auto boolean = dense(DataType::kBool, {2, 3});
  const std::array unary{UnaryKind::kAbs, UnaryKind::kSigmoid, UnaryKind::kSoftplus};
  for (auto kind : unary) {
    auto prepared = prepare(context, Unary{kind}, values);
    ASSERT_FALSE(prepared); EXPECT_TRUE(prepared.error().IsUnimplemented());
  }
  for (auto kind : {BinaryKind::kMultiply, BinaryKind::kMaximum}) {
    auto prepared = prepare(context, Binary{kind}, values, values);
    ASSERT_FALSE(prepared); EXPECT_TRUE(prepared.error().IsUnimplemented());
  }
  for (auto kind : {ReductionKind::kSum, ReductionKind::kMean}) {
    auto prepared = prepare(context, Reduce{.kind = kind, .accumulation_type = DataType::kFloat32}, values);
    ASSERT_FALSE(prepared); EXPECT_TRUE(prepared.error().IsUnimplemented());
  }
  auto select = prepare(context, Select{}, boolean, values, values);
  ASSERT_FALSE(select); EXPECT_TRUE(select.error().IsUnimplemented());
  auto norm = prepare(context, LayerNorm{}, values, parameter);
  ASSERT_FALSE(norm); EXPECT_TRUE(norm.error().IsUnimplemented());
  auto logarithm = prepare(context, Softmax{.kind = SoftmaxKind::kLogSoftmax}, values);
  ASSERT_FALSE(logarithm); EXPECT_TRUE(logarithm.error().IsUnimplemented());
  for (auto kind : {ScatterKind::kAssign, ScatterKind::kAdd}) {
    auto prepared = prepare(context, Scatter{.kind = kind}, values, index, update);
    ASSERT_FALSE(prepared); EXPECT_TRUE(prepared.error().IsUnimplemented());
  }
  const std::array pair{values, values};
  auto concat = prepare(context, Concat{}, pair);
  ASSERT_FALSE(concat); EXPECT_TRUE(concat.error().IsUnimplemented());
  auto rope = prepare(context, RoPE{.rotary_dimensions = 2, .sequence_axis = 1},
                      dense(DataType::kFloat32, {1, 2, 1, 4}),
                      dense(DataType::kInt32, {2}), dense(DataType::kFloat32, {1}));
  ASSERT_FALSE(rope); EXPECT_TRUE(rope.error().IsUnimplemented());
}

TEST(CpuNativeRetirement, MatureEdgeMathPreservesOddBytesAndChangedReplay) {
  auto context = Context::create(cpu::device(), aten_cpu::provider()).value();
  auto queue = context.create_queue().value();
  const auto desc = TensorDesc::strided_bytes(
      DataType::kFloat32, std::array<std::int64_t, 2>{2, 3},
      std::array<std::int64_t, 2>{15, 5}).value();
  const auto out_desc = dense(DataType::kFloat32, {2, 3});
  auto input = hyle::mem::Buffer::create(hyle::mem::system_resource(), 29, 64).value();
  auto rhs = hyle::mem::Buffer::create(hyle::mem::system_resource(), 24, 64).value();
  auto output = hyle::mem::Buffer::create(hyle::mem::system_resource(), 24, 64).value();
  const auto source = ConstTensorView::create(desc, input.view()).value();
  const auto second = ConstTensorView::create(out_desc, rhs.view()).value();
  const auto destination = TensorView::create(out_desc, output.view()).value();
  const std::array unary{UnaryKind::kAbs, UnaryKind::kNeg, UnaryKind::kExp,
                        UnaryKind::kLog, UnaryKind::kSqrt, UnaryKind::kRsqrt,
                        UnaryKind::kTanh, UnaryKind::kRelu};
  for (auto kind : unary) {
    auto prepared = prepare(context, Unary{kind}, desc);
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, aten_cpu::kProviderId);
    auto invocation = prepared->create_invocation().value();
    for (float value : {0.25F, 2.0F}) {
      std::array<std::byte, 29> bytes{};
      for (std::size_t i = 0; i < 6; ++i) std::memcpy(bytes.data() + i * 5, &value, 4);
      ASSERT_TRUE(input.view().write(bytes).ok());
      auto completion = invocation.submit(queue, source, destination);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      std::array<float, 6> actual{};
      ASSERT_TRUE(output.view().read(std::as_writable_bytes(std::span(actual))).ok());
      const float expected = kind == UnaryKind::kAbs ? std::abs(value)
          : kind == UnaryKind::kNeg ? -value : kind == UnaryKind::kExp ? std::exp(value)
          : kind == UnaryKind::kLog ? std::log(value) : kind == UnaryKind::kSqrt ? std::sqrt(value)
          : kind == UnaryKind::kRsqrt ? 1.F / std::sqrt(value) : kind == UnaryKind::kTanh ? std::tanh(value) : value;
      for (float result : actual) EXPECT_NEAR(result, expected, 3e-5F);
      EXPECT_FALSE(invocation.in_flight());
    }
  }
  for (auto kind : {BinaryKind::kAdd, BinaryKind::kSubtract, BinaryKind::kMultiply, BinaryKind::kDivide}) {
    auto prepared = prepare(context, Binary{kind}, desc, out_desc);
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, aten_cpu::kProviderId);
    auto invocation = prepared->create_invocation().value();
    const std::array<float, 6> right{4, 4, 4, 4, 4, 4};
    ASSERT_TRUE(rhs.view().write(std::as_bytes(std::span(right))).ok());
    const std::array bindings{source, second};
    auto completion = invocation.submit(queue, bindings, destination);
    ASSERT_TRUE(completion); ASSERT_TRUE(completion->wait().ok());
    std::array<float, 6> actual{};
    ASSERT_TRUE(output.view().read(std::as_writable_bytes(std::span(actual))).ok());
    const float expected = kind == BinaryKind::kAdd ? 6.F : kind == BinaryKind::kSubtract ? -2.F : kind == BinaryKind::kMultiply ? 8.F : .5F;
    for (float result : actual) EXPECT_FLOAT_EQ(result, expected);
  }
}
} // namespace
