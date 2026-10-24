#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <vector>

#include "fixture.h"
#include "hyle/compute/model_ops.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;

c::TensorDesc description(c::DataType type,
                          std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

TEST(BatchedBfloatVulkan, OriginalFloat32RowsAndBatchOffsets) {
  constexpr std::int64_t batches = 3, rows = 2, width = 66, columns = 4;
  auto context = vulkan_test::context().value();
  ASSERT_FALSE(context.environment().software_device);
  const auto ad = description(c::DataType::kFloat32, {batches, rows, width});
  const auto wd = description(c::DataType::kBFloat16, {batches, columns, width});
  const auto yd = description(c::DataType::kFloat32, {batches, rows, columns});
  std::vector<float> a(std::size_t(batches * rows * width), 0.F);
  std::vector<std::uint16_t> w(std::size_t(batches * columns * width), 0);
  for (std::int64_t b = 0; b < batches; ++b) {
    for (std::int64_t m = 0; m < rows; ++m) {
      const auto start = std::size_t((b * rows + m) * width);
      a[start] = 1.F + float((b + m) % 3 + 1) / 262144.F;
      a[start + 1] = -1.F;
    }
    for (std::int64_t n = 0; n < columns; ++n) {
      const auto start = std::size_t((b * columns + n) * width);
      // These small integer coefficients are exact BF16. Their direct bit
      // representation is independent of product conversion/MatMul code.
      const float coefficient = float((b + 1) * (n + 1));
      w[start] = w[start + 1] = std::uint16_t(
          std::bit_cast<std::uint32_t>(coefficient) >> 16);
    }
  }
  const auto activation = context.upload(ad, std::as_bytes(std::span(a))).value();
  const auto weights = context.upload(wd, std::as_bytes(std::span(w))).value();
  const std::array<v::Argument, 2> arguments{ad, wd};
  auto prepared = v::prepare(context, c::model_ops::bfloat_linear_f32(), arguments);
  ASSERT_TRUE(prepared) << prepared.error().message();
  ASSERT_EQ(prepared->output_descriptions().size(), 1U);
  EXPECT_EQ(prepared->output_descriptions()[0], yd);
  auto invocation = prepared->create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();
  ASSERT_TRUE(invocation->bind(activation, weights).ok());
  auto work = invocation->submit();
  ASSERT_TRUE(work) << work.error().message();
  auto done = work->wait();
  ASSERT_TRUE(done) << done.error().message();
  const auto result = done->get()->tensors();
  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0].description(), yd);
  const auto actual = vulkan_test::read(context, result[0]).value();
  ASSERT_EQ(actual.size(), std::size_t(batches * rows * columns));
  for (std::int64_t b = 0; b < batches; ++b)
    for (std::int64_t m = 0; m < rows; ++m)
      for (std::int64_t n = 0; n < columns; ++n)
        EXPECT_EQ(actual[std::size_t((b * rows + m) * columns + n)],
                  float((b + 1) * (n + 1)) *
                      float((b + m) % 3 + 1) / 262144.F)
            << "batch=" << b << " row=" << m << " column=" << n;
  EXPECT_EQ(vulkan_test::read(context, activation).value(), a);
  std::vector<std::uint16_t> original_weight(w.size());
  ASSERT_TRUE(context.download(weights, std::as_writable_bytes(std::span(original_weight))).ok());
  EXPECT_EQ(original_weight, w);
}

TEST(BatchedBfloatVulkan, ColdDomainAndBfloatActivationRejection) {
  auto context = vulkan_test::context().value();
  // Actual Q32 compact key/value descriptors only: neither original 16MiB
  // weight bank is allocated, uploaded or submitted by this cold check.
  for (const auto nk : std::array{std::array<std::int64_t, 2>{512, 256},
                                 std::array<std::int64_t, 2>{256, 512}}) {
    const auto ad = description(c::DataType::kFloat32, {64, 1, nk[1]});
    const auto wd = description(c::DataType::kBFloat16, {64, nk[0], nk[1]});
    const auto yd = description(c::DataType::kFloat32, {64, 1, nk[0]});
    const std::array<v::Argument, 2> arguments{ad, wd};
    auto prepared = v::prepare(context, c::model_ops::bfloat_linear_f32(), arguments);
    ASSERT_TRUE(prepared) << prepared.error().message();
    ASSERT_EQ(prepared->output_descriptions().size(), 1U);
    EXPECT_EQ(prepared->output_descriptions()[0], yd);
  }
  const auto ad = description(c::DataType::kBFloat16, {3, 2, 66});
  const auto wd = description(c::DataType::kBFloat16, {3, 4, 66});
  const std::array descriptions{ad, wd};
  // The public schema permits BF16 activations; this backend's exact batched
  // native domain admits F32 A only and must reject BF16 A during prepare.
  const auto inferred = c::infer_single_output(c::model_ops::bfloat_linear_f32(), descriptions);
  ASSERT_TRUE(inferred);
  EXPECT_EQ(inferred->data_type(), c::DataType::kFloat32);
  const std::array<v::Argument, 2> arguments{ad, wd};
  const auto rejected = v::prepare(context, c::model_ops::bfloat_linear_f32(), arguments);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsUnimplemented());
  const std::array mismatched{
      description(c::DataType::kFloat32, {3, 2, 66}),
      description(c::DataType::kBFloat16, {1, 4, 66})};
  EXPECT_FALSE(c::infer_single_output(c::model_ops::bfloat_linear_f32(), mismatched));
}
} // namespace
