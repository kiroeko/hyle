#include <array>

#include <gtest/gtest.h>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/providers/onednn.h"

namespace {

TEST(OneDnnComposition, UnsupportedShapeAdvancesToExplicitNativeProvider) {
  const std::array providers{
      hyle::compute::ProviderBinding{hyle::compute::onednn::provider(), 100},
      hyle::compute::ProviderBinding{hyle::compute::native::provider(), 200}};
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                {.providers = providers});
  ASSERT_TRUE(context.has_value()) << context.error().message();

  const std::array<std::int64_t, 2> left_shape{1, 17};
  // Transposed storage is a genuine BRGEMM adapter limit, unlike M1.
  const std::array<std::int64_t, 2> right_shape{65, 17};
  const std::array<std::int64_t, 2> output_shape{1, 65};
  const auto left = hyle::compute::TensorDesc::dense(
      hyle::compute::DataType::kFloat32, left_shape);
  const auto right = hyle::compute::TensorDesc::dense(
      hyle::compute::DataType::kFloat32, right_shape);
  const auto output = hyle::compute::TensorDesc::dense(
      hyle::compute::DataType::kFloat32, output_shape);
  ASSERT_TRUE(left.has_value());
  ASSERT_TRUE(right.has_value());
  ASSERT_TRUE(output.has_value());

  auto prepared =
      hyle::compute::prepare(*context, hyle::compute::MatMul{}, *left,
                             right->transposed().value(), *output);
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, hyle::compute::native::kProviderId);
}

} // namespace
