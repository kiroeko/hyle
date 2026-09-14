#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "hyle/compute/cpu.h"
#include "provider_internal.h"

namespace {

TEST(NativeProviderShape, DoesNotMistakeUpperPointSupportForABoundedArtifact) {
  using namespace hyle::compute;
  using namespace hyle::compute::detail;
  auto context = cpu::create_context().value();
  const std::array<std::int64_t, 1> shape{8};
  const std::array descriptions{
      TensorDesc::dense(DataType::kFloat32, shape).value()};
  std::array ranges{ProviderShapeRange{1, 8, 1}};
  std::array<ProviderTensorAxes, 1> axes{};
  axes[0].symbols[0] = 0;
  const ProviderShapeDomain domain{ranges, axes, axes};
  std::array<ResourceBindingRequirement, 1> inputs{}, outputs{};
  const auto propose = [&]() {
    return OperationPreparationAccess::propose(
        context, 0, Operation(Unary{UnaryKind::kNeg}), descriptions,
        descriptions, inputs, outputs, domain);
  };
  auto bounded = propose();
  ASSERT_FALSE(bounded.has_value());
  EXPECT_TRUE(bounded.error().IsUnimplemented());
  ranges[0].minimum = 8;
  auto exact = propose();
  ASSERT_TRUE(exact.has_value()) << exact.error().message();
  EXPECT_EQ(exact->shape_coverage, ProviderShapeCoverage::kExactPoint);
}

} // namespace
