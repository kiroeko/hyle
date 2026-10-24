#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cublas.h"

namespace {

namespace c = hyle::compute;

TEST(CudaDefaultCompute, BfloatLinearF32UsesAdmittedMatrixRoute) {
  auto context = c::cuda::create_context();
  ASSERT_TRUE(context) << context.error().message();
  auto memory = c::cuda::MemoryResource::create(
      *context, c::cuda::MemoryKind::kDevice);
  ASSERT_TRUE(memory) << memory.error().message();
  const std::array inputs{
      c::TensorDesc::dense(c::DataType::kBFloat16,
                           std::array<std::int64_t, 2>{3, 8}).value(),
      c::TensorDesc::dense(c::DataType::kBFloat16,
                           std::array<std::int64_t, 2>{7, 8}).value()};
  const std::array outputs{
      c::TensorDesc::dense(c::DataType::kFloat32,
                           std::array<std::int64_t, 2>{3, 7}).value()};
  auto prepared = c::prepare(*context, c::model_ops::dense_linear_f32(),
                            inputs, outputs, memory->ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  ASSERT_EQ(prepared->provider().id, c::cublas::kProviderId);
  ASSERT_EQ(prepared->route_preference(), 99u);
  ASSERT_EQ(prepared->route_policy().version, 2u);
}

} // namespace
