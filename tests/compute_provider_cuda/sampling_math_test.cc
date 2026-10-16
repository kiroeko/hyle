#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include "../compute_models/sampling_math_fixture.h"

cudaError_t hyle_g08_cuda_math_probe(const std::uint32_t *,float *,std::size_t);

TEST(CudaSamplingMath, ActualProductionTransformAtOpen64EndpointsAndCorpus) {
  const auto words=sampling_test::endpoint_words();
  std::vector<float> actual(words.size()/2);
  const auto status=hyle_g08_cuda_math_probe(words.data(),actual.data(),actual.size());
  ASSERT_EQ(status,cudaSuccess)<<cudaGetErrorString(status);
  EXPECT_NO_THROW(sampling_test::check_endpoints(words,actual));
}
