#include "../compute_provider_ggml_cuda/test_support.h"
#include "hyle/compute/cuda.h"

TEST(CudaDefault, OriginalGgmlThroughOrdinaryDirectEntry) {
  auto context = hyle::compute::cuda::create_context().value();
  ggml_cuda_test::exercise_direct(context);
}
