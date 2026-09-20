#include "../compute_provider_ggml_cuda/tensor_test_support.h"
#include "../compute_provider_ggml_cuda/test_support.h"
#include "hyle/compute/cuda.h"

TEST(CudaDefault, OriginalGgmlThroughOrdinaryDirectEntry) {
  auto context = hyle::compute::cuda::create_context().value();
  ggml_cuda_test::exercise_direct(context);
}

TEST(CudaDefault, OriginalTensorBatchThroughOrdinaryDirectEntry) {
  auto context = hyle::compute::cuda::create_context().value();
  ggml_cuda_tensor_test::norms<float>(context);
  ggml_cuda_tensor_test::check_general_broadcast<float>(context);
}
