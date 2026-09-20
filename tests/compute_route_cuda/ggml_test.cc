#include "../compute_provider_cuda/collectives_test_support.h"
#include "../compute_provider_ggml_cuda/softmax_test_support.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"
#include "../compute_provider_ggml_cuda/test_support.h"
#include "hyle/compute/cuda.h"

TEST(CudaDefault, OriginalCcclThroughOrdinaryDirectEntry) {
  auto context = hyle::compute::cuda::create_context().value();
  cccl_test::scan<std::int32_t>(context, std::array<std::int64_t, 3>{2, 257, 3},
                                1, 1);
  cccl_test::topk<float>(context, std::array<std::int64_t, 2>{2, 8193}, -1, 0,
                         7);
}

TEST(CudaDefault, OriginalGgmlThroughOrdinaryDirectEntry) {
  auto context = hyle::compute::cuda::create_context().value();
  ggml_cuda_test::exercise_direct(context);
}

TEST(CudaDefault, OriginalTensorBatchThroughOrdinaryDirectEntry) {
  auto context = hyle::compute::cuda::create_context().value();
  ggml_cuda_tensor_test::norms<float>(context);
  ggml_cuda_tensor_test::check_general_broadcast<float>(context);
}

TEST(CudaDefault, OriginalSoftmaxThroughOrdinaryDirectEntry) {
  auto context = hyle::compute::cuda::create_context().value();
  ggml_cuda_tensor_test::softmax_case<float>(context, {2, 3, 17}, 1, true,
                                             false, true);
}
