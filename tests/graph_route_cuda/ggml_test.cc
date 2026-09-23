#include "../graph_provider_ggml_cuda/region_test_support.h"
#include "../graph_provider_ggml_cuda/softmax_test_support.h"
#include "../graph_provider_ggml_cuda/tensor_test_support.h"
#include "hyle/graph/cuda.h"

TEST(CudaDefaultGraph, OriginalTensorRegionThroughOrdinaryEntry) {
  using namespace ggml_cuda_graph_test;
  auto context = graph::cuda::create_context().value();
  exercise_tensor_graph<float>(context, [&](const graph::Graph &logical,
                                            mem::MemoryResourceRef resource) {
    return graph::cuda::compile(logical, context, resource);
  });
}

TEST(CudaDefaultGraph, OriginalSoftmaxRegionThroughOrdinaryEntry) {
  using namespace ggml_cuda_graph_test;
  auto context = graph::cuda::create_context().value();
  exercise_softmax_graph<float>(context, [&](const graph::Graph &logical,
                                             mem::MemoryResourceRef resource) {
    return graph::cuda::compile(logical, context, resource);
  });
}

TEST(CudaDefaultGraph, OriginalGgmlRegionComposesWithCublas) {
  using namespace ggml_cuda_graph_test;
  auto context = graph::cuda::create_context().value();
  exercise(context, true,
           [&](const graph::Graph &logical, mem::MemoryResourceRef resource) {
             return graph::cuda::compile(logical, context, resource);
           });
}

TEST(CudaDefaultGraph, MatureHalfLinearAndHadamardUseOneGgmlRegion) {
  using namespace ggml_cuda_graph_test;
  using F16 = hyle::simd::float16_t;
  auto context = graph::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto left_desc = compute::TensorDesc::dense(
                             compute::DataType::kFloat16,
                             std::array<std::int64_t, 3>{1, 2, 3})
                             .value();
  const auto right_desc = compute::TensorDesc::dense(
                              compute::DataType::kFloat16,
                              std::array<std::int64_t, 3>{1, 3, 4})
                              .value();
  const auto output_desc = compute::TensorDesc::dense(
                               compute::DataType::kFloat16,
                               std::array<std::int64_t, 3>{1, 2, 4})
                               .value();
  auto builder = graph::GraphBuilder::create().value();
  const auto left = builder.input(left_desc).value();
  const auto right = builder.input(right_desc).value();
  const auto projected = builder.add(compute::MatMul{}, left, right).value();
  const auto output = builder.add(compute::Hadamard{}, projected).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto logical = std::move(builder).finish().value();
  auto compiled = graph::cuda::compile(logical, context, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1U);
  EXPECT_EQ(compiled->report.step(0)->provider.id,
            compute::ggml_cuda::kProviderId);
  EXPECT_EQ(compiled->report.step(0)->node_count, 2U);

  std::array<F16, 6> left_values{};
  std::array<F16, 12> right_values{};
  for (std::size_t i = 0; i < left_values.size(); ++i)
    left_values[i] = half_test::encode<F16>(static_cast<double>(i + 1));
  for (std::size_t i = 0; i < right_values.size(); ++i)
    right_values[i] = half_test::encode<F16>(static_cast<double>(i + 1));
  auto left_buffer =
      ggml_cuda_tensor_test::upload<F16>(memory.ref(), left_values);
  auto right_buffer =
      ggml_cuda_tensor_test::upload<F16>(memory.ref(), right_values);
  auto result =
      ggml_cuda_tensor_test::output_buffer(memory.ref(), output_desc);
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(left, compute::ConstTensorView::create(
                                  left_desc, left_buffer.view())
                                  .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(right, compute::ConstTensorView::create(
                                   right_desc, right_buffer.view())
                                   .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(output, compute::TensorView::create(output_desc,
                                                            result.view())
                                    .value())
                  .ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(graph::cuda::prepare_replay(invocation, queue).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  const auto actual = half_test::read<F16>(result);
  const std::array expected{94.0, -6.0, -12.0, 0.0,
                            211.0, -15.0, -30.0, 0.0};
  for (std::size_t i = 0; i < actual.size(); ++i)
    half_test::expect_near(actual[i], expected[i]);
}

TEST(CudaDefaultGraph, UnrelatedCublasNodeDoesNotSplitGgmlRegion) {
  using namespace ggml_cuda_graph_test;
  auto context = graph::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor input(memory.ref()), weights(memory.ref(), 128, 16);
  Tensor residual(memory.ref()), projected(memory.ref(), 4, 16);
  const auto values = input.write(3);
  const auto weight_values = weights.write(1);
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(input.description).value();
  const auto w = builder.input(weights.description).value();
  const auto norm = builder.add(compute::RmsNorm{}, x).value();
  const auto independent = builder.add(compute::MatMul{}, x, w).value();
  const auto y = builder.add(compute::Binary{}, norm, x).value();
  ASSERT_TRUE(builder.publish(y).ok());
  ASSERT_TRUE(builder.publish(independent).ok());
  auto logical = std::move(builder).finish().value();
  auto compiled = graph::cuda::compile(logical, context, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 2u);
  EXPECT_TRUE(compiled->report.mixed_providers());
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(x, input.view.as_const()).ok());
  ASSERT_TRUE(invocation.bind(w, weights.view.as_const()).ok());
  ASSERT_TRUE(invocation.bind(y, residual.view).ok());
  ASSERT_TRUE(invocation.bind(independent, projected.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(graph::cuda::prepare_replay(invocation, queue).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  check_norm(values, residual.read(), true);
  const auto actual = projected.read();
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 16; ++col) {
      double expected = 0;
      for (std::size_t k = 0; k < 128; ++k)
        expected += double(values[row * 128 + k]) * weight_values[k * 16 + col];
      EXPECT_NEAR(actual[row * 16 + col], expected, 0.002);
    }
  }
}
