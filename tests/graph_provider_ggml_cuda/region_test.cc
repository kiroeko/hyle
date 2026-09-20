#include "hyle/graph/providers/ggml_cuda.h"
#include "region_test_support.h"
#include "tensor_test_support.h"

TEST(GgmlCudaGraph, WholeOriginalRegionUsesIndependentInvocations) {
  using namespace ggml_cuda_graph_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  exercise(
      context, false,
      [&](const graph::Graph &logical, mem::MemoryResourceRef resource) {
        const std::array storage{graph::StorageDomain{.resource = resource}};
        const std::array targets{graph::ExecutionTarget{
            .context = &context, .default_storage = graph::StorageDomainId{0}}};
        const std::array lanes{
            graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
        return graph::compile(
            logical,
            {.storage_domains = storage, .targets = targets, .lanes = lanes});
      });
}

TEST(GgmlCudaGraph,
     TensorRegionPreservesRoundingPublishedIntermediatesAndOldResults) {
  using namespace ggml_cuda_graph_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  const auto compile = [&](const graph::Graph &logical,
                           mem::MemoryResourceRef resource) {
    const std::array storage{graph::StorageDomain{.resource = resource}};
    const std::array targets{graph::ExecutionTarget{
        .context = &context, .default_storage = graph::StorageDomainId{0}}};
    const std::array lanes{
        graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
    return graph::compile(
        logical,
        {.storage_domains = storage, .targets = targets, .lanes = lanes});
  };
  exercise_tensor_graph<float>(context, compile);
  exercise_tensor_graph<hyle::simd::float16_t>(context, compile);
  exercise_tensor_graph<hyle::simd::bfloat16_t>(context, compile);
}

TEST(GgmlCudaGraph, VariadicPortsReserveMetadataEvenWhenValuesRepeat) {
  using namespace ggml_cuda_graph_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor input(memory.ref(), 1, 7), joined(memory.ref(), 1, 490),
      output(memory.ref(), 1, 490);
  const auto values = input.write(0);
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(input.description).value();
  std::array<graph::ValueId, 70> ports;
  ports.fill(x);
  auto node = builder.add_node(compute::Operation(compute::Concat{-1}), ports);
  ASSERT_TRUE(node) << node.error().message();
  const auto concat = builder.result(*node).value();
  const auto y =
      builder.add(compute::Unary{compute::UnaryKind::kNeg}, concat).value();
  ASSERT_TRUE(builder.publish(concat).ok());
  ASSERT_TRUE(builder.publish(y).ok());
  auto logical = std::move(builder).finish().value();
  const std::array storage{graph::StorageDomain{.resource = memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      .context = &context, .default_storage = graph::StorageDomainId{0}}};
  const std::array lanes{
      graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
  auto compiled = graph::compile(
      logical,
      {.storage_domains = storage, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1u);
  auto invocation = compiled->executable.create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();
  ASSERT_TRUE(invocation->bind(x, input.view.as_const()).ok());
  ASSERT_TRUE(invocation->bind(concat, joined.view).ok());
  ASSERT_TRUE(invocation->bind(y, output.view).ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      graph::QueueBinding{graph::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(invocation->prepare_replay(queues).ok());
  auto done = invocation->submit(queue);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  const auto positive = joined.read(), negative = output.read();
  for (std::size_t i = 0; i < positive.size(); ++i) {
    EXPECT_EQ(positive[i], values[i % 7]);
    EXPECT_EQ(negative[i], -values[i % 7]);
  }
}

TEST(GgmlCudaGraph, TypedSliceCopyBindsBeforeAlignedNativeMath) {
  using namespace ggml_cuda_graph_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor backing(memory.ref(), 1, 8), copied(memory.ref(), 1, 7),
      output(memory.ref(), 1, 7), unaligned_output(memory.ref(), 1, 8);
  const auto values = backing.write(2);
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(copied.description).value();
  const auto copy = builder.add(compute::Copy{}, x).value();
  const auto y =
      builder.add(compute::Unary{compute::UnaryKind::kNeg}, copy).value();
  ASSERT_TRUE(builder.publish(copy).ok());
  ASSERT_TRUE(builder.publish(y).ok());
  auto logical = std::move(builder).finish().value();
  const std::array storage{graph::StorageDomain{.resource = memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      .context = &context, .default_storage = graph::StorageDomainId{0}}};
  const std::array lanes{
      graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
  auto compiled = graph::compile(
      logical,
      {.storage_domains = storage, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1u);
  auto queue = context.create_queue().value();
  const std::array queues{
      graph::QueueBinding{graph::ExecutionLaneId{0}, &queue}};
  const auto input =
      compute::ConstTensorView::create(
          copied.description, backing.storage.view()
                                  .subview(sizeof(float), 7 * sizeof(float))
                                  .value())
          .value();
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(x, input).ok());
  ASSERT_TRUE(invocation.bind(copy, copied.view).ok());
  ASSERT_TRUE(invocation.bind(y, output.view).ok());
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  auto done = invocation.submit(queue);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  const auto a = copied.read(), b = output.read();
  for (std::size_t i = 0; i < 7; ++i) {
    EXPECT_EQ(a[i], values[i + 1]);
    EXPECT_EQ(b[i], -values[i + 1]);
  }
  // The published copy is also a math input, so writer alignment alone must
  // not permit binding it at a four-byte offset.
  auto rejected = compiled->executable.create_invocation().value();
  ASSERT_TRUE(rejected.bind(x, input).ok());
  ASSERT_TRUE(rejected.bind(y, output.view).ok());
  const auto bad_output =
      compute::TensorView::create(copied.description,
                                  unaligned_output.storage.view()
                                      .subview(sizeof(float), 7 * sizeof(float))
                                      .value())
          .value();
  const auto bound = rejected.bind(copy, bad_output);
  if (bound.ok()) {
    ASSERT_TRUE(rejected.prepare_replay(queues).ok());
    auto attempt = rejected.submit(queue);
    EXPECT_TRUE(!attempt || !attempt->wait().ok());
  } else {
    EXPECT_TRUE(bound.IsInvalidArgument());
  }
}
