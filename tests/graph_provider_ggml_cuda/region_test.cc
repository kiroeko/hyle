#include "hyle/graph/providers/ggml_cuda.h"
#include "../model_fragment_test_support.h"
#include "../sparse_fragment_test_support.h"
#include "region_test_support.h"
#include "softmax_test_support.h"
#include "tensor_test_support.h"

#include <ggml.h>

#include <algorithm>

#include "hyle/compute/low_precision.h"

TEST(GgmlCudaGraph, NativeQ8GraphImportsGgufBytesAndReplays) {
  using namespace ggml_cuda_graph_test;
  constexpr std::int64_t reduction = 256;
  constexpr std::int64_t columns = 2;
  const auto left_desc =
      ggml_cuda_tensor_test::dense(compute::DataType::kFloat32, {1, reduction});
  const auto right_desc = compute::gguf_weight_description(
      compute::GgufQuantizedType::kQ8_0, compute::DataType::kFloat32, reduction,
      columns);
  ASSERT_TRUE(right_desc) << right_desc.error().message();
  const auto output_desc =
      ggml_cuda_tensor_test::dense(compute::DataType::kFloat32, {1, columns});
  auto builder = graph::GraphBuilder::create().value();
  const auto left = builder.input(left_desc).value();
  const auto right = builder.input(*right_desc).value();
  const auto output = builder.add(compute::MatMul{}, left, right).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const std::array domains{graph::StorageDomain{.resource = memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      .context = &context, .default_storage = graph::StorageDomainId{0}}};
  const std::array lanes{
      graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
  auto compiled = graph::compile(
      std::move(builder).finish().value(),
      {.storage_domains = domains, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1U);
  EXPECT_EQ(compiled->report.step(0)->provider.id,
            compute::ggml_cuda::kProviderId);

  std::vector<float> left_values(static_cast<std::size_t>(reduction), 0.25F);
  std::vector<float> physical_weights(
      static_cast<std::size_t>(reduction * columns));
  std::fill_n(physical_weights.begin(), reduction, 1.0F);
  std::fill_n(physical_weights.begin() + reduction, reduction, -2.0F);
  std::vector<std::byte> packed(*right_desc->required_bytes());
  ASSERT_EQ(ggml_quantize_chunk(GGML_TYPE_Q8_0, physical_weights.data(),
                                packed.data(), 0, columns, reduction, nullptr),
            packed.size());
  auto left_buffer =
      ggml_cuda_tensor_test::upload<float>(memory.ref(), left_values);
  auto right_buffer =
      ggml_cuda_tensor_test::upload<std::byte>(memory.ref(), packed);
  auto output_buffer =
      ggml_cuda_tensor_test::output_buffer(memory.ref(), output_desc);
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(left, compute::ConstTensorView::create(
                                  left_desc, left_buffer.view())
                                  .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(right, compute::ConstTensorView::create(
                                   *right_desc, right_buffer.view())
                                   .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(output, compute::TensorView::create(
                                    output_desc, output_buffer.view())
                                    .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      graph::QueueBinding{.lane = graph::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(output_buffer);
    const auto scale = repeat == 0 ? 1.0F : 2.0F;
    ASSERT_EQ(actual.size(), 2U);
    EXPECT_NEAR(actual[0], 64.0F * scale, 0.5F);
    EXPECT_NEAR(actual[1], -128.0F * scale, 0.5F);
    if (repeat == 0) {
      std::fill(left_values.begin(), left_values.end(), 0.5F);
      ASSERT_TRUE(
          left_buffer.view().write(std::as_bytes(std::span(left_values))).ok());
    }
  }
}

TEST(GgmlCudaGraph, MatureMoeAndHyperConnectionBatchUsesOrdinaryGraphEntry) {
  using namespace ggml_cuda_graph_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  for (const auto &test : model_fragment_test::cases()) {
    SCOPED_TRACE(test.name);
    std::vector<mem::Buffer> storage;
    std::vector<graph::ValueId> inputs;
    auto builder = graph::GraphBuilder::create().value();
    storage.reserve(test.inputs.size());
    inputs.reserve(test.inputs.size());
    for (std::size_t i = 0; i < test.inputs.size(); ++i) {
      storage.push_back(mem::Buffer::create(
                            memory.ref(), test.inputs[i].required_bytes().value(),
                            256)
                            .value());
      ASSERT_TRUE(storage.back().view().write(test.payloads[i]).ok());
      inputs.push_back(builder.input(test.inputs[i]).value());
    }
    const auto node = builder.add_node(test.operation, inputs).value();
    const auto output = builder.result(node).value();
    ASSERT_TRUE(builder.publish(output).ok());
    auto logical = std::move(builder).finish().value();
    const std::array domains{
        graph::StorageDomain{.resource = memory.ref()}};
    const std::array targets{graph::ExecutionTarget{
        .context = &context, .default_storage = graph::StorageDomainId{0}}};
    const std::array lanes{
        graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
    auto compiled = graph::compile(
        logical,
        {.storage_domains = domains, .targets = targets, .lanes = lanes});
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_EQ(compiled->report.step_count(), 1U);
    EXPECT_EQ(compiled->report.step(0)->provider.id,
              compute::ggml_cuda::kProviderId);
    auto invocation = compiled->executable.create_invocation().value();
    for (std::size_t i = 0; i < inputs.size(); ++i)
      ASSERT_TRUE(invocation
                      .bind(inputs[i], compute::ConstTensorView::create(
                                           test.inputs[i], storage[i].view())
                                           .value())
                      .ok());
    auto result = mem::Buffer::create(memory.ref(),
                                      test.output.required_bytes().value(), 256)
                      .value();
    ASSERT_TRUE(invocation
                    .bind(output, compute::TensorView::create(
                                      test.output, result.view())
                                      .value())
                    .ok());
    auto queue = context.create_queue().value();
    const std::array queues{graph::QueueBinding{
        .lane = graph::ExecutionLaneId{0}, .queue = &queue}};
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(result);
    ASSERT_EQ(actual.size(), test.expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], test.expected[i], 2.0e-5F) << i;
  }
}

TEST(GgmlCudaGraph, MatureLightningIndexerUsesOrdinaryGraphEntry) {
  using namespace ggml_cuda_graph_test;
  const sparse_fragment_test::Fixture test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto builder = graph::GraphBuilder::create().value();
  std::array<graph::ValueId, 4> inputs;
  std::vector<mem::Buffer> storage;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    inputs[i] = builder.input(test.inputs[i]).value();
    storage.push_back(mem::Buffer::create(
                          memory.ref(), test.inputs[i].required_bytes().value(),
                          256)
                          .value());
    ASSERT_TRUE(storage.back().view().write(test.payloads[i]).ok());
  }
  const auto node = builder.add_node(test.operation, inputs).value();
  const auto output = builder.result(node).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto logical = std::move(builder).finish().value();
  const std::array domains{graph::StorageDomain{.resource = memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      .context = &context, .default_storage = graph::StorageDomainId{0}}};
  const std::array lanes{
      graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
  auto compiled = graph::compile(
      logical,
      {.storage_domains = domains, .targets = targets, .lanes = lanes})
                      .value();
  ASSERT_EQ(compiled.report.step_count(), 1U);
  EXPECT_EQ(compiled.report.step(0)->provider.id,
            compute::ggml_cuda::kProviderId);
  auto invocation = compiled.executable.create_invocation().value();
  for (std::size_t i = 0; i < inputs.size(); ++i)
    ASSERT_TRUE(invocation
                    .bind(inputs[i], compute::ConstTensorView::create(
                                         test.inputs[i], storage[i].view())
                                         .value())
                    .ok());
  auto result = mem::Buffer::create(
                    memory.ref(), test.output.required_bytes().value(), 256)
                    .value();
  ASSERT_TRUE(invocation
                  .bind(output, compute::TensorView::create(
                                    test.output, result.view())
                                    .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{graph::QueueBinding{
      .lane = graph::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(result);
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], test.expected[i], 2.0e-5F) << i;
  }
}

TEST(GgmlCudaGraph, SoftmaxRegionPublishesLogicalAxesAndRetainsOldResults) {
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
  exercise_softmax_graph<float>(context, compile);
  exercise_softmax_graph<hyle::simd::float16_t>(context, compile);
  exercise_softmax_graph<hyle::simd::bfloat16_t>(context, compile);
}

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
