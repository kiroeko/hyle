#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"

namespace {

using hyle::compute::ConstTensorView;
using hyle::compute::Context;
using hyle::compute::DataType;
using hyle::compute::Device;
using hyle::compute::MatMul;
using hyle::compute::Queue;
using hyle::compute::RmsNorm;
using hyle::compute::SiluAndMul;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::graph::CompiledGraph;
using hyle::graph::Executable;
using hyle::graph::Graph;
using hyle::graph::GraphBuilder;
using hyle::graph::Invocation;
using hyle::graph::ValueId;
using hyle::mem::AllocatorRef;
using hyle::mem::Buffer;

TensorDesc row(std::int64_t size) {
  const std::array shape{size};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

TensorDesc matrix(std::int64_t rows, std::int64_t columns) {
  const std::array shape{rows, columns};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

struct LogicalGraph final {
  Graph graph{};
  std::vector<ValueId> inputs{};
  std::vector<ValueId> outputs{};
  std::vector<ValueId> intermediates{};
};

LogicalGraph make_rms_graph(TensorDesc description, bool weighted = false) {
  GraphBuilder builder = GraphBuilder::create().value();
  LogicalGraph result;
  result.inputs.push_back(builder.input(description).value());
  ValueId output;
  if (weighted) {
    const TensorDesc weight = row(description.dimensions().back());
    result.inputs.push_back(builder.input(weight).value());
    output = builder.add(RmsNorm{}, result.inputs[0], result.inputs[1]).value();
  } else {
    output = builder.add(RmsNorm{}, result.inputs[0]).value();
  }
  EXPECT_TRUE(builder.output(output).ok());
  result.outputs.push_back(output);
  result.graph = std::move(builder).finish().value();
  return result;
}

LogicalGraph make_transformer_graph(std::size_t rows, std::size_t hidden,
                                    std::size_t ffn) {
  GraphBuilder builder = GraphBuilder::create().value();
  LogicalGraph result;
  const TensorDesc activation = matrix(static_cast<std::int64_t>(rows),
                                       static_cast<std::int64_t>(hidden));
  const TensorDesc norm_weight = row(static_cast<std::int64_t>(hidden));
  const TensorDesc stored_weight = matrix(static_cast<std::int64_t>(2 * ffn),
                                          static_cast<std::int64_t>(hidden));
  result.inputs.push_back(builder.input(activation).value());
  result.inputs.push_back(builder.input(norm_weight).value());
  result.inputs.push_back(
      builder.input(stored_weight.transposed().value()).value());
  const ValueId normalized =
      builder.add(RmsNorm{}, result.inputs[0], result.inputs[1]).value();
  const ValueId projected =
      builder.add(MatMul{}, normalized, result.inputs[2]).value();
  const ValueId activated = builder.add(SiluAndMul{}, projected).value();
  result.intermediates = {normalized, projected};
  result.outputs.push_back(activated);
  EXPECT_TRUE(builder.output(activated).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

LogicalGraph make_branch_graph() {
  GraphBuilder builder = GraphBuilder::create().value();
  LogicalGraph result;
  result.inputs.push_back(builder.input(matrix(2, 3)).value());
  result.inputs.push_back(builder.input(matrix(3, 2)).value());
  result.inputs.push_back(builder.input(matrix(3, 1)).value());
  result.outputs.push_back(
      builder.add(MatMul{}, result.inputs[0], result.inputs[1]).value());
  result.outputs.push_back(
      builder.add(MatMul{}, result.inputs[0], result.inputs[2]).value());
  EXPECT_TRUE(builder.output(result.outputs[0]).ok());
  EXPECT_TRUE(builder.output(result.outputs[1]).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

struct OwnedTensor final {
  Buffer buffer{};
  TensorView view{};
};

OwnedTensor make_tensor(TensorDesc description) {
  const std::size_t bytes = description.required_bytes().value();
  OwnedTensor result;
  result.buffer =
      Buffer::create(hyle::mem::system_resource(), bytes, 64).value();
  result.view = TensorView::create(description, result.buffer.view()).value();
  return result;
}

void write(OwnedTensor &tensor, std::span<const float> values) {
  ASSERT_TRUE(tensor.buffer.view().write(std::as_bytes(values)).ok());
}

std::vector<float> read(const OwnedTensor &tensor, std::size_t count) {
  std::vector<float> result(count);
  EXPECT_TRUE(tensor.buffer.view()
                  .read(std::as_writable_bytes(std::span(result)))
                  .ok());
  return result;
}

std::vector<float> expected_transformer(std::span<const float> input,
                                        std::span<const float> norm_weight,
                                        std::span<const float> stored_matrix,
                                        std::size_t rows, std::size_t hidden,
                                        std::size_t ffn) {
  std::vector<float> normalized(rows * hidden);
  for (std::size_t r = 0; r < rows; ++r) {
    float sum = 0.0F;
    for (std::size_t k = 0; k < hidden; ++k) {
      const float value = input[r * hidden + k];
      sum += value * value;
    }
    const float scale =
        1.0F / std::sqrt(sum / static_cast<float>(hidden) + RmsNorm{}.epsilon);
    for (std::size_t k = 0; k < hidden; ++k) {
      normalized[r * hidden + k] =
          input[r * hidden + k] * scale * norm_weight[k];
    }
  }

  std::vector<float> projected(rows * 2 * ffn);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t n = 0; n < 2 * ffn; ++n) {
      float sum = 0.0F;
      for (std::size_t k = 0; k < hidden; ++k) {
        sum += normalized[r * hidden + k] * stored_matrix[n * hidden + k];
      }
      projected[r * 2 * ffn + n] = sum;
    }
  }

  std::vector<float> output(rows * ffn);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t n = 0; n < ffn; ++n) {
      const float gate = projected[r * 2 * ffn + n];
      const float value = projected[r * 2 * ffn + ffn + n];
      output[r * ffn + n] = (gate / (1.0F + std::exp(-gate))) * value;
    }
  }
  return output;
}

void expect_near(std::span<const float> actual,
                 std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index],
                std::abs(expected[index]) * 5.0e-5F + 5.0e-6F)
        << "index " << index;
  }
}

struct AllocatorProbe final {
  AllocatorRef backing = hyle::mem::system_allocator();
  std::atomic<std::size_t> allocations{0};
  std::atomic<std::size_t> deallocations{0};
  std::atomic<bool> fail{false};
};

void *probe_allocate(void *opaque, std::size_t size,
                     std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (probe.fail.load())
    return nullptr;
  void *result = probe.backing.allocate(size, alignment);
  if (result != nullptr)
    ++probe.allocations;
  return result;
}

void probe_free(void *opaque, void *pointer, std::size_t size,
                std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (pointer != nullptr)
    ++probe.deallocations;
  probe.backing.deallocate(pointer, size, alignment);
}

void *probe_reallocate(void *opaque, void *pointer, std::size_t old_size,
                       std::size_t new_size, std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (probe.fail.load())
    return nullptr;
  return probe.backing.reallocate(pointer, old_size, new_size, alignment);
}

AllocatorRef probe_allocator(AllocatorProbe &probe) noexcept {
  return AllocatorRef(&probe_allocate, &probe_free, &probe_reallocate, &probe);
}

TEST(GraphCompile, CompilesEveryNodeAndReportsItsWorkspaceHonestly) {
  const LogicalGraph logical = make_transformer_graph(2, 4, 4);
  Context context = Context::create(Device::cpu()).value();
  CompiledGraph compiled = hyle::graph::compile(logical.graph, context).value();

  EXPECT_TRUE(compiled.executable.valid());
  EXPECT_EQ(compiled.executable.input_count(), 3u);
  EXPECT_EQ(compiled.executable.output_count(), 1u);
  EXPECT_EQ(compiled.executable.node_count(), 3u);
  EXPECT_EQ(compiled.executable.input_id(1).value(), logical.inputs[1]);
  EXPECT_EQ(compiled.executable.output_id().value(), logical.outputs[0]);
  EXPECT_EQ(compiled.report.backend, hyle::compute::BackendKind::kCpu);
  EXPECT_EQ(compiled.report.provider, hyle::compute::ProviderKind::kHyleNative);
  EXPECT_EQ(compiled.report.node_count, 3u);
  EXPECT_GT(compiled.report.workspace_bytes, 0u);
  EXPECT_GT(compiled.report.workspace_backing_count, 0u);
  EXPECT_EQ(compiled.report.workspace_bytes,
            compiled.executable.workspace_bytes());
  EXPECT_EQ(compiled.report.workspace_backing_count,
            compiled.executable.workspace_backing_count());
  EXPECT_EQ(compiled.report.copy_bytes, 0u);
  EXPECT_EQ(compiled.report.synchronization_count, 0u);
  EXPECT_FALSE(compiled.report.used_fallback);

  Executable copied = compiled.executable;
  Executable moved = std::move(copied);
  EXPECT_FALSE(copied.valid()); // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(moved.create_invocation().has_value());
}

TEST(Invocation, RunsNaturalTransformerFragmentAndReusesIt) {
  constexpr std::size_t kRows = 2;
  constexpr std::size_t kHidden = 4;
  constexpr std::size_t kFfn = 4;
  const LogicalGraph logical = make_transformer_graph(kRows, kHidden, kFfn);
  Context context = Context::create(Device::cpu()).value();
  Queue queue = context.create_queue().value();
  Invocation invocation = hyle::graph::compile(logical.graph, context)
                              ->executable.create_invocation()
                              .value();

  OwnedTensor input = make_tensor(matrix(kRows, kHidden));
  OwnedTensor norm_weight = make_tensor(row(kHidden));
  const TensorDesc stored_weight = matrix(2 * kFfn, kHidden);
  OwnedTensor matrix_weight = make_tensor(stored_weight.transposed().value());
  OwnedTensor output = make_tensor(matrix(kRows, kFfn));
  const std::vector<float> input_values{1, -2, 3, -4, 2, 1, -1, 0.5F};
  const std::vector<float> norm_values{1, 0.5F, 1.5F, -1};
  std::vector<float> matrix_values(2 * kFfn * kHidden);
  for (std::size_t index = 0; index < matrix_values.size(); ++index) {
    matrix_values[index] =
        static_cast<float>(static_cast<int>(index % 7) - 3) * 0.125F;
  }
  write(input, input_values);
  write(norm_weight, norm_values);
  write(matrix_weight, matrix_values);
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], norm_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], matrix_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());

  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value());
  ASSERT_TRUE(completion->wait().ok());
  const std::vector<float> expected = expected_transformer(
      input_values, norm_values, matrix_values, kRows, kHidden, kFfn);
  expect_near(read(output, kRows * kFfn), expected);

  invocation.reset();
  EXPECT_TRUE(invocation.submit(queue).error().IsFailedPrecondition());
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], norm_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], matrix_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
}

TEST(Invocation, ExecutesSharedInputBranchesWithIndependentOutputs) {
  const LogicalGraph logical = make_branch_graph();
  Context context = Context::create(Device::cpu()).value();
  Queue queue = context.create_queue().value();
  Invocation invocation = hyle::graph::compile(logical.graph, context)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor input = make_tensor(matrix(2, 3));
  OwnedTensor first_weight = make_tensor(matrix(3, 2));
  OwnedTensor second_weight = make_tensor(matrix(3, 1));
  OwnedTensor first_output = make_tensor(matrix(2, 2));
  OwnedTensor second_output = make_tensor(matrix(2, 1));
  const std::vector<float> input_values{1, 2, 3, 4, 5, 6};
  const std::vector<float> first_values{1, 0, 0, 1, 1, 1};
  const std::vector<float> second_values{1, 2, 3};
  write(input, input_values);
  write(first_weight, first_values);
  write(second_weight, second_values);
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], first_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], second_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], first_output.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[1], second_output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(first_output, 4), std::array{4.0F, 5.0F, 10.0F, 11.0F});
  expect_near(read(second_output, 2), std::array{14.0F, 32.0F});
}

TEST(Invocation, RejectsForeignInternalConstOutputAndWrongShapeBindings) {
  const LogicalGraph logical = make_transformer_graph(2, 4, 4);
  const LogicalGraph other = make_rms_graph(matrix(2, 4));
  Context context = Context::create(Device::cpu()).value();
  Invocation invocation = hyle::graph::compile(logical.graph, context)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor activation = make_tensor(matrix(2, 4));
  OwnedTensor wrong = make_tensor(matrix(1, 4));
  OwnedTensor output = make_tensor(matrix(2, 4));

  EXPECT_TRUE(
      invocation.bind(other.inputs[0], activation.view).IsInvalidArgument());
  EXPECT_TRUE(invocation.bind(logical.intermediates[0], activation.view)
                  .IsInvalidArgument());
  EXPECT_TRUE(invocation.bind(logical.outputs[0], output.view.as_const())
                  .IsInvalidArgument());
  EXPECT_TRUE(
      invocation.bind(logical.inputs[0], wrong.view).IsInvalidArgument());
  Queue queue = context.create_queue().value();
  EXPECT_TRUE(invocation.submit(queue).error().IsFailedPrecondition());
}

TEST(GraphCompile, ReportsInvalidClosedAndMetadataAllocationFailures) {
  Context context = Context::create(Device::cpu()).value();
  EXPECT_TRUE(
      hyle::graph::compile(Graph{}, context).error().IsInvalidArgument());

  const LogicalGraph logical = make_rms_graph(row(8));
  ASSERT_TRUE(context.close().ok());
  EXPECT_TRUE(hyle::graph::compile(logical.graph, context)
                  .error()
                  .IsFailedPrecondition());

  AllocatorProbe probe;
  Context oom_context =
      Context::create(Device::cpu(),
                      {.metadata_allocator = probe_allocator(probe)})
          .value();
  probe.fail = true;
  auto oom = hyle::graph::compile(logical.graph, oom_context);
  ASSERT_FALSE(oom.has_value());
  EXPECT_TRUE(oom.error().IsResourceExhausted());

  probe.fail = false;
  Executable executable =
      hyle::graph::compile(logical.graph, oom_context)->executable;
  probe.fail = true;
  auto invocation = executable.create_invocation();
  ASSERT_FALSE(invocation.has_value());
  EXPECT_TRUE(invocation.error().IsResourceExhausted());
}

TEST(Invocation, SubmitDoesNotAllocatePlanMetadata) {
  AllocatorProbe probe;
  Context context = Context::create(Device::cpu(), {.metadata_allocator =
                                                        probe_allocator(probe)})
                        .value();
  const LogicalGraph logical = make_transformer_graph(2, 4, 4);
  Executable executable =
      hyle::graph::compile(logical.graph, context)->executable;
  Invocation invocation = executable.create_invocation().value();
  OwnedTensor input = make_tensor(matrix(2, 4));
  OwnedTensor norm = make_tensor(row(4));
  OwnedTensor weights = make_tensor(matrix(8, 4).transposed().value());
  OwnedTensor output = make_tensor(matrix(2, 4));
  std::vector<float> input_values(8, 1.0F);
  std::vector<float> norm_values(4, 1.0F);
  std::vector<float> weight_values(32, 0.25F);
  write(input, input_values);
  write(norm, norm_values);
  write(weights, weight_values);
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], norm.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], weights.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());
  Queue queue = context.create_queue().value();
  const std::size_t before = probe.allocations.load();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(probe.allocations.load(), before);
}

TEST(Invocation, DistinctInvocationsRunConcurrentlyFromOneExecutable) {
  constexpr std::size_t kCount = 256;
  const LogicalGraph logical = make_rms_graph(row(kCount));
  Context context = Context::create(Device::cpu()).value();
  Executable executable =
      hyle::graph::compile(logical.graph, context)->executable;
  std::atomic<int> failures{0};

  auto worker = [&](float value) {
    Queue queue = context.create_queue().value();
    Invocation invocation = executable.create_invocation().value();
    OwnedTensor input = make_tensor(row(kCount));
    OwnedTensor output = make_tensor(row(kCount));
    std::vector<float> values(kCount, value);
    write(input, values);
    if (!invocation.bind(logical.inputs[0], input.view).ok() ||
        !invocation.bind(logical.outputs[0], output.view).ok()) {
      ++failures;
      return;
    }
    auto completion = invocation.submit(queue);
    if (!completion || !completion->wait().ok())
      ++failures;
  };

  std::thread first(worker, 1.0F);
  std::thread second(worker, 2.0F);
  first.join();
  second.join();
  EXPECT_EQ(failures.load(), 0);
}

TEST(Invocation, PlanAndRunStateRetainTheirContextOwnedArtifacts) {
  AllocatorProbe probe;
  Executable retained;
  {
    Context context =
        Context::create(Device::cpu(),
                        {.metadata_allocator = probe_allocator(probe)})
            .value();
    const LogicalGraph logical = make_rms_graph(row(8));
    retained = hyle::graph::compile(logical.graph, context)->executable;
  }
  EXPECT_TRUE(retained.valid());
  Invocation invocation = retained.create_invocation().value();
  retained = Executable{};
  EXPECT_TRUE(invocation.valid());
}

} // namespace
