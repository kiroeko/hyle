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
using hyle::compute::Queue;
using hyle::compute::RmsNorm;
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

struct LogicalGraph final {
  Graph graph{};
  std::array<ValueId, 2> inputs{};
  ValueId output{};
  std::size_t input_count = 0;
};

TensorDesc row(std::size_t size) {
  const std::array<std::int64_t, 1> shape{
      static_cast<std::int64_t>(size)};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

LogicalGraph make_graph(TensorDesc description, bool weighted = false) {
  GraphBuilder builder;
  LogicalGraph result;
  result.inputs[0] = builder.input(description).value();
  result.input_count = 1;
  if (weighted) {
    result.inputs[1] = builder.input(description).value();
    result.input_count = 2;
    result.output =
        builder.add(RmsNorm{}, result.inputs[0], result.inputs[1]).value();
  } else {
    result.output = builder.add(RmsNorm{}, result.inputs[0]).value();
  }
  EXPECT_TRUE(builder.output(result.output).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

void write(Buffer& buffer, std::span<const float> values) {
  ASSERT_TRUE(buffer.view().write(std::as_bytes(values)).ok());
}

std::vector<float> read(const Buffer& buffer, std::size_t count) {
  std::vector<float> result(count);
  EXPECT_TRUE(buffer.view()
                  .read(std::as_writable_bytes(std::span(result)))
                  .ok());
  return result;
}

void expect_rms(std::span<const float> input,
                std::span<const float> output) {
  float sum = 0.0F;
  for (float value : input) sum += value * value;
  const float scale = 1.0F /
                      std::sqrt(sum / static_cast<float>(input.size()) +
                                RmsNorm{}.epsilon);
  for (std::size_t index = 0; index < input.size(); ++index) {
    const float expected = input[index] * scale;
    EXPECT_NEAR(output[index], expected,
                std::abs(expected) * 3.0e-5F + 3.0e-6F);
  }
}

struct AllocatorProbe final {
  AllocatorRef backing = hyle::mem::system_allocator();
  std::size_t allocations = 0;
  std::size_t deallocations = 0;
  bool fail = false;
};

void* probe_allocate(void* opaque, std::size_t size,
                     std::size_t alignment) noexcept {
  auto& probe = *static_cast<AllocatorProbe*>(opaque);
  if (probe.fail) return nullptr;
  void* result = probe.backing.allocate(size, alignment);
  if (result != nullptr) ++probe.allocations;
  return result;
}

void probe_free(void* opaque, void* pointer, std::size_t size,
                std::size_t alignment) noexcept {
  auto& probe = *static_cast<AllocatorProbe*>(opaque);
  if (pointer != nullptr) ++probe.deallocations;
  probe.backing.deallocate(pointer, size, alignment);
}

void* probe_reallocate(void* opaque, void* pointer, std::size_t old_size,
                       std::size_t new_size,
                       std::size_t alignment) noexcept {
  auto& probe = *static_cast<AllocatorProbe*>(opaque);
  if (probe.fail) return nullptr;
  return probe.backing.reallocate(pointer, old_size, new_size, alignment);
}

AllocatorRef probe_allocator(AllocatorProbe& probe) noexcept {
  return AllocatorRef(&probe_allocate, &probe_free, &probe_reallocate, &probe);
}

TEST(GraphCompile, EmitsHonestCpuPlanAndShareableExecutable) {
  const LogicalGraph logical = make_graph(row(8));
  Context context = Context::create(Device::cpu()).value();
  CompiledGraph compiled = hyle::graph::compile(logical.graph, context).value();

  EXPECT_TRUE(compiled.executable.valid());
  EXPECT_EQ(compiled.executable.input_count(), 1u);
  EXPECT_EQ(compiled.executable.input_id().value(), logical.inputs[0]);
  EXPECT_EQ(compiled.executable.output_id().value(), logical.output);
  EXPECT_EQ(compiled.report.backend, hyle::compute::BackendKind::kCpu);
  EXPECT_EQ(compiled.report.provider,
            hyle::compute::ProviderKind::kHyleNative);
  EXPECT_EQ(compiled.report.workspace_bytes, 0u);
  EXPECT_EQ(compiled.report.copy_bytes, 0u);
  EXPECT_EQ(compiled.report.synchronization_count, 0u);
  EXPECT_FALSE(compiled.report.used_fallback);

  Executable copied = compiled.executable;
  Executable moved = std::move(copied);
  EXPECT_FALSE(copied.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(moved.create_invocation().has_value());
}

TEST(GraphCompile, ReportsInvalidClosedAndAllocationFailures) {
  Context context = Context::create(Device::cpu()).value();
  auto invalid = hyle::graph::compile(Graph{}, context);
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsInvalidArgument());

  const LogicalGraph logical = make_graph(row(8));
  ASSERT_TRUE(context.close().ok());
  auto closed = hyle::graph::compile(logical.graph, context);
  ASSERT_FALSE(closed.has_value());
  EXPECT_TRUE(closed.error().IsFailedPrecondition());

  AllocatorProbe probe;
  Context oom_context = Context::create(
      Device::cpu(), {.metadata_allocator = probe_allocator(probe)}).value();
  EXPECT_EQ(probe.allocations, 1u);
  probe.fail = true;
  auto oom = hyle::graph::compile(logical.graph, oom_context);
  ASSERT_FALSE(oom.has_value());
  EXPECT_TRUE(oom.error().IsResourceExhausted());
  EXPECT_EQ(probe.allocations, 1u);
}

TEST(Executable, AllocatorOwnedPlanOutlivesItsExecutableCopies) {
  AllocatorProbe probe;
  Executable retained;
  {
    Context context = Context::create(
        Device::cpu(), {.metadata_allocator = probe_allocator(probe)}).value();
    const LogicalGraph logical = make_graph(row(0));
    CompiledGraph compiled = hyle::graph::compile(logical.graph, context).value();
    EXPECT_EQ(probe.allocations, 2u);
    retained = compiled.executable;
  }
  EXPECT_EQ(probe.deallocations, 0u);
  retained = Executable{};
  EXPECT_EQ(probe.deallocations, 2u);
}

TEST(Invocation, BindsRunsResetsAndReusesWithoutOwningPayload) {
  const std::vector<float> input{1, -2, 3, -4, 5, -6, 7, -8};
  const std::size_t bytes = input.size() * sizeof(float);
  const TensorDesc description = row(input.size());
  const LogicalGraph logical = make_graph(description);
  Context context = Context::create(Device::cpu()).value();
  Queue queue = context.create_queue().value();
  Executable executable =
      hyle::graph::compile(logical.graph, context).value().executable;
  Invocation invocation = executable.create_invocation().value();

  Buffer input_buffer =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float)).value();
  Buffer output_buffer =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float)).value();
  write(input_buffer, input);
  TensorView input_view =
      TensorView::create(description, input_buffer.view()).value();
  TensorView output_view =
      TensorView::create(description, output_buffer.view()).value();

  EXPECT_TRUE(invocation.bind(logical.inputs[0], input_view).ok());
  EXPECT_TRUE(invocation.bind(logical.output, output_view).ok());
  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value());
  ASSERT_TRUE(completion->wait().ok());
  expect_rms(input, read(output_buffer, input.size()));

  invocation.reset();
  auto incomplete = invocation.submit(queue);
  ASSERT_FALSE(incomplete.has_value());
  EXPECT_TRUE(incomplete.error().IsFailedPrecondition());
  EXPECT_TRUE(invocation.bind(logical.inputs[0], input_view).ok());
  EXPECT_TRUE(invocation.bind(logical.output, output_view).ok());
  EXPECT_TRUE(invocation.submit(queue)->wait().ok());
}

TEST(Invocation, RejectsWrongGraphShapeMutabilityAndIncompleteBindings) {
  const TensorDesc description = row(4);
  const LogicalGraph first = make_graph(description);
  const LogicalGraph other = make_graph(description);
  Context context = Context::create(Device::cpu()).value();
  Executable executable =
      hyle::graph::compile(first.graph, context).value().executable;
  Invocation invocation = executable.create_invocation().value();
  Buffer storage =
      Buffer::create(hyle::mem::system_resource(), 4 * sizeof(float)).value();
  TensorView tensor = TensorView::create(description, storage.view()).value();

  EXPECT_TRUE(invocation.bind(other.inputs[0], tensor).IsInvalidArgument());
  EXPECT_TRUE(invocation.bind(first.output, tensor.as_const()).IsInvalidArgument());
  Queue queue = context.create_queue().value();
  auto incomplete = invocation.submit(queue);
  ASSERT_FALSE(incomplete.has_value());
  EXPECT_TRUE(incomplete.error().IsFailedPrecondition());

  const TensorDesc wrong_desc = row(3);
  Buffer wrong_storage =
      Buffer::create(hyle::mem::system_resource(), 3 * sizeof(float)).value();
  TensorView wrong =
      TensorView::create(wrong_desc, wrong_storage.view()).value();
  EXPECT_TRUE(invocation.bind(first.inputs[0], wrong).IsInvalidArgument());
  EXPECT_TRUE(invocation.bind(first.output, wrong).IsInvalidArgument());
}

TEST(Invocation, WeightedGraphUsesBothIndependentInputs) {
  const std::vector<float> input{1, 2, 3, 4};
  const std::vector<float> weight{2, 2, 2, 2};
  const std::size_t bytes = input.size() * sizeof(float);
  const TensorDesc description = row(input.size());
  const LogicalGraph logical = make_graph(description, true);
  Context context = Context::create(Device::cpu()).value();
  Queue queue = context.create_queue().value();
  Invocation invocation = hyle::graph::compile(logical.graph, context)
                              ->executable.create_invocation()
                              .value();
  Buffer input_buffer = Buffer::create(hyle::mem::system_resource(), bytes).value();
  Buffer weight_buffer = Buffer::create(hyle::mem::system_resource(), bytes).value();
  Buffer output_buffer = Buffer::create(hyle::mem::system_resource(), bytes).value();
  write(input_buffer, input);
  write(weight_buffer, weight);
  auto input_view =
      TensorView::create(description, input_buffer.view()).value();
  auto weight_view =
      TensorView::create(description, weight_buffer.view()).value();
  auto output_view =
      TensorView::create(description, output_buffer.view()).value();
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input_view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], weight_view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output_view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  auto result = read(output_buffer, input.size());
  std::vector<float> unweighted(result.size());
  for (std::size_t index = 0; index < result.size(); ++index) {
    unweighted[index] = result[index] / weight[index];
  }
  expect_rms(input, unweighted);
}

TEST(Invocation, DistinctInvocationsRunConcurrentlyFromOneExecutable) {
  constexpr std::size_t count = 256;
  const TensorDesc description = row(count);
  const LogicalGraph logical = make_graph(description);
  Context context = Context::create(Device::cpu()).value();
  Executable executable =
      hyle::graph::compile(logical.graph, context).value().executable;
  std::atomic<int> failures{0};

  auto worker = [&](float seed) {
    Queue queue = context.create_queue().value();
    Invocation invocation = executable.create_invocation().value();
    std::vector<float> input(count, seed);
    Buffer input_buffer =
        Buffer::create(hyle::mem::system_resource(), count * sizeof(float)).value();
    Buffer output_buffer =
        Buffer::create(hyle::mem::system_resource(), count * sizeof(float)).value();
    if (!input_buffer.view().write(std::as_bytes(std::span(input))).ok()) {
      ++failures;
      return;
    }
    auto input_view = TensorView::create(description, input_buffer.view());
    auto output_view = TensorView::create(description, output_buffer.view());
    if (!input_view || !output_view ||
        !invocation.bind(logical.inputs[0], *input_view).ok() ||
        !invocation.bind(logical.output, *output_view).ok()) {
      ++failures;
      return;
    }
    auto completion = invocation.submit(queue);
    if (!completion || !completion->wait().ok()) ++failures;
  };

  std::thread first(worker, 1.0F);
  std::thread second(worker, 2.0F);
  first.join();
  second.join();
  EXPECT_EQ(failures.load(), 0);
}

}  // namespace
