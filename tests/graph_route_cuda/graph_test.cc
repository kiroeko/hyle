#include "hyle/graph/cuda.h"

#include "../compute_route_cuda/ffn_support.h"

namespace {
using namespace cuda_route_test;
namespace graph = hyle::graph;

struct FfnGraph final {
  graph::Graph logical;
  graph::ValueId input, output;
  std::vector<graph::ConstantBinding> constants;

  FfnGraph(compute::TensorDesc description, const Linear &first,
           const Linear &second, mem::AllocatorRef allocator) {
    auto builder =
        graph::GraphBuilder::create({.metadata_allocator = allocator}).value();
    input = builder.input(description).value();
    auto linear = [&](const Linear &weights, graph::ValueId activation) {
      const auto w = builder.constant(weights.weights.description()).value();
      constants.push_back({w, weights.weights});
      if (weights.quantized) {
        const auto s = builder.constant(weights.scales.description()).value();
        constants.push_back({s, weights.scales});
        return builder
            .add(cuda::QuantizedWeights::operation(), activation, w, s)
            .value();
      }
      return builder.add(compute::MatMul{}, activation, w).value();
    };
    const auto gate_up = linear(first, input);
    const auto activated =
        builder.add(compute::GatedActivation{}, gate_up).value();
    output = linear(second, activated);
    EXPECT_TRUE(builder.publish(output).ok());
    logical = std::move(builder).finish().value();
  }
};

void run_graph(compute::DataType type, std::int64_t rows, bool first_q4,
               bool second_q4, bool commands) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        graph::cuda::create_context({.metadata_allocator = stats.get()})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    Linear first(owner, resource.ref(), type, 128, 256, first_q4, 1);
    Linear second(owner, resource.ref(), type, 128, 128, second_q4, 4);
    FfnGraph logical(matrix(type, rows, 128), first, second, stats.get());
    auto compiled = graph::cuda::compile(logical.logical, owner, resource.ref(),
                                         {.constants = logical.constants});
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    EXPECT_EQ(compiled->report.step_count(), 3U);
    EXPECT_EQ(compiled->report.target_count(), 1U);
    EXPECT_EQ(compiled->report.lane_count(), 1U);
    EXPECT_EQ(compiled->report.copy_bytes(), 0U);
    EXPECT_TRUE(compiled->report.mixed_providers());
    EXPECT_TRUE(compiled->report.supports_concurrent_invocations());
    EXPECT_EQ(compiled->report.borrowed_constant_count(),
              logical.constants.size());
    EXPECT_EQ(compiled->report.step(0)->provider.id, first.provider());
    EXPECT_EQ(compiled->report.step(1)->provider.id,
              compute::cuda_native::kSglangProviderId);
    EXPECT_EQ(compiled->report.step(2)->provider.id, second.provider());
    std::array inputs{Tensor(matrix(type, rows, 128), resource.ref()),
                      Tensor(matrix(type, rows, 128), resource.ref())};
    std::array outputs{Tensor(matrix(type, rows, 128), resource.ref()),
                       Tensor(matrix(type, rows, 128), resource.ref())};
    std::array queues{owner.create_queue().value(),
                      owner.create_queue().value()};
    std::array slots{compiled->executable.create_invocation().value(),
                     compiled->executable.create_invocation().value()};
    for (std::size_t lane = 0; lane < slots.size(); ++lane) {
      ASSERT_TRUE(
          slots[lane].bind(logical.input, inputs[lane].view.as_const()).ok());
      ASSERT_TRUE(slots[lane].bind(logical.output, outputs[lane].view).ok());
      hyle::Status prepared;
      if (commands) {
        prepared = graph::cuda::prepare_replay(
            slots[lane], queues[lane],
            {.mode = graph::ReplayMode::kCommands,
             .host_wait = graph::HostWaitPolicy::kReject});
      } else if (lane == 0) {
        // The normal defaults must actually choose native capture here.
        prepared = graph::cuda::prepare_replay(slots[lane], queues[lane]);
      } else {
        prepared = graph::cuda::prepare_replay(
            slots[lane], queues[lane],
            {.mode = graph::ReplayMode::kRequireCapture,
             .host_wait = graph::HostWaitPolicy::kReject});
      }
      ASSERT_TRUE(prepared.ok()) << prepared.message();
      EXPECT_EQ(slots[lane].replay_route(),
                commands ? graph::ReplayRoute::kCommands
                         : graph::ReplayRoute::kCaptured);
      EXPECT_EQ(slots[lane].replay_report()->host_wait_dependency_count, 0U);
    }
    const auto before = stats.snapshot();
    for (int seed = 0; seed < 3; ++seed) {
      std::array<std::vector<float>, 2> data;
      std::array<compute::Completion, 2> completed;
      for (std::size_t lane = 0; lane < slots.size(); ++lane) {
        data[lane] = activations(type, static_cast<std::size_t>(rows) * 128,
                                 seed + static_cast<int>(lane) * 3);
        inputs[lane].write(data[lane]);
      }
      for (std::size_t lane = 0; lane < slots.size(); ++lane) {
        auto done = slots[lane].submit(queues[lane]);
        ASSERT_TRUE(done.has_value()) << done.error().message();
        completed[lane] = std::move(*done);
      }
      for (std::size_t lane = 0; lane < slots.size(); ++lane) {
        ASSERT_TRUE(completed[lane].wait().ok());
        ASSERT_TRUE(slots[lane].output(logical.output).has_value());
        check_ffn(outputs[lane], data[lane], first, second);
      }
    }
    EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
    EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);

    // New addresses require a cold prepare; changing contents above did not.
    ASSERT_TRUE(slots[0].reset().ok());
    EXPECT_EQ(slots[0].replay_route(), graph::ReplayRoute::kUnprepared);
    ASSERT_TRUE(slots[0].bind(logical.input, inputs[1].view.as_const()).ok());
    ASSERT_TRUE(slots[0].bind(logical.output, outputs[0].view).ok());
    ASSERT_TRUE(graph::cuda::prepare_replay(slots[0], queues[0]).ok());
    // Destroy the unrelated idle slot before blocking device progress. Its
    // ordinary cuMemFree teardown may synchronize with other device work.
    slots[1] = {};
    ASSERT_TRUE(queues[0].reserve_completions(2).ok());
    HostGate gate(queues[0]);
    auto final = slots[0].submit(queues[0]);
    ASSERT_TRUE(final.has_value()) << final.error().message();
    EXPECT_TRUE(slots[0].in_flight());
    // Captured native artifact and per-slot cuBLAS/Marlin scratch must survive.
    slots[0] = {};
    compiled = graph::CompiledGraph{};
    ASSERT_TRUE(owner.close().ok());
    gate.open();
    ASSERT_TRUE(final->wait().ok());
    check_ffn(outputs[0],
              activations(type, static_cast<std::size_t>(rows) * 128, 5), first,
              second);
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(CudaDefaultGraph, DenseAndQ4FfnCaptureTwoSlotsAndChangingInputs) {
  for (const auto rows : {1, 17}) {
    SCOPED_TRACE(rows);
    run_graph(compute::DataType::kFloat32, rows, false, false, false);
    run_graph(compute::DataType::kFloat16, rows, false, false, false);
    run_graph(compute::DataType::kFloat16, rows, true, false, false);
    run_graph(compute::DataType::kFloat16, rows, true, true, false);
  }
}

TEST(CudaDefaultGraph, ExplicitCommandsUseTheSameNumericalChain) {
  run_graph(compute::DataType::kFloat16, 7, true, false, true);
}

TEST(CudaDefaultGraph, DefaultRulesCaptureSeparateAndPackedTailCapability) {
  auto owner = graph::cuda::create_context().value();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  for (const bool packed : {false, true}) {
    SCOPED_TRACE(packed);
    const std::int64_t hidden = packed ? 127 : 128;
    Tensor first(
        matrix(compute::DataType::kFloat32, 1, hidden * (packed ? 2 : 1)),
        resource.ref());
    Tensor second(matrix(compute::DataType::kFloat32, 1, hidden),
                  resource.ref());
    Tensor output(matrix(compute::DataType::kFloat32, 1, hidden),
                  resource.ref());
    auto builder = graph::GraphBuilder::create().value();
    const auto input = builder.input(first.view.description()).value();
    const auto second_input =
        packed ? input : builder.input(second.view.description()).value();
    const compute::GatedActivation operation{
        .inputs = packed ? compute::GatedInputKind::kPackedHalves
                         : compute::GatedInputKind::kSeparate};
    const auto result =
        packed ? builder.add(operation, input).value()
               : builder.add(operation, input, second_input).value();
    ASSERT_TRUE(builder.publish(result).ok());
    auto logical = std::move(builder).finish().value();
    auto compiled = graph::cuda::compile(logical, owner, resource.ref());
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    ASSERT_EQ(compiled->report.step_count(), 1U);
    EXPECT_EQ(compiled->report.step(0)->provider.id,
              compute::cuda_native::kProviderId);
    auto slot = compiled->executable.create_invocation().value();
    ASSERT_TRUE(slot.bind(input, first.view.as_const()).ok());
    if (!packed) {
      ASSERT_TRUE(slot.bind(second_input, second.view.as_const()).ok());
    }
    ASSERT_TRUE(slot.bind(result, output.view).ok());
    ASSERT_TRUE(graph::cuda::prepare_replay(slot, queue).ok());
    EXPECT_EQ(slot.replay_route(), graph::ReplayRoute::kCaptured);
    for (const float value : {1.0F, 2.0F}) {
      first.write(std::vector<float>(
          static_cast<std::size_t>(hidden) * (packed ? 2 : 1), value));
      second.write(std::vector<float>(static_cast<std::size_t>(hidden), value));
      auto done = slot.submit(queue);
      ASSERT_TRUE(done.has_value()) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      for (const auto actual : output.read())
        EXPECT_NEAR(actual, value * value / (1.0 + std::exp(-value)), 0.000001);
    }
  }
}

TEST(CudaDefaultGraph, RejectsMissingResourcesAndUnsupportedOperations) {
  auto owner = graph::cuda::create_context().value();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto builder = graph::GraphBuilder::create().value();
  const auto input =
      builder.input(matrix(compute::DataType::kFloat32, 1, 128)).value();
  const auto output =
      builder.add(compute::Unary{compute::UnaryKind::kNeg}, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto logical = std::move(builder).finish().value();
  EXPECT_FALSE(graph::cuda::compile(logical, compute::Context{}, resource.ref())
                   .has_value());
  auto missing = graph::cuda::compile(logical, owner, {});
  ASSERT_FALSE(missing.has_value());
  EXPECT_TRUE(missing.error().IsInvalidArgument());
  auto unsupported = graph::cuda::compile(logical, owner, resource.ref());
  ASSERT_FALSE(unsupported.has_value());
  EXPECT_TRUE(unsupported.error().IsUnimplemented());
  graph::Invocation slot;
  compute::Queue queue;
  EXPECT_TRUE(graph::cuda::prepare_replay(slot, queue).IsInvalidArgument());
}

} // namespace
