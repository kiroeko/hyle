#include "hyle/graph/cuda.h"

#include "../compute/workload_fixture.h"
#include "../compute_models/d5_qwen35_fixture.h"
#include "../compute_models/d5_qwen38_gdn_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "../compute_route_cuda/ffn_support.h"
#include "hyle/compute/providers/ggml_cuda.h"

namespace {
using namespace cuda_route_test;
namespace graph = hyle::graph;

TEST(CudaDefaultGraph, BfloatGatesUseNormalDirectAndPreparedGraphEntries) {
  using Value = hyle::simd::bfloat16_t;
  auto owner = graph::cuda::create_context().value();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  for (const std::int64_t width : {129, 2048}) {
    const auto input_desc = matrix(compute::DataType::kBFloat16, 2, 2 * width);
    const auto output_desc = matrix(compute::DataType::kBFloat16, 2, width);
    auto input = mem::Buffer::create(resource.ref(),
                                     input_desc.required_bytes().value(), 256)
                     .value();
    auto direct_output =
        mem::Buffer::create(resource.ref(),
                            output_desc.required_bytes().value(), 256)
            .value();
    auto graph_output =
        mem::Buffer::create(resource.ref(),
                            output_desc.required_bytes().value(), 256)
            .value();
    const auto source =
        compute::ConstTensorView::create(input_desc, input.view()).value();
    const auto direct_view =
        compute::TensorView::create(output_desc, direct_output.view()).value();
    for (const auto kind : {compute::GatedActivationKind::kSilu,
                            compute::GatedActivationKind::kGelu,
                            compute::GatedActivationKind::kGeluTanh}) {
      const compute::GatedActivation operation{.activation = kind};
      auto direct =
          compute::prepare(owner, operation, input_desc, resource.ref());
      ASSERT_TRUE(direct.has_value()) << direct.error().message();
      const auto selected = width % 8 == 0
                                ? compute::cuda_native::kSglangProviderId
                            : kind == compute::GatedActivationKind::kSilu
                                ? compute::ggml_cuda::kProviderId
                                : compute::cuda_native::kProviderId;
      EXPECT_EQ(direct->provider().id, selected);
      auto direct_slot = direct->create_invocation().value();
      auto builder = graph::GraphBuilder::create().value();
      const auto input_id = builder.input(input_desc).value();
      const auto output_id = builder.add(operation, input_id).value();
      ASSERT_TRUE(builder.publish(output_id).ok());
      auto logical = std::move(builder).finish().value();
      auto compiled = graph::cuda::compile(logical, owner, resource.ref());
      ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
      EXPECT_EQ(compiled->report.step(0)->provider.id, selected);
      auto graph_slot = compiled->executable.create_invocation().value();
      ASSERT_TRUE(graph_slot.bind(input_id, source).ok());
      ASSERT_TRUE(graph_slot
                      .bind(output_id, compute::TensorView::create(
                                           output_desc, graph_output.view())
                                           .value())
                      .ok());
      ASSERT_TRUE(graph::cuda::prepare_replay(graph_slot, queue).ok());
      EXPECT_EQ(graph_slot.replay_route(),
                selected == compute::ggml_cuda::kProviderId
                    ? graph::ReplayRoute::kCommands
                    : graph::ReplayRoute::kCaptured);
      std::vector<Value> values(static_cast<std::size_t>(4 * width));
      for (int seed : {1, 3, 7}) {
        for (std::size_t i = 0; i < values.size(); ++i)
          values[i] = half_test::encode<Value>(
              (static_cast<int>(i % 23) - 11 + seed) / 16.0);
        ASSERT_TRUE(input.view().write(std::as_bytes(std::span(values))).ok());
        auto direct_done = direct_slot.submit(queue, source, direct_view);
        ASSERT_TRUE(direct_done.has_value());
        ASSERT_TRUE(direct_done->wait().ok());
        auto graph_done = graph_slot.submit(queue);
        ASSERT_TRUE(graph_done.has_value());
        ASSERT_TRUE(graph_done->wait().ok());
        const auto direct_values = half_test::read<Value>(direct_output);
        const auto graph_values = half_test::read<Value>(graph_output);
        for (std::size_t i = 0; i < direct_values.size(); ++i) {
          const auto column = i % static_cast<std::size_t>(width);
          const auto row = i / static_cast<std::size_t>(width);
          const auto base = row * static_cast<std::size_t>(2 * width) + column;
          const double gate = half_test::decode(values[base]);
          const double value =
              half_test::decode(values[base + static_cast<std::size_t>(width)]);
          double activated = gate / (1 + std::exp(-gate));
          if (kind == compute::GatedActivationKind::kGelu)
            activated = 0.5 * gate * (1 + std::erf(gate / std::sqrt(2.0)));
          else if (kind == compute::GatedActivationKind::kGeluTanh)
            activated = 0.5 * gate *
                        (1 + std::tanh(std::sqrt(2.0 / 3.14159265358979323846) *
                                       (gate + 0.044715 * gate * gate * gate)));
          half_test::expect_near(direct_values[i], activated * value);
          EXPECT_EQ(graph_values[i].bits, direct_values[i].bits);
        }
      }
    }
  }
}

TEST(CudaDefaultGraph, D5RealQwenProjectionUsesExactProfileAndReplays) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  d5_qwen35_fixture::Projection fixture;
  ASSERT_NO_THROW(fixture = d5_qwen35_fixture::load_projection(*fixture_root));

  const auto activation_desc =
      matrix(compute::DataType::kBFloat16, 1,
             d5_qwen35_fixture::kHidden);
  const auto stored_weight_desc =
      matrix(compute::DataType::kBFloat16, d5_qwen35_fixture::kHidden,
             d5_qwen35_fixture::kHidden);
  const auto weight_desc = stored_weight_desc.transposed().value();
  auto builder = graph::GraphBuilder::create().value();
  const auto activation_value = builder.input(activation_desc).value();
  const auto weight_value = builder.input(weight_desc).value();
  const auto output_value =
      builder.add(compute::MatMul{}, activation_value, weight_value).value();
  ASSERT_TRUE(builder.publish(output_value).ok());
  auto logical = std::move(builder).finish().value();

  auto owner = graph::cuda::create_context().value();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  const auto workload = d5_qwen35_fixture::workload(
      compute::WorkloadPhase::kDecode);
  auto compiled = graph::cuda::compile(logical, owner, resource.ref(),
                                       {.workload = workload});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->executable.workload(), workload);
  ASSERT_EQ(compiled->report.step_count(), 1U);
  EXPECT_EQ(compiled->report.step(0)->provider.id,
            compute::cublas::kProviderId);

  auto activation = mem::Buffer::create(resource.ref(), fixture.activation.size(),
                                        256)
                        .value();
  auto weight =
      mem::Buffer::create(resource.ref(), fixture.weight.size(), 256).value();
  const auto output_desc = logical.value_description(output_value).value();
  auto output = mem::Buffer::create(resource.ref(),
                                    output_desc.required_bytes().value(), 256)
                    .value();
  ASSERT_TRUE(weight.view().write(fixture.weight).ok());
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(
      invocation
          .bind(activation_value,
                compute::ConstTensorView::create(activation_desc,
                                                 activation.view())
                    .value())
          .ok());
  ASSERT_TRUE(invocation
                  .bind(weight_value,
                        compute::ConstTensorView::create(weight_desc,
                                                         weight.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(output_value,
                        compute::TensorView::create(output_desc, output.view())
                            .value())
                  .ok());
  auto queue = owner.create_queue().value();
  ASSERT_TRUE(graph::cuda::prepare_replay(invocation, queue).ok());
  for (int run = 0; run < 2; ++run) {
    auto values = fixture.activation;
    if (run != 0) {
      for (std::size_t index = 0; index < d5_qwen35_fixture::kHidden;
           ++index)
        values[index * 2 + 1] ^= std::byte{0x80};
    }
    ASSERT_TRUE(activation.view().write(values).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<hyle::simd::bfloat16_t>(output);
    ASSERT_EQ(actual.size(), fixture.expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index)
      half_test::expect_near(actual[index],
                             run == 0 ? fixture.expected[index]
                                      : -fixture.expected[index]);
  }
}

class CudaGraphQwen38GdnProfiles
    : public ::testing::TestWithParam<
          const compute::ModelArtifactProfileDescription *> {};

TEST_P(CudaGraphQwen38GdnProfiles, GroupedHeadUsesOrdinaryGraphAndReplays) {
  const d5_qwen38_gdn_fixture::Fixture fixture;
  auto builder = graph::GraphBuilder::create().value();
  std::array<graph::ValueId, 6> inputs;
  for (std::size_t port = 0; port < inputs.size(); ++port)
    inputs[port] = builder.input(fixture.inputs[port]).value();
  const auto node =
      builder.add_node(compute::Operation(compute::GatedDeltaNet{}), inputs)
          .value();
  const std::array outputs{builder.result(node, 0).value(),
                           builder.result(node, 1).value()};
  ASSERT_TRUE(builder.publish(outputs[0]).ok());
  ASSERT_TRUE(builder.publish(outputs[1]).ok());
  auto logical = std::move(builder).finish().value();

  auto owner = graph::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  const auto workload = d5_qwen38_gdn_fixture::workload(
      *GetParam(), compute::WorkloadPhase::kDecode);
  auto compiled = graph::cuda::compile(logical, owner, resource,
                                       {.workload = workload});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->executable.workload(), workload);
  ASSERT_EQ(compiled->report.step_count(), 1U);
  EXPECT_EQ(compiled->report.step(0)->provider.id,
            compute::ggml_cuda::kProviderId);

  std::array<mem::Buffer, 6> input_storage;
  std::array<mem::Buffer, 2> output_storage;
  auto invocation = compiled->executable.create_invocation().value();
  for (std::size_t port = 0; port < inputs.size(); ++port) {
    input_storage[port] =
        mem::Buffer::create(resource,
                            fixture.inputs[port].required_bytes().value(), 256)
            .value();
    ASSERT_TRUE(input_storage[port]
                    .view()
                    .write(std::as_bytes(std::span(fixture.data[port])))
                    .ok());
    ASSERT_TRUE(invocation
                    .bind(inputs[port],
                          compute::ConstTensorView::create(
                              fixture.inputs[port], input_storage[port].view())
                              .value())
                    .ok());
  }
  for (std::size_t port = 0; port < outputs.size(); ++port) {
    output_storage[port] =
        mem::Buffer::create(resource,
                            fixture.outputs[port].required_bytes().value(), 256)
            .value();
    ASSERT_TRUE(invocation
                    .bind(outputs[port],
                          compute::TensorView::create(
                              fixture.outputs[port], output_storage[port].view())
                              .value())
                    .ok());
  }
  auto queue = owner.create_queue().value();
  ASSERT_TRUE(graph::cuda::prepare_replay(invocation, queue).ok());
  const auto expected = fixture.oracle();
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    for (std::size_t port = 0; port < outputs.size(); ++port) {
      const auto actual = half_test::read<float>(output_storage[port]);
      ASSERT_EQ(actual.size(), expected[port].size());
      for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[port][i], 4.e-6) << port << ':' << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    D5OfficialAndPrimaryGguf, CudaGraphQwen38GdnProfiles,
    ::testing::Values(&d5_qwen38_gdn_fixture::kOfficialProfile,
                      &d5_qwen38_gdn_fixture::kPrimaryGgufProfile));

INSTANTIATE_TEST_SUITE_P(
    D5FidelityGguf, CudaGraphQwen38GdnProfiles,
    ::testing::Values(&d5_qwen38_gdn_fixture::kFidelityGgufProfile));

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

void run_graph(compute::DataType type, std::int64_t rows, bool commands) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        graph::cuda::create_context({.metadata_allocator = stats.get()})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    Linear first(resource.ref(), type, 128, 256, 1);
    Linear second(resource.ref(), type, 128, 128, 4);
    FfnGraph logical(matrix(type, rows, 128), first, second, stats.get());
    auto workload =
        workload_test::known(rows == 1 ? compute::WorkloadPhase::kDecode
                                       : compute::WorkloadPhase::kPrefill);
    auto compiled = graph::cuda::compile(
        logical.logical, owner, resource.ref(),
        {.constants = logical.constants, .workload = workload});
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    EXPECT_EQ(compiled->executable.workload(), workload);
    workload = {};
    EXPECT_TRUE(
        compiled->executable.workload().profile.matches(workload_test::kModel));
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
    // Captured native artifact and per-slot cuBLAS scratch must survive.
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

TEST(CudaDefaultGraph, DenseFfnCapturesTwoSlotsAndChangingInputs) {
  for (const auto rows : {1, 17}) {
    SCOPED_TRACE(rows);
    run_graph(compute::DataType::kFloat32, rows, false);
    run_graph(compute::DataType::kFloat16, rows, false);
  }
}

TEST(CudaDefaultGraph, ExplicitCommandsUseTheSameNumericalChain) {
  run_graph(compute::DataType::kFloat16, 7, true);
}

TEST(CudaDefaultGraph, DefaultRulesPrepareSeparateAndPackedTailCapability) {
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
              compute::ggml_cuda::kProviderId);
    auto slot = compiled->executable.create_invocation().value();
    ASSERT_TRUE(slot.bind(input, first.view.as_const()).ok());
    if (!packed) {
      ASSERT_TRUE(slot.bind(second_input, second.view.as_const()).ok());
    }
    ASSERT_TRUE(slot.bind(result, output.view).ok());
    ASSERT_TRUE(graph::cuda::prepare_replay(slot, queue).ok());
    EXPECT_EQ(slot.replay_route(), graph::ReplayRoute::kCommands);
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
      builder.input(matrix(compute::DataType::kFloat64, 1, 128)).value();
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
