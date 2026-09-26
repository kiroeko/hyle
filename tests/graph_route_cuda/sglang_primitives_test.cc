#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/mem/stats.h"

namespace {
namespace compute = hyle::compute;
namespace graph = hyle::graph;
namespace cuda = compute::cuda;
namespace mem = hyle::mem;
using Value = hyle::simd::bfloat16_t;

struct Tensor {
  mem::Buffer buffer;
  compute::TensorView view;
  Tensor(compute::TensorDesc description, mem::MemoryResourceRef resource)
      : buffer(mem::Buffer::create(resource,
                                   description.required_bytes().value(), 256)
                   .value()),
        view(compute::TensorView::create(description, buffer.view()).value()) {}
  template <class T> void write(const std::vector<T> &values) {
    ASSERT_TRUE(buffer.view().write(std::as_bytes(std::span(values))).ok());
  }
};

TEST(CudaDefaultGraph, HashRouterCaptureChecksIndicesAndKeepsInvocationErrorsIndependent) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner = cuda::create_context({.metadata_allocator = stats.get()}).value();
    auto memory = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    const auto desc = [](compute::DataType type,
                           std::initializer_list<std::int64_t> shape) {
      return compute::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
    };
    const auto logits_desc = desc(compute::DataType::kFloat32, {5, 256});
    const auto token_desc = desc(compute::DataType::kInt64, {5});
    const auto table_desc = desc(compute::DataType::kInt32, {129280, 6});
    const auto weights_desc = desc(compute::DataType::kFloat32, {5, 6});
    const auto ids_desc = desc(compute::DataType::kInt32, {5, 6});
    auto builder = graph::GraphBuilder::create().value();
    const std::array ports{builder.input(logits_desc).value(),
                            builder.input(token_desc).value(),
                            builder.input(table_desc).value()};
    const auto operation = compute::model_ops::sqrt_softplus_hash_routing();
    const auto node = builder.add_node(operation, ports).value();
    const std::array results{builder.result(node, 0).value(), builder.result(node, 1).value()};
    for (const auto id : results) ASSERT_TRUE(builder.publish(id).ok());
    auto logical = std::move(builder).finish().value();
    auto compiled = graph::cuda::compile(logical, owner, memory.ref(),
        {.provider_replay = graph::ProviderReplayPolicy::kRequirePrepared}).value();
    EXPECT_EQ(compiled.report.step(0)->provider.id, compute::cuda_native::kSglangProviderId);
    std::array logits{Tensor(logits_desc, memory.ref()), Tensor(logits_desc, memory.ref())};
    std::array tokens{Tensor(token_desc, memory.ref()), Tensor(token_desc, memory.ref())};
    std::array tables{Tensor(table_desc, memory.ref()), Tensor(table_desc, memory.ref())};
    std::array weights{Tensor(weights_desc, memory.ref()), Tensor(weights_desc, memory.ref())};
    std::array ids{Tensor(ids_desc, memory.ref()), Tensor(ids_desc, memory.ref())};
    std::array lanes{compiled.executable.create_invocation().value(),
                      compiled.executable.create_invocation().value()};
    std::array queues{owner.create_queue().value(), owner.create_queue().value()};
    const auto bind_and_capture = [&](std::size_t lane) {
      ASSERT_TRUE(lanes[lane].bind(ports[0], logits[lane].view.as_const()).ok());
      ASSERT_TRUE(lanes[lane].bind(ports[1], tokens[lane].view.as_const()).ok());
      ASSERT_TRUE(lanes[lane].bind(ports[2], tables[lane].view.as_const()).ok());
      ASSERT_TRUE(lanes[lane].bind(results[0], weights[lane].view).ok());
      ASSERT_TRUE(lanes[lane].bind(results[1], ids[lane].view).ok());
      logits[lane].write(std::vector<float>(5 * 256, 1.0F));
      ASSERT_TRUE(graph::cuda::prepare_replay(lanes[lane], queues[lane],
          {.mode = graph::ReplayMode::kRequireCapture,
           .host_wait = graph::HostWaitPolicy::kReject}).ok());
      EXPECT_EQ(lanes[lane].replay_route(), graph::ReplayRoute::kCaptured);
    };
    for (std::size_t lane = 0; lane < 2; ++lane) bind_and_capture(lane);
    const std::array direct_inputs{logits[0].view.as_const(), tokens[0].view.as_const(),
                                    tables[0].view.as_const()};
    const std::array direct_outputs{weights[0].view, ids[0].view};
    auto prepared = compute::prepare(owner, operation, direct_inputs, direct_outputs).value();
    EXPECT_EQ(prepared.provider().id, compute::cuda_native::kSglangProviderId);
    EXPECT_EQ(prepared.workspace_requirement().size, 0u);
    EXPECT_EQ(prepared.invocation_private_bytes(), 0u);
    auto direct = prepared.create_invocation().value();
    // A failed Graph slot is retired. Recreate only that Invocation while
    // retaining the Executable, queues and independently replaying second lane.
    for (int round = 0; round < 12; ++round) {
      SCOPED_TRACE(round);
      const bool invalid = round % 2 == 0;
      std::array<std::vector<std::int64_t>, 2> token_values;
      std::array<std::vector<std::int32_t>, 2> table_values;
      for (std::size_t lane = 0; lane < 2; ++lane) {
        token_values[lane] = {0, 17, 129279, 3, 8};
        auto &lookup = table_values[lane];
        lookup.resize(129280 * 6);
        for (std::size_t i = 0; i < lookup.size(); ++i)
          lookup[i] = std::int32_t((i * 7 + lane + round) % 256);
        if (invalid && lane == 0) {
          const int failure = round / 2;
          if (failure < 4) {
            const std::array<std::int64_t, 4> bad{-1, 129280,
                std::numeric_limits<std::int64_t>::min(),
                std::numeric_limits<std::int64_t>::max()};
            token_values[lane][4] = bad[failure];
          } else {
            lookup[129279 * 6 + 5] = failure == 4 ? -1 : 256;
          }
        }
        tokens[lane].write(token_values[lane]);
        tables[lane].write(lookup);
      }
      auto first = lanes[0].submit(queues[0]);
      auto second = lanes[1].submit(queues[1]);
      ASSERT_TRUE(first) << first.error().message();
      ASSERT_TRUE(second) << second.error().message();
      auto status = first->wait();
      EXPECT_EQ(status.IsOutOfRange(), invalid) << status.message();
      EXPECT_EQ(status.ok(), !invalid) << status.message();
      ASSERT_TRUE(second->wait().ok());
      const auto check = [&](std::size_t lane) {
        const auto actual_ids = half_test::read<std::int32_t>(ids[lane].buffer);
        const auto actual_weights = half_test::read<float>(weights[lane].buffer);
        for (std::size_t row = 0; row < 5; ++row)
          for (std::size_t slot = 0; slot < 6; ++slot) {
            EXPECT_EQ(actual_ids[row * 6 + slot],
                       table_values[lane][token_values[lane][row] * 6 + slot]);
            EXPECT_NEAR(actual_weights[row * 6 + slot], 1.0F / 6, 1.0e-6F);
          }
      };
      check(1);
      if (!invalid) check(0);
      if (invalid) {
        auto retired = lanes[0].submit(queues[0]);
        ASSERT_FALSE(retired);
        EXPECT_TRUE(retired.error().IsFailedPrecondition());
        lanes[0] = compiled.executable.create_invocation().value();
        bind_and_capture(0);
      }
      auto direct_done = direct.submit(queues[0], direct_inputs, direct_outputs);
      ASSERT_TRUE(direct_done) << direct_done.error().message();
      status = direct_done->wait();
      EXPECT_EQ(status.IsOutOfRange(), invalid) << status.message();
      EXPECT_EQ(status.ok(), !invalid) << status.message();
      if (!invalid) check(0);
      if (invalid) {
        auto retired = direct.submit(queues[0], direct_inputs, direct_outputs);
        ASSERT_FALSE(retired);
        EXPECT_TRUE(retired.error().IsFailedPrecondition());
        direct = prepared.create_invocation().value();
      }
    }
    ASSERT_TRUE(owner.close().ok());
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(CudaDefaultGraph,
     PreparedNormUnaryCaptureAndDefaultDirectKeepTwoIndependentStates) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        cuda::create_context({.metadata_allocator = stats.get()}).value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    const auto desc =
        compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                   std::array<std::int64_t, 3>{2, 17, 2048})
            .value();
    const auto wdesc =
        compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                   std::array<std::int64_t, 1>{2048})
            .value();
    Tensor weight(wdesc, resource.ref());
    std::vector<Value> scale(2048);
    for (std::size_t i = 0; i < scale.size(); ++i)
      scale[i] =
          half_test::encode<Value>(1.0 + (static_cast<int>(i % 7) - 3) * 0.125);
    weight.write(scale);
    std::array inputs{Tensor(desc, resource.ref()),
                      Tensor(desc, resource.ref())};
    std::array normals{Tensor(desc, resource.ref()),
                       Tensor(desc, resource.ref())};
    std::array outputs{Tensor(desc, resource.ref()),
                       Tensor(desc, resource.ref())};
    auto builder = graph::GraphBuilder::create().value();
    const auto input_id = builder.input(desc).value();
    const auto weight_id = builder.input(wdesc).value();
    const auto normal_id =
        builder.add(compute::RmsNorm{}, input_id, weight_id).value();
    const auto unary_id =
        builder.add(compute::Unary{compute::UnaryKind::kSilu}, normal_id)
            .value();
    const auto result_id =
        builder.add(compute::Binary{}, unary_id, input_id).value();
    ASSERT_TRUE(builder.publish(normal_id).ok());
    ASSERT_TRUE(builder.publish(result_id).ok());
    auto logical = std::move(builder).finish().value();
    auto compiled =
        graph::cuda::compile(
            logical, owner, resource.ref(),
            {.provider_replay = graph::ProviderReplayPolicy::kRequirePrepared})
            .value();
    EXPECT_EQ(compiled.report.step(0)->provider.id,
              compute::cuda_native::kSglangProviderId);
    EXPECT_EQ(compiled.report.step(1)->provider.id,
              compute::cuda_native::kSglangProviderId);
    std::array lanes{compiled.executable.create_invocation().value(),
                     compiled.executable.create_invocation().value()};
    std::array queues{owner.create_queue().value(),
                      owner.create_queue().value()};
    Tensor direct(desc, resource.ref());
    auto direct_program =
        compute::prepare(
            owner, compute::Operation(compute::RmsNorm{}),
            std::array{inputs[0].view.as_const(), weight.view.as_const()},
            std::array{direct.view})
            .value();
    EXPECT_EQ(direct_program.provider().id,
              compute::cuda_native::kSglangProviderId);
    EXPECT_TRUE(direct_program.implementation().valid());
    EXPECT_EQ(direct_program.route_preference(), 100U);
    auto direct_slot = direct_program.create_invocation().value();
    for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
      ASSERT_TRUE(
          lanes[lane].bind(input_id, inputs[lane].view.as_const()).ok());
      ASSERT_TRUE(lanes[lane].bind(weight_id, weight.view.as_const()).ok());
      ASSERT_TRUE(lanes[lane].bind(normal_id, normals[lane].view).ok());
      ASSERT_TRUE(lanes[lane].bind(result_id, outputs[lane].view).ok());
      ASSERT_TRUE(graph::cuda::prepare_replay(
                      lanes[lane], queues[lane],
                      {.mode = graph::ReplayMode::kRequireCapture,
                       .host_wait = graph::HostWaitPolicy::kReject})
                      .ok());
      EXPECT_EQ(lanes[lane].replay_route(), graph::ReplayRoute::kCaptured);
      ASSERT_TRUE(queues[lane].reserve_completions(2).ok());
    }
    const auto before = stats.snapshot();
    std::vector<Value> retained;
    for (int repeat = 0; repeat < 3; ++repeat) {
      for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
        if (repeat == 2 && lane == 0)
          continue;
        std::vector<Value> values(2 * 17 * 2048);
        for (std::size_t i = 0; i < values.size(); ++i)
          values[i] =
              half_test::encode<Value>((static_cast<int>(i % 41) - 20 + repeat +
                                        static_cast<int>(lane)) *
                                       0.125);
        inputs[lane].write(values);
        auto done = lanes[lane].submit(queues[lane]);
        ASSERT_TRUE(done.has_value()) << done.error().message();
        ASSERT_TRUE(done->wait().ok());
        const auto normal = half_test::read<Value>(normals[lane].buffer);
        const auto actual = half_test::read<Value>(outputs[lane].buffer);
        for (std::size_t row = 0; row < 34; ++row) {
          double sum = 0;
          for (std::size_t col = 0; col < scale.size(); ++col) {
            const double value = half_test::decode(values[row * 2048 + col]);
            sum += value * value;
          }
          const double inverse = 1 / std::sqrt(sum / 2048 + 1.0e-5F);
          for (std::size_t col = 0; col < scale.size(); ++col) {
            const auto index = row * 2048 + col;
            const double x = half_test::decode(values[index]);
            const double expected_norm =
                x * inverse * half_test::decode(scale[col]);
            half_test::expect_near(normal[index], expected_norm);
            const double n =
                half_test::decode(half_test::encode<Value>(expected_norm));
            const double activated = half_test::decode(
                half_test::encode<Value>(n / (1 + std::exp(-n))));
            half_test::expect_near(actual[index], activated + x);
          }
        }
        if (lane == 0) {
          auto done_direct = direct_slot
                                 .submit(queues[0],
                                         std::array{inputs[0].view.as_const(),
                                                    weight.view.as_const()},
                                         std::array{direct.view})
                                 .value();
          ASSERT_TRUE(done_direct.wait().ok());
          const auto direct_values = half_test::read<Value>(direct.buffer);
          for (std::size_t i = 0; i < normal.size(); ++i)
            half_test::expect_near(direct_values[i],
                                   half_test::decode(normal[i]));
          retained = actual;
        }
      }
    }
    EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
    EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
    ASSERT_TRUE(owner.close().ok());
    const auto after_close = half_test::read<Value>(outputs[0].buffer);
    for (std::size_t i = 0; i < retained.size(); ++i)
      EXPECT_EQ(after_close[i].bits, retained[i].bits);
  }
  EXPECT_TRUE(stats.balanced());
}
} // namespace
