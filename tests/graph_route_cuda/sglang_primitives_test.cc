#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <span>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
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
  void write(const std::vector<Value> &values) {
    ASSERT_TRUE(buffer.view().write(std::as_bytes(std::span(values))).ok());
  }
};

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
