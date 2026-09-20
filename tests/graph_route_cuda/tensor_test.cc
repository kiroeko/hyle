#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <span>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/mem/stats.h"

namespace {
namespace compute = hyle::compute;
namespace cuda = compute::cuda;
namespace graph = hyle::graph;
namespace mem = hyle::mem;

struct Tensor final {
  mem::Buffer buffer;
  compute::TensorView view;
  Tensor(const compute::TensorDesc &desc, mem::MemoryResourceRef resource)
      : buffer(mem::Buffer::create(resource, desc.required_bytes().value(), 256)
                   .value()),
        view(compute::TensorView::create(desc, buffer.view()).value()) {}
};

TEST(CudaDefaultGraph,
     TensorChainCapturesViewsConversionsMasksAndIndependentResults) {
  using Value = hyle::simd::bfloat16_t;
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        graph::cuda::create_context({.metadata_allocator = stats.get()})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    for (const std::int64_t width : {129, 2048, 6144}) {
      const auto input_desc =
          compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                     std::array<std::int64_t, 2>{2, 2 * width})
              .value();
      const auto output_desc =
          compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                     std::array<std::int64_t, 2>{2, width})
              .value();
      const auto mask_desc =
          compute::TensorDesc::dense(compute::DataType::kBool,
                                     std::array<std::int64_t, 2>{2, 1})
              .value();
      auto builder = graph::GraphBuilder::create().value();
      const auto input_id = builder.input(input_desc).value();
      const auto mask_id = builder.input(mask_desc).value();
      const auto slice = compute::View::slice(1, 1, width, 2).value();
      const auto sliced = builder.add(slice, input_id).value();
      const auto copied = builder.add(compute::Copy{}, sliced).value();
      const auto wide =
          builder.add(compute::Cast{compute::DataType::kFloat32}, copied)
              .value();
      const auto transformed =
          builder.add(compute::Unary{compute::UnaryKind::kSoftplus}, wide)
              .value();
      const auto narrowed =
          builder.add(compute::Cast{compute::DataType::kBFloat16}, transformed)
              .value();
      const auto residual =
          builder.add(compute::Binary{}, copied, narrowed).value();
      const auto zero =
          builder
              .add(compute::Fill{
                  compute::TensorShape::create(std::span<const std::int64_t>{})
                      .value(),
                  compute::DataType::kBFloat16,
                  compute::ScalarValue::from_bits(compute::DataType::kBFloat16,
                                                  0)})
              .value();
      const auto output_id =
          builder.add(compute::Select{}, mask_id, residual, zero).value();
      ASSERT_TRUE(builder.publish(output_id).ok());
      auto logical = std::move(builder).finish().value();
      auto compiled = graph::cuda::compile(
          logical, owner, resource.ref(),
          {.provider_replay = graph::ProviderReplayPolicy::kRequirePrepared});
      ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
      std::array inputs{Tensor(input_desc, resource.ref()),
                        Tensor(input_desc, resource.ref())};
      std::array masks{Tensor(mask_desc, resource.ref()),
                       Tensor(mask_desc, resource.ref())};
      std::array outputs{Tensor(output_desc, resource.ref()),
                         Tensor(output_desc, resource.ref())};
      std::array queues{owner.create_queue().value(),
                        owner.create_queue().value()};
      std::array slots{compiled->executable.create_invocation().value(),
                       compiled->executable.create_invocation().value()};
      for (std::size_t lane = 0; lane < 2; ++lane) {
        ASSERT_TRUE(
            slots[lane].bind(input_id, inputs[lane].view.as_const()).ok());
        ASSERT_TRUE(
            slots[lane].bind(mask_id, masks[lane].view.as_const()).ok());
        ASSERT_TRUE(slots[lane].bind(output_id, outputs[lane].view).ok());
        const auto status = graph::cuda::prepare_replay(
            slots[lane], queues[lane],
            {.mode = graph::ReplayMode::kRequireCapture,
             .host_wait = graph::HostWaitPolicy::kReject});
        ASSERT_TRUE(status.ok()) << status.message();
        EXPECT_EQ(slots[lane].replay_route(), graph::ReplayRoute::kCaptured);
      }
      std::array<compute::TensorDesc, 1> slice_desc{};
      ASSERT_TRUE(compute::infer_output_descriptions(compute::Operation(slice),
                                                     std::array{input_desc},
                                                     slice_desc)
                      .ok());
      auto direct = compute::prepare(owner, compute::Copy{}, slice_desc[0],
                                     resource.ref());
      ASSERT_TRUE(direct.has_value()) << direct.error().message();
      EXPECT_EQ(direct->provider().id, compute::ggml_cuda::kProviderId);
      auto direct_slot = direct->create_invocation().value();
      Tensor direct_output(output_desc, resource.ref());
      ASSERT_TRUE(queues[0].reserve_completions(2).ok());
      const auto before = stats.snapshot();
      std::vector<Value> saved;
      for (int repeat = 0; repeat < 3; ++repeat) {
        std::array<compute::Completion, 2> completions;
        std::array<std::vector<Value>, 2> values;
        for (std::size_t lane = 0; lane < 2; ++lane) {
          values[lane].resize(static_cast<std::size_t>(4 * width));
          for (std::size_t i = 0; i < values[lane].size(); ++i)
            values[lane][i] =
                half_test::encode<Value>((static_cast<int>(i % 41) - 20 +
                                          repeat + static_cast<int>(lane)) *
                                         0.125);
          ASSERT_TRUE(inputs[lane]
                          .buffer.view()
                          .write(std::as_bytes(std::span(values[lane])))
                          .ok());
          const std::array<std::uint8_t, 2> mask{
              static_cast<std::uint8_t>((repeat + lane) % 2), 1};
          ASSERT_TRUE(masks[lane]
                          .buffer.view()
                          .write(std::as_bytes(std::span(mask)))
                          .ok());
          if (repeat == 2 && lane == 0)
            continue; // Retain this slot's previous result.
          auto done = slots[lane].submit(queues[lane]);
          ASSERT_TRUE(done.has_value()) << done.error().message();
          completions[lane] = std::move(*done);
        }
        for (std::size_t lane = 0; lane < 2; ++lane) {
          if (repeat == 2 && lane == 0) {
            const auto retained = half_test::read<Value>(outputs[lane].buffer);
            ASSERT_EQ(retained.size(), saved.size());
            for (std::size_t i = 0; i < retained.size(); ++i)
              EXPECT_EQ(retained[i].bits, saved[i].bits);
            continue;
          }
          ASSERT_TRUE(completions[lane].wait().ok());
          const auto actual = half_test::read<Value>(outputs[lane].buffer);
          for (std::size_t i = 0; i < actual.size(); ++i) {
            const float x =
                static_cast<float>(half_test::decode(values[lane][2 * i + 1]));
            const auto transformed_value =
                half_test::encode<Value>(static_cast<float>(
                    std::log1p(std::exp(static_cast<double>(x)))));
            const double expected =
                i < static_cast<std::size_t>(width) && (repeat + lane) % 2 == 0
                    ? 0
                    : half_test::decode(half_test::encode<Value>(
                          x + static_cast<float>(
                                  half_test::decode(transformed_value))));
            half_test::expect_near(actual[i], expected);
          }
          if (lane == 0)
            saved = actual;
        }
        const auto direct_input =
            compute::ConstTensorView::create(
                slice_desc[0],
                inputs[0]
                    .buffer.view()
                    .subview(sizeof(Value),
                             slice_desc[0].required_bytes().value())
                    .value())
                .value();
        auto copied_done =
            direct_slot.submit(queues[0], direct_input, direct_output.view);
        ASSERT_TRUE(copied_done.has_value());
        ASSERT_TRUE(copied_done->wait().ok());
        const auto copied_values = half_test::read<Value>(direct_output.buffer);
        for (std::size_t i = 0; i < copied_values.size(); ++i)
          EXPECT_EQ(copied_values[i].bits, values[0][2 * i + 1].bits);
      }
      EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
      EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
    }
    ASSERT_TRUE(owner.close().ok());
  }
  EXPECT_TRUE(stats.balanced());
}
} // namespace
