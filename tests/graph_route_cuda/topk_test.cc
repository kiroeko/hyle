#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <span>
#include <vector>

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
struct Tensor final {
  mem::Buffer buffer;
  compute::TensorView view;
  Tensor(const compute::TensorDesc &description,
         mem::MemoryResourceRef resource)
      : buffer(mem::Buffer::create(resource,
                                   description.required_bytes().value(), 256)
                   .value()),
        view(compute::TensorView::create(description, buffer.view()).value()) {}
  template <class T> hyle::Status write(std::span<const T> values) {
    return buffer.view().write(std::as_bytes(values));
  }
  template <class T> std::vector<T> read() const {
    std::vector<T> values(buffer.size() / sizeof(T));
    EXPECT_TRUE(
        buffer.view().read(std::as_writable_bytes(std::span(values))).ok());
    return values;
  }
};

TEST(CudaDefaultGraph, SoftmaxTopKAndRenormalizationCaptureBothOutputs) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        graph::cuda::create_context({.metadata_allocator = stats.get()})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    const std::array shapes{std::pair{256LL, 6U}, std::pair{288LL, 8U},
                            std::pair{384LL, 6U}, std::pair{512LL, 10U},
                            std::pair{896LL, 16U}};
    for (const auto &[experts, configured] : shapes) {
      for (const std::int64_t rows : {1, 17, 23}) {
        for (const std::size_t count :
             {std::size_t{1}, static_cast<std::size_t>(configured)}) {
          SCOPED_TRACE(::testing::Message()
                       << rows << '/' << experts << '/' << count);
          const auto desc = compute::TensorDesc::dense(
                                compute::DataType::kFloat32,
                                std::array<std::int64_t, 2>{rows, experts})
                                .value();
          const auto values_desc =
              compute::TensorDesc::dense(
                  compute::DataType::kFloat32,
                  std::array<std::int64_t, 2>{rows,
                                              static_cast<std::int64_t>(count)})
                  .value();
          const auto indices_desc =
              compute::TensorDesc::dense(
                  compute::DataType::kInt32,
                  std::array<std::int64_t, 2>{rows,
                                              static_cast<std::int64_t>(count)})
                  .value();
          const compute::TopK top_op{.count = count};
          auto builder = graph::GraphBuilder::create().value();
          const auto input_id = builder.input(desc).value();
          const auto probabilities =
              builder.add(compute::Softmax{}, input_id).value();
          const auto top = builder
                               .add_node(compute::Operation(top_op),
                                         std::array{probabilities})
                               .value();
          const auto values_id = builder.result(top, 0).value();
          const auto indices_id = builder.result(top, 1).value();
          const auto sum =
              builder
                  .add(compute::Reduce{.axes =
                                           compute::Axes::from(
                                               std::array<std::int32_t, 1>{-1})
                                               .value(),
                                       .keep_dimensions = true,
                                       .accumulation_type =
                                           compute::DataType::kFloat32},
                       values_id)
                  .value();
          const auto weights_id =
              builder
                  .add(compute::Binary{compute::BinaryKind::kDivide}, values_id,
                       sum)
                  .value();
          ASSERT_TRUE(builder.publish(values_id).ok());
          ASSERT_TRUE(builder.publish(indices_id).ok());
          ASSERT_TRUE(builder.publish(weights_id).ok());
          auto logical = std::move(builder).finish().value();
          auto compiled = graph::cuda::compile(
              logical, owner, resource.ref(),
              {.provider_replay =
                   graph::ProviderReplayPolicy::kRequirePrepared});
          ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
          // Ordinary preparation must skip providers of other families. A
          // multi-output TopK must not be treated as a malformed gated op.
          auto direct_softmax =
              compute::prepare(owner, compute::Softmax{}, desc, resource.ref());
          ASSERT_TRUE(direct_softmax.has_value())
              << direct_softmax.error().message();
          auto direct_top =
              compute::prepare(owner, top_op, desc, resource.ref());
          ASSERT_TRUE(direct_top.has_value()) << direct_top.error().message();
          EXPECT_EQ(direct_top->provider().id,
                    compute::cuda_native::kProviderId);
          auto softmax_slot = direct_softmax->create_invocation().value();
          auto top_slot = direct_top->create_invocation().value();
          Tensor direct_probabilities(desc, resource.ref()),
              direct_values(values_desc, resource.ref()),
              direct_indices(indices_desc, resource.ref());
          std::array inputs{Tensor(desc, resource.ref()),
                            Tensor(desc, resource.ref())};
          std::array values{Tensor(values_desc, resource.ref()),
                            Tensor(values_desc, resource.ref())};
          std::array indices{Tensor(indices_desc, resource.ref()),
                             Tensor(indices_desc, resource.ref())};
          std::array weights{Tensor(values_desc, resource.ref()),
                             Tensor(values_desc, resource.ref())};
          std::array queues{owner.create_queue().value(),
                            owner.create_queue().value()};
          std::array slots{compiled->executable.create_invocation().value(),
                           compiled->executable.create_invocation().value()};
          for (std::size_t lane = 0; lane < 2; ++lane) {
            ASSERT_TRUE(
                slots[lane].bind(input_id, inputs[lane].view.as_const()).ok());
            ASSERT_TRUE(slots[lane].bind(values_id, values[lane].view).ok());
            ASSERT_TRUE(slots[lane].bind(indices_id, indices[lane].view).ok());
            ASSERT_TRUE(slots[lane].bind(weights_id, weights[lane].view).ok());
            const auto status = graph::cuda::prepare_replay(
                slots[lane], queues[lane],
                {.mode = graph::ReplayMode::kRequireCapture,
                 .host_wait = graph::HostWaitPolicy::kReject});
            ASSERT_TRUE(status.ok()) << status.message();
            EXPECT_EQ(slots[lane].replay_route(),
                      graph::ReplayRoute::kCaptured);
          }
          ASSERT_TRUE(queues[0].reserve_completions(2).ok());
          const auto before = stats.snapshot();
          std::vector<float> retained_values, retained_weights;
          std::vector<std::int32_t> retained_indices;
          for (int repeat = 0; repeat < 3; ++repeat) {
            std::array<compute::Completion, 2> done;
            std::array<std::vector<float>, 2> data;
            for (std::size_t lane = 0; lane < 2; ++lane) {
              if (repeat == 2 && lane == 0)
                continue;
              data[lane].resize(static_cast<std::size_t>(rows * experts));
              for (std::size_t i = 0; i < data[lane].size(); ++i)
                data[lane][i] = static_cast<float>(
                    (static_cast<int>((i + repeat * 17 + lane * 11) % 53) -
                     26) *
                    0.125);
              ASSERT_TRUE(inputs[lane].write<float>(data[lane]).ok());
              auto submitted = slots[lane].submit(queues[lane]);
              ASSERT_TRUE(submitted.has_value()) << submitted.error().message();
              done[lane] = std::move(*submitted);
            }
            for (std::size_t lane = 0; lane < 2; ++lane) {
              if (repeat == 2 && lane == 0) {
                EXPECT_EQ(values[lane].read<float>(), retained_values);
                EXPECT_EQ(indices[lane].read<std::int32_t>(), retained_indices);
                EXPECT_EQ(weights[lane].read<float>(), retained_weights);
                continue;
              }
              ASSERT_TRUE(done[lane].wait().ok());
              const auto actual_values = values[lane].read<float>();
              const auto actual_indices = indices[lane].read<std::int32_t>();
              const auto actual_weights = weights[lane].read<float>();
              std::vector<double> expected_values(actual_values.size());
              for (std::size_t row = 0; row < static_cast<std::size_t>(rows);
                   ++row) {
                std::vector<std::int32_t> order(
                    static_cast<std::size_t>(experts));
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(),
                                 [&](auto left, auto right) {
                                   return data[lane][row * experts + left] >
                                          data[lane][row * experts + right];
                                 });
                double all_exp = 0, selected_exp = 0;
                for (std::size_t col = 0;
                     col < static_cast<std::size_t>(experts); ++col)
                  all_exp += std::exp(
                      static_cast<double>(data[lane][row * experts + col]));
                for (std::size_t col = 0; col < count; ++col)
                  selected_exp += std::exp(static_cast<double>(
                      data[lane][row * experts + order[col]]));
                for (std::size_t col = 0; col < count; ++col) {
                  const auto index = row * count + col;
                  EXPECT_EQ(actual_indices[index], order[col]);
                  const auto exponent = std::exp(static_cast<double>(
                      data[lane][row * experts + order[col]]));
                  expected_values[index] = exponent / all_exp;
                  EXPECT_NEAR(actual_values[index], exponent / all_exp,
                              exponent / all_exp * 4e-6 + 2e-7);
                  EXPECT_NEAR(actual_weights[index], exponent / selected_exp,
                              exponent / selected_exp * 4e-6 + 2e-7);
                }
              }
              if (lane == 0) {
                auto direct_done =
                    softmax_slot.submit(queues[0], inputs[0].view.as_const(),
                                        direct_probabilities.view);
                ASSERT_TRUE(direct_done.has_value());
                ASSERT_TRUE(direct_done->wait().ok());
                direct_done = top_slot.submit(
                    queues[0], direct_probabilities.view.as_const(),
                    direct_values.view, direct_indices.view);
                ASSERT_TRUE(direct_done.has_value());
                ASSERT_TRUE(direct_done->wait().ok());
                // Ordinary Direct can bind managed ggml while this graph
                // requires a prepared capture program. Check both against
                // the same independent oracle and its existing tolerance.
                const auto direct_actual = direct_values.read<float>();
                for (std::size_t i = 0; i < direct_actual.size(); ++i)
                  EXPECT_NEAR(direct_actual[i], expected_values[i],
                              expected_values[i] * 4e-6 + 2e-7);
                EXPECT_EQ(direct_indices.read<std::int32_t>(), actual_indices);
                retained_values = actual_values;
                retained_indices = actual_indices;
                retained_weights = actual_weights;
              }
            }
          }
          EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
          EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
        }
      }
    }
    ASSERT_TRUE(owner.close().ok());
  }
  EXPECT_TRUE(stats.balanced());
}
} // namespace
