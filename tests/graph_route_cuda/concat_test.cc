#include <gtest/gtest.h>

#include <array>
#include <cstdint>
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
  hyle::Status write(std::span<const std::uint16_t> values) {
    return buffer.view().write(std::as_bytes(values));
  }
  std::vector<std::uint16_t> read() const {
    std::vector<std::uint16_t> values(buffer.size() / sizeof(std::uint16_t));
    EXPECT_TRUE(
        buffer.view().read(std::as_writable_bytes(std::span(values))).ok());
    return values;
  }
};

TEST(CudaDefaultGraph,
     ConcatReassemblesViewsAndAppendsWindowWithCapturedSlots) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        graph::cuda::create_context({.metadata_allocator = stats.get()})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    for (const std::int64_t width : {128, 129, 2048}) {
      for (const std::int64_t rows : {1, 17, 23}) {
        SCOPED_TRACE(::testing::Message() << rows << '/' << width);
        const auto desc = compute::TensorDesc::dense(
                              compute::DataType::kBFloat16,
                              std::array<std::int64_t, 2>{rows, 2 * width})
                              .value();
        const auto joined_desc =
            compute::TensorDesc::dense(
                compute::DataType::kBFloat16,
                std::array<std::int64_t, 2>{rows, 5 * width})
                .value();
        const auto window_desc =
            compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                       std::array<std::int64_t, 2>{rows, width})
                .value();
        auto builder = graph::GraphBuilder::create().value();
        const auto input_id = builder.input(desc).value();
        const auto even_op = compute::View::slice(1, 0, width, 2).value();
        const auto odd_op = compute::View::slice(1, 1, width, 2).value();
        const auto even = builder.add(even_op, input_id).value();
        const auto odd = builder.add(odd_op, input_id).value();
        const auto concat_node =
            builder
                .add_node(compute::Operation(compute::Concat{.axis = -1}),
                          std::array{even, odd, even, odd, even})
                .value();
        const auto joined_id = builder.result(concat_node).value();
        const auto retained =
            builder.add(compute::View::slice(1, 1, width - 1).value(), even)
                .value();
        const auto appended =
            builder.add(compute::View::slice(1, 0, 1).value(), odd).value();
        const auto window_id =
            builder.add(compute::Concat{.axis = 1}, retained, appended).value();
        ASSERT_TRUE(builder.publish(joined_id).ok());
        ASSERT_TRUE(builder.publish(window_id).ok());
        auto logical = std::move(builder).finish().value();
        auto compiled = graph::cuda::compile(logical, owner, resource.ref());
        ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
        std::array<compute::TensorDesc, 1> slice_desc;
        ASSERT_TRUE(
            compute::infer_output_descriptions(compute::Operation(even_op),
                                               std::array{desc}, slice_desc)
                .ok());
        std::array<compute::TensorDesc, 5> descriptions;
        descriptions.fill(slice_desc[0]);
        auto direct = compute::prepare(owner, compute::Concat{.axis = -1},
                                       descriptions, resource.ref());
        ASSERT_TRUE(direct.has_value()) << direct.error().message();
        EXPECT_EQ(direct->provider().id, compute::cuda_native::kProviderId);
        auto direct_slot = direct->create_invocation().value();
        Tensor direct_output(joined_desc, resource.ref());
        std::array inputs{Tensor(desc, resource.ref()),
                          Tensor(desc, resource.ref())};
        std::array joined{Tensor(joined_desc, resource.ref()),
                          Tensor(joined_desc, resource.ref())};
        std::array windows{Tensor(window_desc, resource.ref()),
                           Tensor(window_desc, resource.ref())};
        std::array queues{owner.create_queue().value(),
                          owner.create_queue().value()};
        std::array slots{compiled->executable.create_invocation().value(),
                         compiled->executable.create_invocation().value()};
        for (std::size_t lane = 0; lane < 2; ++lane) {
          ASSERT_TRUE(
              slots[lane].bind(input_id, inputs[lane].view.as_const()).ok());
          ASSERT_TRUE(slots[lane].bind(joined_id, joined[lane].view).ok());
          ASSERT_TRUE(slots[lane].bind(window_id, windows[lane].view).ok());
          const auto status = graph::cuda::prepare_replay(
              slots[lane], queues[lane],
              {.mode = graph::ReplayMode::kRequireCapture,
               .host_wait = graph::HostWaitPolicy::kReject});
          ASSERT_TRUE(status.ok()) << status.message();
          EXPECT_EQ(slots[lane].replay_route(), graph::ReplayRoute::kCaptured);
        }
        ASSERT_TRUE(queues[0].reserve_completions(2).ok());
        const auto before = stats.snapshot();
        std::vector<std::uint16_t> retained_joined, retained_window;
        for (int repeat = 0; repeat < 3; ++repeat) {
          std::array<compute::Completion, 2> done;
          std::array<std::vector<std::uint16_t>, 2> data;
          for (std::size_t lane = 0; lane < 2; ++lane) {
            if (repeat == 2 && lane == 0)
              continue;
            data[lane].resize(static_cast<std::size_t>(rows * 2 * width));
            for (std::size_t index = 0; index < data[lane].size(); ++index)
              data[lane][index] = static_cast<std::uint16_t>(
                  index * 997 + repeat * 133 + lane * 37);
            ASSERT_TRUE(inputs[lane].write(data[lane]).ok());
            auto submitted = slots[lane].submit(queues[lane]);
            ASSERT_TRUE(submitted.has_value()) << submitted.error().message();
            done[lane] = std::move(*submitted);
          }
          for (std::size_t lane = 0; lane < 2; ++lane) {
            if (repeat == 2 && lane == 0) {
              EXPECT_EQ(joined[lane].read(), retained_joined);
              EXPECT_EQ(windows[lane].read(), retained_window);
              continue;
            }
            ASSERT_TRUE(done[lane].wait().ok());
            std::vector<std::uint16_t> expected_joined, expected_window;
            for (std::size_t row = 0; row < static_cast<std::size_t>(rows);
                 ++row) {
              for (std::size_t block = 0; block < 5; ++block)
                for (std::size_t col = 0; col < static_cast<std::size_t>(width);
                     ++col)
                  expected_joined.push_back(
                      data[lane][row * 2 * width + 2 * col + block % 2]);
              for (std::size_t col = 1; col < static_cast<std::size_t>(width);
                   ++col)
                expected_window.push_back(
                    data[lane][row * 2 * width + 2 * col]);
              expected_window.push_back(data[lane][row * 2 * width + 1]);
            }
            EXPECT_EQ(joined[lane].read(), expected_joined);
            EXPECT_EQ(windows[lane].read(), expected_window);
            if (lane == 0) {
              const auto even_view =
                  compute::ConstTensorView::create(
                      slice_desc[0],
                      inputs[0]
                          .buffer.view()
                          .subview(0, slice_desc[0].required_bytes().value())
                          .value())
                      .value();
              const auto odd_view =
                  compute::ConstTensorView::create(
                      slice_desc[0],
                      inputs[0]
                          .buffer.view()
                          .subview(sizeof(std::uint16_t),
                                   slice_desc[0].required_bytes().value())
                          .value())
                      .value();
              auto direct_done =
                  direct_slot.submit(queues[0],
                                     std::array{even_view, odd_view, even_view,
                                                odd_view, even_view},
                                     direct_output.view);
              ASSERT_TRUE(direct_done.has_value())
                  << direct_done.error().message();
              ASSERT_TRUE(direct_done->wait().ok());
              EXPECT_EQ(direct_output.read(), expected_joined);
              retained_joined = expected_joined;
              retained_window = expected_window;
            }
          }
        }
        EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
        EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
      }
    }
    ASSERT_TRUE(owner.close().ok());
  }
  EXPECT_TRUE(stats.balanced());
}
} // namespace
