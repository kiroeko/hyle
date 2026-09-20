#include "../compute_provider_cuda/collectives_test_support.h"
#include "hyle/graph/cuda.h"

namespace {
using namespace cccl_test;
namespace graph = hyle::graph;
TEST(CudaDefaultGraph,
     OriginalCcclTopKAndScanKeepIndependentWorkspaceAndOldResults) {
  auto owner = graph::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  const auto desc =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 2>{2, 17})
          .value();
  const auto selected =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 2>{2, 4})
          .value();
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(desc).value();
  const auto top =
      builder.add_node(Operation(TopK{.count = 4}), std::array{x}).value();
  const auto values = builder.result(top, 0).value(),
             indices = builder.result(top, 1).value();
  const auto prefix = builder.add(Scan{}, values).value();
  for (auto id : {values, indices, prefix})
    ASSERT_TRUE(builder.publish(id).ok());
  auto logical = std::move(builder).finish().value();
  auto compiled = graph::cuda::compile(logical, owner, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 2u);
  for (std::size_t i = 0; i < 2; ++i)
    EXPECT_EQ(compiled->report.step(i)->provider.id,
              cuda_native::kCcclProviderId);
  std::array input{Tensor(desc, memory.ref()), Tensor(desc, memory.ref())};
  std::array output{Tensor(selected, memory.ref()),
                    Tensor(selected, memory.ref())};
  std::array index{Tensor(selected, memory.ref()),
                   Tensor(selected, memory.ref())};
  std::array sums{Tensor(selected, memory.ref()),
                  Tensor(selected, memory.ref())};
  std::array queues{owner.create_queue().value(), owner.create_queue().value()};
  std::array slots{compiled->executable.create_invocation().value(),
                   compiled->executable.create_invocation().value()};
  for (std::size_t i = 0; i < 2; ++i) {
    ASSERT_TRUE(slots[i].bind(x, input[i].view.as_const()).ok());
    ASSERT_TRUE(slots[i].bind(values, output[i].view).ok());
    ASSERT_TRUE(slots[i].bind(indices, index[i].view).ok());
    ASSERT_TRUE(slots[i].bind(prefix, sums[i].view).ok());
    const auto status = graph::cuda::prepare_replay(slots[i], queues[i]);
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(slots[i].replay_route(), graph::ReplayRoute::kCommands);
  }
  std::vector<std::int32_t> old_values, old_indices, old_sums;
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (std::size_t lane = repeat == 2 ? 1 : 0; lane < 2; ++lane) {
      std::vector<std::int32_t> data(34);
      for (std::size_t j = 0; j < data.size(); ++j)
        data[j] = static_cast<std::int32_t>((j * 3 + lane + repeat) % 11);
      input[lane].write<std::int32_t>(data);
      auto done = slots[lane].submit(queues[lane]);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      const auto actual = output[lane].read<std::int32_t>();
      const auto chosen = index[lane].read<std::int32_t>();
      const auto accumulated = sums[lane].read<std::int32_t>();
      for (std::size_t row = 0; row < 2; ++row) {
        std::array<std::int32_t, 17> order{};
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
          return data[row * 17 + a] > data[row * 17 + b];
        });
        std::int32_t total = 0;
        for (std::size_t j = 0; j < 4; ++j) {
          const auto expected = data[row * 17 + order[j]];
          total += expected;
          EXPECT_EQ(actual[row * 4 + j], expected);
          EXPECT_EQ(chosen[row * 4 + j], order[j]);
          EXPECT_EQ(accumulated[row * 4 + j], total);
        }
      }
      if (lane == 0) {
        old_values = actual;
        old_indices = chosen;
        old_sums = accumulated;
      }
    }
    EXPECT_EQ(output[0].read<std::int32_t>(), old_values);
    EXPECT_EQ(index[0].read<std::int32_t>(), old_indices);
    EXPECT_EQ(sums[0].read<std::int32_t>(), old_sums);
  }
}

TEST(CudaDefaultGraph,
     OriginalCcclOverflowPreservesOutputAndFreshInvocationRecovers) {
  auto owner = graph::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto desc =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 1>{3})
          .value();
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(desc).value(),
             y = builder.add(Scan{}, x).value();
  ASSERT_TRUE(builder.publish(y).ok());
  auto logical = std::move(builder).finish().value();
  auto compiled = graph::cuda::compile(logical, owner, memory.ref()).value();
  ASSERT_EQ(compiled.report.step(0)->provider.id, cuda_native::kCcclProviderId);
  Tensor input(desc, memory.ref()), output(desc, memory.ref());
  for (bool invalid : {true, false}) {
    input.write<std::int32_t>(invalid ? std::array{INT32_MAX, 1, -1}
                                      : std::array{1, 2, 3});
    output.write<std::int32_t>(std::array{-4, -5, -6});
    auto slot = compiled.executable.create_invocation().value();
    ASSERT_TRUE(slot.bind(x, input.view.as_const()).ok());
    ASSERT_TRUE(slot.bind(y, output.view).ok());
    ASSERT_TRUE(graph::cuda::prepare_replay(
                    slot, queue,
                    {.mode = graph::ReplayMode::kCommands,
                     .host_wait = graph::HostWaitPolicy::kAllow})
                    .ok());
    auto done = slot.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    const auto status = done->wait();
    EXPECT_EQ(status.IsOutOfRange(), invalid);
    EXPECT_EQ(status.ok(), !invalid);
    EXPECT_EQ(output.read<std::int32_t>(),
              (invalid ? std::vector{-4, -5, -6} : std::vector{1, 3, 6}));
  }
}
} // namespace
