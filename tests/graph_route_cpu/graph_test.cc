#include <gtest/gtest.h>

#include <array>
#include <span>
#include <vector>

#include "hyle/graph/cpu.h"
#include "hyle/mem/stats.h"

namespace {
namespace compute = hyle::compute;
namespace graph = hyle::graph;

TEST(CpuDefaultGraph, NonModelGraphReplaysAndReclaimsItsOwners) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto context =
        graph::cpu::create_context({.metadata_allocator = stats.get()}).value();
    auto queue = context.create_queue().value();
    const auto description =
        compute::TensorDesc::dense(compute::DataType::kFloat32,
                                   std::array<std::int64_t, 1>{5})
            .value();
    auto builder = graph::GraphBuilder::create().value();
    const auto input = builder.input(description).value();
    const auto negative =
        builder.add(compute::Unary{compute::UnaryKind::kNeg}, input).value();
    const auto absolute =
        builder.add(compute::Unary{compute::UnaryKind::kAbs}, negative).value();
    ASSERT_TRUE(builder.publish(absolute).ok());
    auto logical = std::move(builder).finish().value();
    auto compiled = graph::compile(logical, context).value();
    EXPECT_EQ(compiled.report.node_count(), 2U);
    EXPECT_EQ(compiled.report.region_count(), 0U);
    auto source =
        hyle::mem::Buffer::create(hyle::mem::system_resource(), 20, 64).value();
    auto destination =
        hyle::mem::Buffer::create(hyle::mem::system_resource(), 20, 64).value();
    auto invocation = compiled.executable.create_invocation().value();
    ASSERT_TRUE(invocation
                    .bind(input, compute::ConstTensorView::create(description,
                                                                  source.view())
                                     .value())
                    .ok());
    ASSERT_TRUE(invocation
                    .bind(absolute, compute::TensorView::create(
                                        description, destination.view())
                                        .value())
                    .ok());
    const std::array queues{graph::QueueBinding{
        .lane = graph::ExecutionLaneId{0}, .queue = &queue}};
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    const auto before = stats.snapshot();
    for (float seed : {1.0F, 3.0F, 7.0F}) {
      const std::array values{-seed, 0.0F, seed, -2 * seed, 3 * seed};
      ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
      auto completion = invocation.submit(queue);
      ASSERT_TRUE(completion.has_value()) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      std::array<float, 5> actual{};
      ASSERT_TRUE(destination.view()
                      .read(std::as_writable_bytes(std::span(actual)))
                      .ok());
      EXPECT_EQ(actual, (std::array{seed, 0.0F, seed, 2 * seed, 3 * seed}));
    }
    EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
    EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
    ASSERT_TRUE(context.close().ok());
    auto closed = invocation.submit(queue);
    ASSERT_FALSE(closed.has_value());
    EXPECT_TRUE(closed.error().IsFailedPrecondition());
  }
  EXPECT_TRUE(stats.balanced());
}

} // namespace
