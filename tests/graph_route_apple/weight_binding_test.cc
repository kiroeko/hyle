#include <gtest/gtest.h>

#include <array>
#include <bit>

#include "hyle/graph/apple.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/result.h"
#include "hyle/mem/quota_allocator.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace g = hyle::graph;
namespace m = hyle::mem;
const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
    std::array<std::int64_t, 1>{4}).value();
struct HostWeight final { m::Buffer storage; c::ConstTensorView tensor; };
HostWeight host_weight(const std::array<float, 4> &values) {
  auto storage = m::Buffer::create(m::system_resource(), sizeof(values), 4).value();
  EXPECT_TRUE(storage.view().write(std::as_bytes(std::span(values))).ok());
  auto tensor = c::ConstTensorView::create(description, storage.view()).value();
  return {std::move(storage), tensor};
}
void expect_weight(const g::GraphOutput &output, const std::array<float, 4> &expected) {
  auto tensor = output.tensor();
  ASSERT_TRUE(tensor) << tensor.error().message();
  std::array<float, 4> actual{};
  ASSERT_TRUE(tensor->storage().read(std::as_writable_bytes(std::span(actual))).ok());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]), std::bit_cast<std::uint32_t>(expected[i]));
}

TEST(AppleWeightBinding, RealSelectionsShareOneProducerAndGuardTwoOriginalSlots) {
  m::QuotaAllocator metadata(m::system_allocator());
  auto context = a::Context::create().value();
  auto memory = a::MemoryResource::create(context).value();
  const std::array catalog{a::WeightSpec{10, description}, a::WeightSpec{20, description}};
  auto weights = a::WeightSlots::create(context, memory, catalog, 1024,
      {.allocator = metadata.get()}).value();
  struct CloseWeights {
    a::WeightSlots &slots;
    ~CloseWeights() { slots.close().IgnoreError(); }
  } close_weights{weights};
  auto x = host_weight({1.0F, -2.0F, 3.5F, .25F});
  auto y = host_weight({-.5F, 4.0F, 2.5F, -.25F});
  auto z = host_weight({2.0F, .5F, -1.0F, 4.0F});
  auto t = host_weight({-1.0F, -.25F, 2.0F, -2.0F});
  const std::array selection0{a::WeightFragment{10, x.tensor}, a::WeightFragment{20, y.tensor}};
  const std::array selection1{a::WeightFragment{10, z.tensor}, a::WeightFragment{20, t.tensor}};
  auto builder = g::GraphBuilder::create().value();
  const auto left = builder.input(description).value(), right = builder.input(description).value();
  const auto sum = builder.add(c::Binary{c::BinaryKind::kAdd}, left, right).value();
  ASSERT_TRUE(builder.publish(sum).ok());
  const auto graph = std::move(builder).finish().value();
  const auto source = g::apple::execution_source(context).value();
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  auto compiled = g::compile(graph, {domains, targets});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->executable.required_queue_count(), 0U);
  auto first = compiled->executable.create_invocation().value();
  auto second = compiled->executable.create_invocation().value();
  auto batch0 = weights.upload(0, selection0).value();
  auto batch1 = weights.upload(1, selection1).value();
  // No caller wait/binding() before the ordinary Graph bind. This exercises
  // the real upload owner, without claiming the physical transfer stays pending.
  ASSERT_TRUE(g::apple::bind(first, left, batch0, 10).ok());
  ASSERT_TRUE(g::apple::bind(first, right, batch0, 20).ok());
  EXPECT_TRUE(g::apple::bind(first, left, batch0, 99).IsNotFound());
  EXPECT_TRUE(first.prepare_replay({}, {.host_wait = g::HostWaitPolicy::kReject}).IsUnimplemented());
  ASSERT_TRUE(first.prepare_replay({}).ok());
  const auto prepared_bytes = metadata.used();
  ASSERT_TRUE(first.prepare_replay({}).ok());
  EXPECT_EQ(metadata.used(), prepared_bytes);
  auto report = first.input_source_report().value();
  EXPECT_EQ(report.source_count, 1U);
  EXPECT_EQ(report.host_wait_count, 1U);
  EXPECT_EQ(report.device_wait_count, 0U);
  auto work0 = first.submit();
  ASSERT_TRUE(work0) << work0.error().message();
  ASSERT_TRUE(work0->wait().ok());
  auto old = first.result(sum).value();
  expect_weight(old, {.5F, 2.0F, 6.0F, 0.0F});
  EXPECT_FALSE(g::apple::bind(first, left, batch1, 10).ok());
  EXPECT_FALSE(first.reset().ok());
  ASSERT_TRUE(g::apple::bind(second, left, batch1, 10).ok());
  ASSERT_TRUE(g::apple::bind(second, right, batch0, 20).ok());
  ASSERT_TRUE(second.prepare_replay({}).ok());
  report = second.input_source_report().value();
  EXPECT_EQ(report.source_count, 2U);
  EXPECT_EQ(report.host_wait_count, 2U);
  auto work1 = second.submit();
  ASSERT_TRUE(work1) << work1.error().message();
  ASSERT_TRUE(work1->wait().ok());
  expect_weight(second.result(sum).value(), {1.5F, 4.5F, 1.5F, 3.75F});
  expect_weight(old, {.5F, 2.0F, 6.0F, 0.0F});
  old.reset(); work0 = c::Completion{}; work1 = c::Completion{};
  batch0 = {}; batch1 = {};
  EXPECT_FALSE(weights.upload(0, selection0));
  ASSERT_TRUE(first.close().ok()); ASSERT_TRUE(second.close().ok());
  first = {}; second = {};
  auto reused = weights.upload(0, selection0);
  ASSERT_TRUE(reused) << reused.error().message();
  ASSERT_TRUE(reused->wait().ok());
}
} // namespace
