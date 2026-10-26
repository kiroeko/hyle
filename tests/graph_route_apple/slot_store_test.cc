#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#include "hyle/graph/result.h"
#include "hyle/graph/slots.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace g = hyle::graph;
namespace s = g::slots;

constexpr std::size_t kElements = 4, kSlots = 5;

std::vector<std::byte> values(c::DataType type, bool second) {
  const auto unit = c::data_type_size(type);
  std::vector<std::byte> bytes(kElements * unit);
  if (unit == 4) {
    const auto words = second
        ? std::array<std::uint32_t, 4>{0x40000000, 0xbf800000, 1, 0xff800000}
        : std::array<std::uint32_t, 4>{0x3f800001, 0x80000000, 0x7fc10001, 0x7f800000};
    std::memcpy(bytes.data(), words.data(), bytes.size());
  } else {
    const auto words = type == c::DataType::kFloat16
        ? (second ? std::array<std::uint16_t, 4>{0x4000, 0xbc00, 1, 0xfc00}
                  : std::array<std::uint16_t, 4>{0x3c01, 0x8000, 0x7e51, 0x7c00})
        : (second ? std::array<std::uint16_t, 4>{0x4000, 0xbf80, 1, 0xff80}
                  : std::array<std::uint16_t, 4>{0x3f81, 0x8000, 0x7fc1, 0x7f80});
    std::memcpy(bytes.data(), words.data(), bytes.size());
  }
  return bytes;
}

template<class Lease> std::vector<std::byte> read(const Lease &lease) {
  const auto view = lease.view();
  EXPECT_TRUE(view.has_value()) << view.error().message();
  if (!view) return {};
  std::vector<std::byte> bytes(view->size());
  const auto status = view->read(bytes);
  EXPECT_TRUE(status.ok()) << status.message();
  return bytes;
}

// A real native producer supplies Store rows. Only two exact source item
// leases are readable; padding and other pool items have no Graph read rights.
// No Compute Context/Queue is manufactured for the native route.
void exercise(c::DataType type, std::size_t stride, bool reset_before_submit) {
  SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<unsigned>(type)
      << " stride=" << stride << " reset=" << reset_before_submit);
  auto created = a::Context::create();
  ASSERT_TRUE(created) << created.error().message();
  auto context = std::move(*created);
  auto memory = a::MemoryResource::create(context);
  ASSERT_TRUE(memory) << memory.error().message();
  const auto item = c::TensorDesc::dense(type,
      std::array<std::int64_t, 1>{kElements}).value();
  const auto bytes = item.required_bytes().value();
  ASSERT_GE(stride, bytes);
  auto pool_created = c::LeasePool::create(memory->ref(), kSlots * stride,
      {.max_blocks = 2 * kSlots, .backing_alignment = 64});
  ASSERT_TRUE(pool_created) << pool_created.error().message();
  auto pool = std::move(*pool_created);
  auto layout = c::LeaseSlotLayout::create(pool, item, kSlots, stride);
  ASSERT_TRUE(layout) << layout.error().message();
  std::array<c::WriteLease, kSlots> blocks;
  std::vector<c::ReadLease> padding;
  const std::vector<std::byte> untouched(bytes, std::byte{0xca});
  const std::vector<std::byte> sentinel(bytes, std::byte{0x5c});
  const std::vector<std::byte> padding_bytes(stride - bytes, std::byte{0xa5});
  for (std::size_t i = 0; i < kSlots; ++i) {
    auto block = pool.allocate(bytes, c::data_type_size(type));
    ASSERT_TRUE(block) << block.error().message();
    blocks[i] = std::move(*block);
    ASSERT_EQ(layout->locate(blocks[i])->index, i);
    ASSERT_TRUE(blocks[i].view()->write(untouched).ok());
    if (stride > bytes) {
      auto gap = pool.allocate(stride - bytes, 1);
      ASSERT_TRUE(gap) << gap.error().message();
      ASSERT_TRUE(gap->view()->write(padding_bytes).ok());
      auto ready = gap->publish();
      ASSERT_TRUE(ready) << ready.error().message();
      padding.push_back(std::move(*ready));
    }
  }
  const auto first = values(type, false), second = values(type, true);
  ASSERT_TRUE(blocks[1].view()->write(first).ok());
  ASSERT_TRUE(blocks[3].view()->write(second).ok());
  ASSERT_TRUE(blocks[2].view()->write(sentinel).ok());
  std::array<c::ReadLease, 2> sources{blocks[1].publish().value(), blocks[3].publish().value()};
  auto middle = blocks[2].publish().value();
  std::array<c::WriteLease, 2> destinations{std::move(blocks[0]), std::move(blocks[4])};

  const auto schema = s::Schema::create(item, kSlots, stride).value();
  const auto id_description = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{2}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto capacity = builder.input(schema.capacity_description().value()).value();
  const auto source_ids = builder.input(id_description).value();
  const auto destination_ids = builder.input(id_description).value();
  const auto gather = builder.add_node(c::Operation(c::Gather{.axis = 0,
      .bounds = c::IndexBoundsPolicy::kChecked}), std::array{capacity, source_ids}).value();
  const auto rows = builder.result(gather, 0).value();
  const auto store = builder.add_node(c::Operation::custom(s::Store{schema}),
      std::array{rows, destination_ids}).value();
  ASSERT_TRUE(builder.publish(rows).ok());
  const auto graph = std::move(builder).finish().value();
  const auto source = g::apple::execution_source(context).value();
  const std::array domains{g::StorageDomain{memory->ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  auto compiled = g::compile(graph, {.storage_domains = domains, .targets = targets});
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto invocation = compiled->executable.create_invocation().value();
  const std::array associations{s::SourceBinding{store, gather}};
  auto status = s::prepare(invocation, graph, {}, associations);
  ASSERT_TRUE(status.ok()) << status.message();
  auto bindings = s::SlotBindings::create(*layout, sources, destinations);
  ASSERT_TRUE(bindings) << bindings.error().message();
  const std::array generations{bindings->destinations()[0].generation(),
      bindings->destinations()[1].generation()};
  status = s::bind(invocation, graph, store, std::move(*bindings));
  ASSERT_TRUE(status.ok()) << status.message();
  status = invocation.prepare_replay({});
  ASSERT_TRUE(status.ok()) << status.message();
  if (reset_before_submit) {
    ASSERT_TRUE(invocation.reset().ok());
    EXPECT_FALSE(invocation.prepare_replay({}).ok());
    EXPECT_FALSE(invocation.submit());
  } else {
    auto completed = invocation.submit();
    ASSERT_TRUE(completed) << completed.error().message();
    EXPECT_FALSE(s::take(invocation, store));
    status = completed->wait();
    ASSERT_TRUE(status.ok()) << status.message();
    *completed = {};
    {
      auto output = invocation.result(rows);
      ASSERT_TRUE(output) << output.error().message();
      auto lease = output->storage_lease();
      ASSERT_TRUE(lease) << lease.error().message();
      auto expected = first;
      expected.insert(expected.end(), second.begin(), second.end());
      EXPECT_EQ(read(*lease), expected);
      EXPECT_FALSE(s::take(invocation, store));
    }
  }
  auto returned = s::take(invocation, store);
  ASSERT_TRUE(returned) << returned.error().message();
  ASSERT_EQ(returned->destination_count(), 2u);
  for (std::size_t i = 0; i < 2; ++i)
    EXPECT_EQ(returned->destinations()[i].generation(), generations[i]);
  EXPECT_EQ(read(returned->destinations()[0]), reset_before_submit ? untouched : first);
  EXPECT_EQ(read(returned->destinations()[1]), reset_before_submit ? untouched : second);
  EXPECT_EQ(read(sources[0]), first);
  EXPECT_EQ(read(sources[1]), second);
  EXPECT_EQ(read(middle), sentinel);
  for (const auto &gap : padding) EXPECT_EQ(read(gap), padding_bytes);
  *returned = {};
  auto source_writer = sources[0].try_write();
  ASSERT_TRUE(source_writer) << source_writer.error().message();
  EXPECT_FALSE(invocation.submit());
}

TEST(AppleSlotStore, DenseGeneratedRowsPreserveF32BitsAndResultRetirement) {
  exercise(c::DataType::kFloat32, kElements * sizeof(float), false);
}
TEST(AppleSlotStore, PaddedGeneratedRowsPreserveHalfAndBfloatHistoryBits) {
  exercise(c::DataType::kFloat16, 64, false);
  exercise(c::DataType::kBFloat16, 64, false);
}
TEST(AppleSlotStore, ResetRevokesNativeWritesAndReturnsOriginalTransaction) {
  exercise(c::DataType::kFloat32, 64, true);
}
} // namespace
