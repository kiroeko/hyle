#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyle/compute/vulkan_memory.h"
#include "hyle/graph/result.h"
#include "hyle/graph/slots.h"
#include "hyle/graph/vulkan.h"
#include "../../src/compute_route_vulkan/vulkan_copy_internal.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace s = g::slots;
namespace m = hyle::mem;
template<class T> T take(hyle::Result<T> result, const char *what) {
  if (!result) throw std::runtime_error(std::string(what) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
void exercise(c::DataType type, std::size_t item_bytes, std::size_t stride) {
  auto context = take(v::Context::create(), "Vulkan Context");
  auto device = take(v::MemoryResource::create(context, v::MemoryKind::kDevice), "device storage");
  auto staging = take(v::MemoryResource::create(context, v::MemoryKind::kStaging), "host staging");
  const auto item = take(c::TensorDesc::dense(type, std::array<std::int64_t, 1>{8}), "item");
  const auto schema = take(s::Schema::create(item, 5, stride), "schema");
  std::vector<std::byte> original(5 * stride, std::byte{0xa5});
  // Raw words include signed zero/NaN-like payloads. No floating arithmetic or
  // conversion is part of Gather or Store; compare every byte, including gaps.
  for (const auto slot : {1u, 3u})
    for (std::size_t byte = 0; byte < item_bytes; ++byte)
      original[slot * stride + byte] = static_cast<std::byte>((slot * 47 + byte * 13) & 255u);
  auto backing = take(m::Buffer::create(device.ref(), original.size(), 64), "pool backing");
  ASSERT_TRUE(backing.view().write(original).ok());
  const m::ConstBufferView diagnostic_backing = backing.view();
  auto pool = take(c::LeasePool::adopt(std::move(backing), {.max_blocks = 5}), "pool adopt");
  const auto layout = take(c::LeaseSlotLayout::create(pool, item, 5, stride), "layout");
  std::array<c::WriteLease, 5> blocks;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    blocks[i] = take(pool.allocate(item_bytes, stride), "exact item");
    ASSERT_EQ(take(layout.locate(blocks[i]), "item address").index, i);
  }
  const std::array sources{take(blocks[3].publish(), "source3"), take(blocks[1].publish(), "source1")};
  const auto sentinel = take(blocks[2].publish(), "unused source");
  std::array destinations{std::move(blocks[4]), std::move(blocks[0])};
  auto builder = take(g::GraphBuilder::create(), "GraphBuilder");
  const auto bank = take(builder.input(take(schema.capacity_description(), "capacity")), "bank input");
  const auto ids_desc = take(c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{2}), "IDs description");
  const auto src_ids = take(builder.input(ids_desc), "source IDs");
  const auto dst_ids = take(builder.input(ids_desc), "destination IDs");
  const auto gather = take(builder.add_node(c::Operation(c::Gather{.axis = 0,
      .bounds = c::IndexBoundsPolicy::kChecked}), std::array{bank, src_ids}), "Gather");
  const auto rows = take(builder.result(gather), "Gather rows");
  const auto store = take(builder.add_node(c::Operation::custom(s::Store{schema}),
      std::array{rows, dst_ids}), "Store");
  ASSERT_TRUE(builder.publish(rows).ok());
  const auto graph = take(std::move(builder).finish(), "Graph");
  const auto source = take(g::vulkan::execution_source(context), "native execution source");
  const std::array storage{g::StorageDomain{device.ref()}, g::StorageDomain{staging.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  auto compiled = g::compile(graph, {.storage_domains = storage, .targets = targets});
  ASSERT_TRUE(compiled) << compiled.error().message();
  bool native_store = false;
  for (std::size_t i = 0; i < compiled->report.step_count(); ++i) {
    const auto step = compiled->report.step(i).value();
    for (std::size_t node = 0; node < step.node_count; ++node)
      if (compiled->report.step_node(i, node).value() == store) {
        EXPECT_EQ(step.runtime.mode, c::ProviderRuntimeMode::kDelegated);
        EXPECT_EQ(step.implementation, c::ProviderImplementationId{0x534c4f545354564bULL});
        native_store = true;
      }
  }
  ASSERT_TRUE(native_store);
  auto invocation = take(compiled->executable.create_invocation(), "native Invocation");
  const std::array association{s::SourceBinding{store, gather}};
  auto prepared = s::prepare(invocation, graph, {}, association, {.host_workspace = staging.ref()});
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  auto bindings = take(s::SlotBindings::create(layout, sources, destinations), "source/writer transaction");
  auto bound = s::bind(invocation, graph, store, std::move(bindings));
  ASSERT_TRUE(bound.ok()) << bound.message();
  prepared = invocation.prepare_replay({});
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  auto completed = invocation.submit();
  ASSERT_TRUE(completed) << completed.error().message();
  ASSERT_FALSE(s::take(invocation, store));
  ASSERT_TRUE(completed->wait().ok()); *completed = {};
  std::vector<std::byte> expected_rows(2 * item_bytes);
  std::copy_n(original.begin() + 3 * stride, item_bytes, expected_rows.begin());
  std::copy_n(original.begin() + stride, item_bytes, expected_rows.begin() + item_bytes);
  {
    auto output = take(invocation.result(rows), "actual generated result");
    auto tensor = take(output.tensor(), "result tensor");
    std::vector<std::byte> actual(expected_rows.size());
    ASSERT_TRUE(tensor.storage().read(actual).ok());
    EXPECT_EQ(actual, expected_rows);
    // Actual retained GraphOutput, not an idle flag, prevents owner recovery.
    auto held = s::take(invocation, store); ASSERT_FALSE(held);
    EXPECT_TRUE(held.error().IsUnavailable());
  }
  auto returned = s::take(invocation, store);
  ASSERT_TRUE(returned) << returned.error().message();
  auto expected = original;
  std::copy_n(expected_rows.begin(), item_bytes, expected.begin() + 4 * stride);
  std::copy_n(expected_rows.begin() + item_bytes, item_bytes, expected.begin());
  std::vector<std::byte> observed(expected.size());
  ASSERT_TRUE(diagnostic_backing.read(observed).ok()); EXPECT_EQ(observed, expected);
  for (std::size_t i = 0; i < 2; ++i) {
    std::vector<std::byte> unchanged(item_bytes);
    ASSERT_TRUE(sources[i].view()->read(unchanged).ok());
    EXPECT_EQ(unchanged, (i == 0 ? std::vector<std::byte>(original.begin()+3*stride,original.begin()+3*stride+item_bytes)
                                : std::vector<std::byte>(original.begin()+stride,original.begin()+stride+item_bytes)));
  }
  EXPECT_TRUE(sentinel.valid());
  // A second, successfully prepared transaction is withdrawn by reset. No
  // pages/IDs/borrowed views may revive its write or projected-read authority.
  bound = s::bind(invocation, graph, store, std::move(*returned));
  ASSERT_TRUE(bound.ok()) << bound.message();
  prepared = invocation.prepare_replay({}); ASSERT_TRUE(prepared.ok()) << prepared.message();
  ASSERT_TRUE(invocation.reset().ok());
  EXPECT_FALSE(invocation.prepare_replay({}).ok()); EXPECT_FALSE(invocation.submit());
  auto recovered = s::take(invocation, store); ASSERT_TRUE(recovered) << recovered.error().message();
  ASSERT_EQ(recovered->source_count(), 2u); ASSERT_EQ(recovered->destination_count(), 2u);
  EXPECT_EQ(recovered->destination_addresses()[0].index, 4u);
  EXPECT_EQ(recovered->destination_addresses()[1].index, 0u);
  ASSERT_TRUE(diagnostic_backing.read(observed).ok()); EXPECT_EQ(observed, expected);
}
TEST(VulkanSlotStore, Float32DenseGeneratedRowsKeepExactOwnersAcrossReset) {
  exercise(c::DataType::kFloat32, 32, 32);
}
TEST(VulkanSlotStore, Bfloat16PaddedGeneratedRowsDoNotReadOrWriteGaps) {
  exercise(c::DataType::kBFloat16, 16, 64);
}
void exercise_halfword_store(std::size_t elements) {
  auto context = take(v::Context::create(), "Vulkan Context");
  auto device = take(v::MemoryResource::create(context, v::MemoryKind::kDevice), "device");
  auto staging = take(v::MemoryResource::create(context, v::MemoryKind::kStaging), "staging");
  const auto bytes = elements * 2, stride = bytes + 4;
  const auto item = take(c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 1>{static_cast<std::int64_t>(elements)}), "halfword item");
  const auto schema = take(s::Schema::create(item, 5, stride), "schema");
  std::vector<std::byte> original(5 * stride, std::byte{0xa5});
  auto backing = take(m::Buffer::create(device.ref(), original.size(), 64), "pool backing");
  ASSERT_TRUE(backing.view().write(original).ok());
  const m::ConstBufferView diagnostic = backing.view();
  auto pool = take(c::LeasePool::adopt(std::move(backing), {.max_blocks = 10}), "pool");
  const auto layout = take(c::LeaseSlotLayout::create(pool, item, 5, stride), "layout");
  std::array<c::WriteLease, 5> blocks, gaps;
  for (std::size_t slot = 0; slot < blocks.size(); ++slot) {
    blocks[slot] = take(pool.allocate(bytes, 2), "exact item");
    ASSERT_EQ(take(layout.locate(blocks[slot]), "slot address").index, slot);
    gaps[slot] = take(pool.allocate(4, 2), "unowned row padding");
  }
  const auto untouched = take(blocks[2].publish(), "unused initialized item");
  std::array destinations{std::move(blocks[4]), std::move(blocks[1])};
  auto builder = take(g::GraphBuilder::create(), "builder");
  const auto row_desc = take(c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 2>{2, static_cast<std::int64_t>(elements)}), "rows");
  const auto rows = take(builder.input(row_desc), "rows input");
  const auto ids = take(builder.input(take(c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{2}), "IDs")), "IDs input");
  const auto store = take(builder.add_node(c::Operation::custom(s::Store{schema}), std::array{rows,ids}), "Store");
  const auto graph = take(std::move(builder).finish(), "Graph");
  const auto source = take(g::vulkan::execution_source(context), "source");
  const std::array storage{g::StorageDomain{device.ref()}, g::StorageDomain{staging.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  auto compiled = take(g::compile(graph, {.storage_domains=storage,.targets=targets}), "halfword compile");
  auto invocation = take(compiled.executable.create_invocation(), "Invocation");
  auto input = take(m::Buffer::create(device.ref(), bytes * 2, 2), "source rows");
  // Includes raw BF16 NaN payloads and signed zero; copying must not convert.
  std::vector<std::byte> payload(bytes * 2);
  for (std::size_t byte = 0; byte < payload.size(); ++byte)
    payload[byte] = static_cast<std::byte>((byte * 79 + 0x81) & 255u);
  payload[0] = std::byte{0x81}; payload[1] = std::byte{0x7f};
  ASSERT_TRUE(input.view().write(payload).ok());
  const auto tensor = take(c::ConstTensorView::create(row_desc, input.view()), "source tensor");
  ASSERT_TRUE(invocation.bind(rows, tensor).ok());
  auto status = s::prepare(invocation, {}, {.host_workspace=staging.ref()});
  ASSERT_TRUE(status.ok()) << status.message();
  auto transaction = take(s::SlotBindings::create(layout, {}, destinations), "writers");
  status = s::bind(invocation, graph, store, std::move(transaction));
  ASSERT_TRUE(status.ok()) << status.message();
  status = invocation.prepare_replay({}); ASSERT_TRUE(status.ok()) << status.message();
  auto done = invocation.submit(); ASSERT_TRUE(done) << done.error().message();
  EXPECT_FALSE(s::take(invocation, store));
  ASSERT_TRUE(done->wait().ok()); *done = {};
  auto returned = s::take(invocation, store); ASSERT_TRUE(returned) << returned.error().message();
  auto expected = original;
  std::copy_n(payload.begin(), bytes, expected.begin()+4*stride);
  std::copy_n(payload.begin()+bytes, bytes, expected.begin()+stride);
  std::vector<std::byte> observed(original.size());
  ASSERT_TRUE(diagnostic.read(observed).ok()); EXPECT_EQ(observed, expected);
  // Replay retains the original range owners and re-records updated controls;
  // swapping writer order must not reuse the first episode's descriptor views.
  transaction = std::move(*returned);
  std::array reversed{std::move(transaction.destinations()[1]), std::move(transaction.destinations()[0])};
  transaction.clear(); ASSERT_TRUE(transaction.refill({}, reversed).ok());
  status = s::bind(invocation, graph, store, std::move(transaction));
  ASSERT_TRUE(status.ok()) << status.message();
  status = invocation.prepare_replay({}); ASSERT_TRUE(status.ok()) << status.message();
  done = invocation.submit(); ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok()); *done = {};
  returned = s::take(invocation, store); ASSERT_TRUE(returned) << returned.error().message();
  std::copy_n(payload.begin(), bytes, expected.begin()+stride);
  std::copy_n(payload.begin()+bytes, bytes, expected.begin()+4*stride);
  ASSERT_TRUE(diagnostic.read(observed).ok()); EXPECT_EQ(observed, expected);
  ASSERT_TRUE(untouched.valid());
  status = s::bind(invocation, graph, store, std::move(*returned));
  ASSERT_TRUE(status.ok()) << status.message();
  status = invocation.prepare_replay({}); ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_TRUE(invocation.reset().ok());
  EXPECT_FALSE(invocation.submit()); EXPECT_FALSE(invocation.prepare_replay({}).ok());
  returned = s::take(invocation, store); ASSERT_TRUE(returned) << returned.error().message();
  ASSERT_TRUE(diagnostic.read(observed).ok()); EXPECT_EQ(observed, expected);
}
TEST(VulkanSlotStore, TwoByteItemsUseExactStorageCopyAndKeepPadding) {
  exercise_halfword_store(1);
}
TEST(VulkanSlotStore, SixByteItemsUseExactStorageCopyAndKeepPadding) {
  exercise_halfword_store(3);
}
TEST(VulkanSlotStore, HalfwordCompletionOutlivesItsRecordingWrapper) {
  namespace vd = v::detail;
  auto context = take(v::Context::create(), "Vulkan Context");
  auto device = take(v::MemoryResource::create(context, v::MemoryKind::kDevice), "device");
  auto input = take(m::Buffer::create(device.ref(), 8, 4), "source");
  auto output = take(m::Buffer::create(device.ref(), 8, 4), "destination");
  const std::array payload{std::byte{0x80},std::byte{0x7f},std::byte{0x31},std::byte{0x42},
      std::byte{0x53},std::byte{0x64},std::byte{0x75},std::byte{0x86}};
  ASSERT_TRUE(input.view().write(payload).ok());
  ASSERT_TRUE(output.view().fill(std::byte{0xa5}).ok());
  auto recording = take(vd::prepare_copy_recording_runtime(vd::Access::runtime(context), 1,
      m::system_allocator(), 2, vd::CopyRecordingMode::kOnce, vd::CopyGranularity::kHalfword), "halfword recording");
  const std::array<c::detail::CopyBackendRange, 1> copies{{{
      take(output.view().subview(2, 2), "exact destination"),
      take(input.view().subview(0, 2), "exact source")}}};
  const auto status = vd::bind_copy_recording(*recording, copies);
  ASSERT_TRUE(status.ok()) << status.message();
  auto done = take(vd::submit_copy_recording(*recording), "actual copy submission");
  recording.reset();
  ASSERT_TRUE(done.wait().ok()); done = {};
  std::array<std::byte, 8> observed{}, expected;
  expected.fill(std::byte{0xa5}); expected[2]=payload[0]; expected[3]=payload[1];
  ASSERT_TRUE(output.view().read(observed).ok()); EXPECT_EQ(observed, expected);
}
} // namespace
