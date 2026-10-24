#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../compute_batch/native_vulkan_failure_fixture.h"
#include "../compute_models/row_cache_native_fixture.h"
#include "../../src/compute_route_vulkan/vulkan_file_stream_access.h"
#include "../../src/graph_batch/row_cache_node.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/vulkan_memory.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/result.h"
#include "hyle/graph/vulkan.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace gb = g::batch;
using row_cache_native_test::take;

constexpr unsigned queries = 3, selections = 3, heads = 64;
constexpr std::array<std::int32_t, 9> logical_ids{-1, 0, 2, 0, 1, 2, 0, 1, 2};
constexpr std::array<std::int32_t, 6> visibility{0, 0, 0, 3, 0, 1};

void check(hyle::Status status, const char *what) {
  if (!status.ok())
    throw std::runtime_error(std::string(what) + ": " + std::string(status.message()));
}
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, std::span(shape.begin(), shape.size())).value();
}
c::RowCacheBatchSchema schema(c::RowCacheEncoding encoding) {
  return c::RowCacheBatchSchema::create({
      .encoding = encoding, .max_requests = 2, .max_query_rows = queries,
      .max_kv_rows = 8, .max_pages = 4, .max_page_refs = 4, .page_size = 2}).value();
}
c::RowCacheLayoutVersion layout(const c::RowCacheBatchSchema &owned) {
  const std::array<std::int32_t, 3> q{0, 2, 3}, kv{0, 2, 3}, pages{3, 1, 0};
  const std::array<std::int32_t, 2> tails{1, 1};
  return owned.version({q, kv, pages, tails}).value();
}

// The same original F2 uploads used by the ready row Graph fixture. Both the
// physical bank and its Graph-prepared metadata remain real device storage.
struct Route final {
  v::Context context = v::Context::create().value();
  v::MemoryResource device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  v::MemoryResource staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  c::LeasePool host = c::LeasePool::create(staging.ref(), 1u << 20).value();
  c::LeasePool pool = c::LeasePool::create(device.ref(), 2u << 20).value();
  v::CopyBatchProgram upload_program = v::CopyBatchProgram::prepare(context,
      std::array{c::CopyResourceSpec{staging.ref(), 1u << 20},
                 c::CopyResourceSpec{device.ref(), 1u << 20}}).value();
  v::CopyBatchInvocation upload_copy = v::CopyBatchInvocation::create(upload_program).value();

  c::WriteLease upload(c::WriteLease destination, std::span<const std::byte> bytes) {
    auto writer = host.allocate(bytes.size()).value();
    check(writer.view()->write(bytes), "write upload staging");
    auto source = writer.publish().value();
    check(upload_copy.reset(), "reset original upload");
    check(upload_copy.bind_source({0}, source), "bind upload source");
    check(upload_copy.bind_mutable({1}, std::move(destination)), "bind upload destination");
    const std::array<c::CopyPhase, 1> phases{{{}}};
    const std::array<c::CopyRange, 1> copies{{{{0}, {1}, {0}, 0, 0, bytes.size()}}};
    check(upload_copy.prepare_replay(phases, copies), "prepare original upload");
    { auto done = take(upload_copy.submit(), "submit original upload"); check(done.wait(), "wait original upload"); }
    auto result = take(upload_copy.unbind_mutable({1}), "return upload writer");
    check(upload_copy.reset(), "release original upload");
    return result;
  }
  c::ReadLease load(std::span<const std::byte> bytes) {
    return upload(pool.allocate(bytes.size()).value(), bytes).publish().value();
  }
};

struct Recipe final {
  g::Graph graph;
  gb::PagedRowCacheGatherPadded rows;
  g::ValueId ids, ranges, query, sinks, attention, negative;
};
Recipe recipe(const c::RowCacheBatchSchema &owned) {
  auto builder = g::GraphBuilder::create().value();
  const auto depth = owned.capacity().encoding.depth();
  Recipe result;
  result.ids = builder.input(dense(c::DataType::kInt32, {queries, selections})).value();
  result.ranges = builder.input(dense(c::DataType::kInt32, {queries, 2})).value();
  if (depth == 512) {
    result.query = builder.input(dense(c::DataType::kFloat32, {queries, 1, heads, depth})).value();
    result.sinks = builder.input(dense(c::DataType::kFloat32, {heads})).value();
  }
  result.rows = gb::add_row_cache_gather_padded(builder, owned, result.ids, result.ranges, 4).value();
  const auto reshape = [&](g::ValueId value, std::initializer_list<std::int64_t> shape) {
    return builder.add(c::View::reshape(c::TensorShape::create(
        std::span(shape.begin(), shape.size())).value()).value(), value).value();
  };
  const auto wide = builder.add(c::Cast{c::DataType::kFloat32}, result.rows.output()).value();
  // FP4 is the actual D128 index bank. The model Sink contract is D512;
  // keep its row/mask consumer in its own supported mathematical domain.
  if (depth == 512) {
    const auto kv = reshape(wide, {queries, selections, 1, depth});
    const auto mask = reshape(result.rows.mask(), {queries, 1, selections});
    result.attention = take(builder.result(take(builder.add_node(c::model_ops::sink_attention(.0625F),
        std::array{result.query, kv, kv, mask, result.sinks}), "FP8 Sink node")), "FP8 Sink result");
  } else {
    result.attention = wide;
  }
  result.negative = builder.add(c::Unary{c::UnaryKind::kNeg}, result.attention).value();
  for (auto output : {result.rows.output(), result.rows.mask(), result.attention, result.negative})
    check(builder.publish(output), "publish original row Graph result");
  result.graph = std::move(builder).finish().value();
  return result;
}
g::CompiledGraph compile(Route &route, const Recipe &model, bool split = false) {
  const auto source = g::vulkan::execution_source(route.context).value();
  const std::array domains{g::StorageDomain{route.device.ref()}, g::StorageDomain{route.staging.ref()}};
  if (!split) {
    const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
    return take(gb::compile(model.graph, {.storage_domains = domains, .targets = targets}),
                "ordinary projected row Graph");
  }
  // Two required placements split only the selected artifacts. Both sources
  // still use the one original Context/Runtime, kernels and completion owner.
  const g::vulkan::CustomLowering configuration = [](const c::Operation &,
      std::span<const c::TensorDesc>, const c::TensorDesc &)
      -> hyle::Result<g::vulkan::CustomDispatch> {
    return std::unexpected(hyle::Status::Unimplemented(
        "projected row test uses only original native mathematics"));
  };
  const auto other = g::vulkan::execution_source(route.context,
      {.custom = configuration}).value();
  EXPECT_FALSE(source.same_owner(other));
  EXPECT_TRUE(source.same_device(other));
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}},
                           g::ExecutionTarget{other, g::StorageDomainId{0}}};
  const std::array placements{
      g::NodePlacement{gb::detail::RowCacheAccess::map(model.rows.reader()),
                       g::ExecutionTargetId{0}, g::PlacementStrength::kRequired},
      g::NodePlacement{model.rows.node(), g::ExecutionTargetId{1}, g::PlacementStrength::kRequired}};
  return take(gb::compile(model.graph, {.storage_domains = domains, .targets = targets},
                         {.node_placements = placements}), "split projected row Graph");
}
std::size_t selected_step(const g::CompileReport &report, g::NodeId node) {
  for (std::size_t step = 0; step < report.step_count(); ++step) {
    const auto facts = report.step(step).value();
    for (std::size_t index = 0; index < facts.node_count; ++index)
      if (report.step_node(step, index).value() == node) return step;
  }
  throw std::runtime_error("selected row node is absent from the actual compile report");
}

row_cache_test::Reference data(c::RowCacheEncoding encoding) {
  row_cache_test::Reference result;
  const auto depth = encoding.depth();
  for (unsigned phase : {0u, 1u}) {
    std::vector<std::uint16_t> values(5 * depth);
    for (unsigned row = 0; row < 5; ++row)
      for (unsigned channel = 0; channel < depth; ++channel)
        values[row * depth + channel] = row_cache_test::bfloat(
            (int((row * 7 + channel * 3 + phase) % 17) - 8) / 8.);
    const auto encoded = row_cache_test::reference(encoding, values);
    result.bytes.insert(result.bytes.end(), encoded.bytes.begin(), encoded.bytes.end());
    result.decoded.insert(result.decoded.end(), encoded.decoded.begin(), encoded.decoded.end());
  }
  return result;
}
struct Expected final { std::vector<std::uint16_t> rows; std::vector<float> mask; };
Expected expected(const row_cache_test::Reference &bank, unsigned depth) {
  Expected result{std::vector<std::uint16_t>(queries * selections * depth),
                  std::vector<float>(queries * selections, -std::numeric_limits<float>::infinity())};
  for (unsigned index = 0; index < logical_ids.size(); ++index) {
    const auto query = index / selections;
    const auto id = logical_ids[index];
    if (id < visibility[query * 2] || id >= visibility[query * 2 + 1]) continue;
    const auto page = query < 2 ? (id < 2 ? 3u : 1u) : 0u;
    const auto physical = page * 2 + static_cast<unsigned>(id % 2);
    std::copy_n(bank.decoded.begin() + physical * depth, depth, result.rows.begin() + index * depth);
    result.mask[index] = 0.F;
  }
  return result;
}
template<class T>std::vector<T> read(Route &route, const g::GraphOutput &output) {
  const auto description = output.description().value();
  auto tensor = route.device.binding(output.storage_lease().value(), description).value();
  std::vector<T> result(description.element_count().value());
  check(route.context.download(tensor, std::as_writable_bytes(std::span(result))), "download original Graph result");
  return result;
}
void verify(Route &route, g::Invocation &slot, const Recipe &model,
            const Expected &wanted, unsigned depth) {
  EXPECT_EQ(read<std::uint16_t>(route, slot.result(model.rows.output()).value()), wanted.rows);
  EXPECT_EQ(read<float>(route, slot.result(model.rows.mask()).value()), wanted.mask);
  const auto actual = read<float>(route, slot.result(model.attention).value());
  const auto negative = read<float>(route, slot.result(model.negative).value());
  if (depth != 512) {
    ASSERT_EQ(actual.size(), wanted.rows.size());
    ASSERT_EQ(negative.size(), actual.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(actual[i], row_cache_test::decode(wanted.rows[i])) << i;
      EXPECT_EQ(negative[i], -actual[i]) << i;
    }
    return;
  }
  ASSERT_EQ(actual.size(), queries * heads * depth);
  ASSERT_EQ(negative.size(), actual.size());
  for (unsigned query = 0; query < queries; ++query) {
    double denominator = std::exp(.25);
    for (unsigned key = 0; key < selections; ++key)
      if (wanted.mask[query * selections + key] == 0) denominator += 1;
    for (unsigned channel = 0; channel < depth; ++channel) {
      double value = 0;
      for (unsigned key = 0; key < selections; ++key)
        if (wanted.mask[query * selections + key] == 0)
          value += row_cache_test::decode(wanted.rows[(query * selections + key) * depth + channel]);
      value /= denominator;
      for (unsigned head = 0; head < heads; ++head) {
        const auto index = (query * heads + head) * depth + channel;
        // The ready native padded->Sink fixture's independent FP64 budget.
        ASSERT_NEAR(actual[index], value, 2e-5 + 2e-5 * std::abs(value)) << index;
        ASSERT_EQ(negative[index], -actual[index]) << index;
      }
    }
  }
}

// Reuse the native pending risk fixture's retirement-observation boundary.
// Every observation still queries the actual original fence. Holding its
// successful observation at NOT_READY proves Completion/permission lifetime;
// it does not claim that the GPU has not already executed those commands.
struct RetirementObservationGate final {
  hyle_f5_vk_failure_fixture::Boundary boundary;
  static inline std::atomic<bool> held{false};
  explicit RetirementObservationGate(const v::Context &context) : boundary(context) {
    held = true;
    boundary.device.fn.GetFenceStatus = [](VkDevice device, VkFence fence) -> VkResult {
      auto &hook = *hyle_f5_vk_failure_fixture::Boundary::active;
      ++hook.polls;
      const auto actual = hook.original.GetFenceStatus(device, fence);
      return held.load() && (actual == VK_SUCCESS || actual == VK_NOT_READY) ? VK_NOT_READY : actual;
    };
  }
  void release() { held = false; }
  ~RetirementObservationGate() { release(); }
};

void exercise(c::RowCacheEncoding encoding, bool split = false) {
  SCOPED_TRACE(split ? "split artifacts" : "ordinary fusion");
  SCOPED_TRACE(encoding.depth());
  Route route;
  ::testing::Test::RecordProperty("device", route.context.environment().device_name);
  ASSERT_FALSE(route.context.environment().software_device);
  const auto owned = schema(encoding);
  const auto version = layout(owned);
  const auto model = recipe(owned);
  const auto compiled = compile(route, model, split);
  const auto map_step = selected_step(compiled.report, gb::detail::RowCacheAccess::map(model.rows.reader()));
  const auto decoder_step = selected_step(compiled.report, model.rows.node());
  if (split) EXPECT_NE(map_step, decoder_step);
  else EXPECT_EQ(map_step, decoder_step);
  auto slot = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  auto arena = c::PagedArena::create(owned, route.device.ref(),
      {.max_backing_bytes = 8 * encoding.row_bytes() + 1}).value();
  const auto original = data(encoding);
  std::array<c::ReadLease, 2> ready;
  // The layout references physical pages 3,1,0 in that order; these ready
  // permissions deliberately arrive in another order, each as a one-row tail.
  for (unsigned page : {0u, 1u}) {
    auto writer = arena.acquire(page).value();
    writer.lease() = route.upload(std::move(writer.lease()), std::as_bytes(std::span(original.bytes)
        .subspan(page * arena.page_bytes(), arena.page_bytes())));
    ready[page] = writer.publish(1).value();
  }
  auto unused_writer = arena.acquire(2).value();
  auto page = arena.acquire(3).value();
  const auto row_bytes = encoding.row_bytes(), copy_bytes = 2 * row_bytes;
  auto changed = original;
  std::copy_n(original.bytes.begin(), copy_bytes, changed.bytes.begin() + 3 * arena.page_bytes());
  std::copy_n(original.decoded.begin(), 2 * encoding.depth(), changed.decoded.begin() + 6 * encoding.depth());
  const auto wanted = expected(changed, encoding.depth());
  ASSERT_NE(wanted.rows, expected(original, encoding.depth()).rows);

  const auto ids = route.load(std::as_bytes(std::span(logical_ids)));
  const auto ranges = route.load(std::as_bytes(std::span(visibility)));
  const std::vector<float> zero(queries * heads * encoding.depth()), sinks(heads, .25F);
  const auto query = route.load(std::as_bytes(std::span(zero)));
  const auto denominator = route.load(std::as_bytes(std::span(sinks)));
  const std::array versions{gb::RowCacheBinding{model.rows.reader(), version}};
  const auto prepare = [&] {
    return gb::prepare(slot, model.graph, {}, {}, versions,
        {.host_workspace = route.staging.ref(), .max_source_completions = 2});
  };
  auto prepared = prepare();
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  for (const auto &[port, input] : std::array{
           std::pair{model.ids, ids}, std::pair{model.ranges, ranges},
           std::pair{model.query, query}, std::pair{model.sinks, denominator}}) {
    if (!port.valid()) continue;
    ASSERT_TRUE(slot.bind(port, model.graph.value_description(port).value(), input).ok());
  }

  auto host_writer = route.host.allocate(copy_bytes).value();
  ASSERT_TRUE(host_writer.view()->write(std::as_bytes(std::span(original.bytes)).first(copy_bytes)).ok());
  const auto payload = host_writer.publish().value();
  const std::array specs{c::CopyResourceSpec{route.staging.ref(), copy_bytes},
                         c::CopyResourceSpec{route.device.ref(), arena.page_bytes()}};
  const auto program = v::CopyBatchProgram::prepare(route.context, specs).value();
  auto copy = v::CopyBatchInvocation::create(program).value();
  ASSERT_TRUE(copy.bind_source({0}, payload).ok());
  ASSERT_TRUE(copy.bind_mutable({1}, std::move(page.lease())).ok());
  const std::array<c::CopyPhase, 2> phases{{{}, {.dependency_mask = 1}}};
  const std::array<c::CopyRange, 1> full{{{{0}, {1}, {1}, 0, 0, copy_bytes}}};
  const std::array<c::PagedPageUpdate, 1> updates{{{&page, {1}}}};
  auto frame = take(arena.pending_read(version, ready, updates), "pending projected two-row page");
  const auto bound = gb::bind(slot, model.rows.bank_input(), frame);
  ASSERT_TRUE(bound.ok()) << bound.message();
  ASSERT_TRUE(slot.prepare_replay({},
      {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());

  // This first episode is intentionally completed before admission. One
  // initialized row cannot authorize the selected second row of physical page3.
  auto short_copy = full;
  short_copy[0].bytes = row_bytes; // Vulkan's original four-byte copy alignment.
  ASSERT_TRUE(copy.prepare_replay(phases, short_copy).ok());
  auto partial = take(copy.submit(), "original short native page copy");
  ASSERT_TRUE(partial.wait().ok());
  auto update = take(v::CopyBatchUpdate::take(std::move(copy), std::move(partial)), "original short Vulkan Update");
  const auto rejected = g::vulkan::submit(slot, {}, std::move(update));
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsOutOfRange()) << rejected.error().message();
  ASSERT_TRUE(update.valid());
  EXPECT_FALSE(slot.result(model.rows.output()));
  EXPECT_FALSE(slot.result(model.rows.mask()));
  EXPECT_FALSE(slot.result(model.attention));
  EXPECT_FALSE(g::vulkan::take_update(slot));
  EXPECT_FALSE(arena.acquire(3));
  ASSERT_TRUE(unused_writer.valid());
  page.lease() = take(update.take_mutable({1}), "return rejected original page writer");
  copy = take(update.release_invocation(), "return rejected original copy invocation");
  ASSERT_TRUE(copy.reset().ok());
  ASSERT_TRUE(copy.bind_source({0}, payload).ok());
  ASSERT_TRUE(copy.bind_mutable({1}, std::move(page.lease())).ok());
  ASSERT_TRUE(copy.prepare_replay(phases, full).ok());

  // All cold native compilation, metadata copies and uploads precede the gate.
  RetirementObservationGate gate(route.context);
  auto copied = take(copy.submit(), "original full native page copy");
  update = take(v::CopyBatchUpdate::take(std::move(copy), std::move(copied)), "original pending Vulkan Update");
  std::array<c::Completion *, 2> original_sources{};
  const auto source_count = v::detail::CopyStreamAccess::retirement_sources(update, original_sources);
  ASSERT_TRUE(source_count) << source_count.error().message();
  ASSERT_EQ(*source_count, original_sources.size());
  ASSERT_NE(original_sources.back(), nullptr);
  const auto source_poll = original_sources.back()->poll();
  ASSERT_TRUE(source_poll) << source_poll.error().message();
  ASSERT_EQ(*source_poll, c::CompletionPoll::kPending);
  EXPECT_GT(gate.boundary.accepted, 0u);
  EXPECT_GT(gate.boundary.polls.load(), 0u);
  EXPECT_TRUE(update.in_flight());

  auto done = g::vulkan::submit(slot, {}, std::move(update));
  ASSERT_TRUE(done) << done.error().message();
  EXPECT_FALSE(update.valid());
  const auto pending = done->poll();
  ASSERT_TRUE(pending) << pending.error().message();
  EXPECT_EQ(*pending, c::CompletionPoll::kPending);
  EXPECT_FALSE(slot.result(model.rows.output()));
  EXPECT_FALSE(slot.result(model.rows.mask()));
  EXPECT_FALSE(slot.result(model.attention));
  EXPECT_FALSE(slot.result(model.negative));
  EXPECT_FALSE(g::vulkan::take_update(slot));
  EXPECT_FALSE(arena.acquire(3));
  EXPECT_TRUE(unused_writer.valid());
  EXPECT_TRUE(prepare().IsUnavailable());

  gate.release();
  ASSERT_TRUE(done->wait().ok()); // The original native fence now proves retirement.
  verify(route, slot, model, wanted, encoding.depth());
  auto output = slot.result(model.rows.output()).value();
  auto old = output;
  *done = c::Completion{};
  EXPECT_FALSE(g::vulkan::take_update(slot));
  output.reset();
  EXPECT_FALSE(g::vulkan::take_update(slot));
  EXPECT_EQ(read<std::uint16_t>(route, old), wanted.rows);
  EXPECT_FALSE(arena.acquire(3));
  EXPECT_TRUE(unused_writer.valid());
  old.reset();

  update = take(g::vulkan::take_update(slot), "recover original native page Update");
  EXPECT_FALSE(update.in_flight());
  // Recovering source controls does not turn the consumed frame into ready
  // permission or authorize an ordinary replay without the original Update.
  EXPECT_FALSE(slot.submit({}));
  EXPECT_TRUE(slot.prepare_replay({}).IsFailedPrecondition());
  page.lease() = take(update.take_mutable({1}), "recover original initialized page writer");
  auto published = take(page.publish(2), "publish exact two-row initialized page");
  const std::array complete{published, ready[1], ready[0]};
  EXPECT_TRUE(arena.read(version, complete));
  copy = take(update.release_invocation(), "recover original Vulkan copy invocation");
  ASSERT_TRUE(copy.reset().ok());
  frame = {};
  EXPECT_TRUE(unused_writer.valid());
  const auto reused = prepare();
  EXPECT_TRUE(reused.ok()) << reused.message();
}

TEST(VulkanProjectedRowGraph, PendingNativeFp8CopyKeepsPageSourceAndOldResults) {
  exercise(c::RowCacheEncoding::float8_block64_tail64());
}
TEST(VulkanProjectedRowGraph, PendingNativeFp8SplitArtifactsKeepExactSource) {
  exercise(c::RowCacheEncoding::float8_block64_tail64(), true);
}
TEST(VulkanProjectedRowGraph, PendingNativeFp4CopyKeepsPageSourceAndOldResults) {
  exercise(c::RowCacheEncoding::float4_block32());
}
} // namespace
