#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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

#include "../compute_models/row_cache_native_fixture.h"
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
namespace m = hyle::mem;
using row_cache_native_test::take;

constexpr unsigned queries = 3, selections = 3, heads = 64, depth = 512;
constexpr std::array<std::int32_t, 9> logical_ids{-1, 0, 2, 0, 1, 2, 0, 1, 2};

void check(hyle::Status status, const char *what) {
  if (!status.ok())
    throw std::runtime_error(std::string(what) + ": " + std::string(status.message()));
}
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, std::span(shape.begin(), shape.size())).value();
}
c::RowCacheBatchSchema schema() {
  return c::RowCacheBatchSchema::create({
      .encoding = c::RowCacheEncoding::float8_block64_tail64(),
      .max_requests = 2, .max_query_rows = queries, .max_kv_rows = 8,
      .max_pages = 4, .max_page_refs = 4, .page_size = 2}).value();
}
c::RowCacheLayoutVersion layout(const c::RowCacheBatchSchema &owned, unsigned phase,
                                bool empty = false) {
  const std::array<std::int32_t, 3> q = phase ? std::array<std::int32_t, 3>{0, 1, 3}
                                                   : std::array<std::int32_t, 3>{0, 2, 3};
  const std::array<std::int32_t, 3> kv = empty ? std::array<std::int32_t, 3>{}
                                                    : std::array<std::int32_t, 3>{0, 2, 3};
  const std::array<std::int32_t, 3> pages = phase ? std::array<std::int32_t, 3>{1, 3, 0}
                                                       : std::array<std::int32_t, 3>{3, 1, 0};
  const std::array<std::int32_t, 2> tails = empty ? std::array<std::int32_t, 2>{}
                                                       : std::array<std::int32_t, 2>{1, 1};
  return owned.version({q, kv, empty ? std::span<const std::int32_t>{} : std::span(pages), tails}).value();
}
std::array<std::int32_t, 6> ranges(unsigned phase) {
  return phase ? std::array<std::int32_t, 6>{0, 3, 0, 1, 0, 1}
               : std::array<std::int32_t, 6>{0, 0, 0, 3, 0, 1};
}

// All uploads use the original Vulkan F2 recording. The bank and metadata
// remain real device allocations, rather than host-visible substitutes.
struct Route final {
  v::Context context = v::Context::create().value();
  v::MemoryResource device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  v::MemoryResource staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  c::LeasePool host = c::LeasePool::create(staging.ref(), 1u << 20).value();
  c::LeasePool pool = c::LeasePool::create(device.ref(), 2u << 20).value();
  v::CopyBatchProgram copy_program = v::CopyBatchProgram::prepare(context,
      std::array{c::CopyResourceSpec{staging.ref(), 1u << 20},
                 c::CopyResourceSpec{device.ref(), 1u << 20}}).value();
  v::CopyBatchInvocation copy = v::CopyBatchInvocation::create(copy_program).value();

  c::WriteLease upload(c::WriteLease destination, std::span<const std::byte> bytes) {
    auto writer = host.allocate(bytes.size()).value();
    check(writer.view()->write(bytes), "write upload staging");
    auto source = writer.publish().value();
    check(copy.reset(), "reset upload");
    check(copy.bind_source({0}, source), "bind upload source");
    check(copy.bind_mutable({1}, std::move(destination)), "bind upload destination");
    const std::array<c::CopyPhase, 1> phases{{{}}};
    const std::array<c::CopyRange, 1> copies{{{{0}, {1}, {0}, 0, 0, bytes.size()}}};
    check(copy.prepare_replay(phases, copies), "prepare original upload");
    { auto done = take(copy.submit(), "submit original upload"); check(done.wait(), "wait original upload"); }
    auto result = take(copy.unbind_mutable({1}), "return upload writer");
    check(copy.reset(), "release original upload");
    return result;
  }
  c::ReadLease load(std::span<const std::byte> bytes) {
    return upload(pool.allocate(bytes.size()).value(), bytes).publish().value();
  }
};

struct Recipe final {
  g::Graph graph;
  gb::PagedRowCacheGatherPadded rows;
  g::ValueId ids, visibility, query, sinks, attention, negative;
};
Recipe recipe(const c::RowCacheBatchSchema &owned, unsigned invalid = 0) {
  auto builder = g::GraphBuilder::create().value();
  Recipe result;
  result.ids = builder.input(dense(c::DataType::kInt32, {queries, selections})).value();
  result.visibility = builder.input(dense(c::DataType::kInt32, {queries, 2})).value();
  result.query = builder.input(dense(c::DataType::kFloat32, {queries, 1, heads, depth})).value();
  result.sinks = builder.input(dense(c::DataType::kFloat32, {heads})).value();
  result.rows = gb::add_row_cache_gather_padded(builder, owned, result.ids,
                                              result.visibility, 4).value();
  const auto reshape = [&](g::ValueId value, std::initializer_list<std::int64_t> shape) {
    return builder.add(c::View::reshape(c::TensorShape::create(
        std::span(shape.begin(), shape.size())).value()).value(), value).value();
  };
  const auto wide = builder.add(c::Cast{c::DataType::kFloat32}, result.rows.output()).value();
  const auto kv = reshape(wide, {queries, selections, 1, depth});
  const auto mask = reshape(result.rows.mask(), {queries, 1, selections});
  result.attention = builder.result(builder.add_node(c::model_ops::sink_attention(.0625F),
      std::array{result.query, kv, kv, mask, result.sinks}).value()).value();
  result.negative = builder.add(c::Unary{c::UnaryKind::kNeg}, result.attention).value();
  for (auto output : {result.rows.output(), result.rows.mask(), result.attention, result.negative})
    check(builder.publish(output), "publish ordinary result");
  if (invalid == 1) {
    // This is a real second reader, not another declared checked decoder.
    const auto unqualified = builder.add(c::Copy{}, result.rows.bank_input()).value();
    check(builder.publish(unqualified), "publish unqualified capacity reader");
  } else if (invalid == 2) {
    // Only the decoded rows and mask are public. The private physical IDs
    // cannot escape merely because their native kernel supports I32 output.
    const auto map = gb::detail::RowCacheAccess::map(result.rows.reader());
    check(builder.publish(builder.result(map).value()), "publish private mapped IDs");
  }
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
template<class T>std::vector<T> read(Route &route, const g::GraphOutput &output) {
  const auto description = output.description().value();
  auto tensor = route.device.binding(output.storage_lease().value(), description).value();
  std::vector<T> result(description.element_count().value());
  check(route.context.download(tensor, std::as_writable_bytes(std::span(result))), "download Graph result");
  return result;
}
row_cache_test::Reference data() {
  const auto encoding = c::RowCacheEncoding::float8_block64_tail64();
  row_cache_test::Reference result;
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
Expected expected(const row_cache_test::Reference &bank, unsigned phase, bool empty = false) {
  Expected result{std::vector<std::uint16_t>(queries * selections * depth),
                  std::vector<float>(queries * selections, -std::numeric_limits<float>::infinity())};
  const auto visible = empty ? std::array<std::int32_t, 6>{} : ranges(phase);
  for (unsigned index = 0; index < logical_ids.size(); ++index) {
    const auto query = index / selections;
    const auto id = logical_ids[index];
    if (id < visible[query * 2] || id >= visible[query * 2 + 1]) continue;
    const bool first = query < (phase ? 1u : 2u);
    const auto page = first ? (id < 2 ? (phase ? 1u : 3u) : (phase ? 3u : 1u)) : 0u;
    const auto physical = page * 2 + static_cast<unsigned>(id % 2);
    std::copy_n(bank.decoded.begin() + physical * depth, depth, result.rows.begin() + index * depth);
    result.mask[index] = 0.F;
  }
  return result;
}
void verify(Route &route, g::Invocation &slot, const Recipe &model, const Expected &wanted) {
  EXPECT_EQ(read<std::uint16_t>(route, slot.result(model.rows.output()).value()), wanted.rows);
  EXPECT_EQ(read<float>(route, slot.result(model.rows.mask()).value()), wanted.mask);
  const auto actual = read<float>(route, slot.result(model.attention).value());
  const auto negative = read<float>(route, slot.result(model.negative).value());
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
        // Existing native padded->Sink FP64 budget, not fitted to this run.
        ASSERT_NEAR(actual[index], value, 2e-5 + 2e-5 * std::abs(value)) << index;
        ASSERT_EQ(negative[index], -actual[index]) << index;
      }
    }
  }
}
void bind_inputs(g::Invocation &slot, const Recipe &model,
                 c::ReadLease ids, c::ReadLease visibility, c::ReadLease query, c::ReadLease sinks) {
  check(slot.bind(model.ids, model.graph.value_description(model.ids).value(), std::move(ids)), "bind logical IDs");
  check(slot.bind(model.visibility, model.graph.value_description(model.visibility).value(), std::move(visibility)), "bind visibility");
  check(slot.bind(model.query, model.graph.value_description(model.query).value(), std::move(query)), "bind Sink query");
  check(slot.bind(model.sinks, model.graph.value_description(model.sinks).value(), std::move(sinks)), "bind Sink denominator");
}

TEST(VulkanProjectedRowGraph, ReadyPermutedPagesKeepTwoLayoutsSlotsAndOldResults) {
  Route route;
  RecordProperty("device", route.context.environment().device_name);
  ASSERT_FALSE(route.context.environment().software_device);
  const auto owned = schema();
  const auto model = recipe(owned);
  const auto reference = data();
  auto arena = c::PagedArena::create(owned, route.device.ref(),
      {.max_backing_bytes = 8 * owned.capacity().encoding.row_bytes() + 1}).value();
  std::array<c::ReadLease, 4> pages;
  for (unsigned page : {0u, 1u, 3u}) {
    auto writer = arena.acquire(page).value();
    writer.lease() = route.upload(std::move(writer.lease()), std::as_bytes(std::span(reference.bytes)
        .subspan(page * arena.page_bytes(), arena.page_bytes())));
    pages[page] = writer.publish(2).value();
  }
  // A nonempty read must not turn the private capacity into a whole-arena read.
  auto unused_writer = arena.acquire(2).value();
  const auto ids = route.load(std::as_bytes(std::span(logical_ids)));
  const std::vector<float> zero(queries * heads * depth), sinks(heads, .25F);
  const auto query = route.load(std::as_bytes(std::span(zero)));
  const auto denominator = route.load(std::as_bytes(std::span(sinks)));
  for (bool split : {false, true}) {
    SCOPED_TRACE(split ? "actual split artifacts" : "ordinary fusion");
    auto compiled = compile(route, model, split);
    const auto map_step = selected_step(compiled.report, gb::detail::RowCacheAccess::map(model.rows.reader()));
    const auto decode_step = selected_step(compiled.report, model.rows.node());
    if (split) {
      ASSERT_NE(map_step, decode_step);
      EXPECT_EQ(compiled.report.step(map_step)->target, g::ExecutionTargetId{0});
      EXPECT_EQ(compiled.report.step(decode_step)->target, g::ExecutionTargetId{1});
    } else {
      ASSERT_EQ(map_step, decode_step);
    }
    std::array slots{compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value(),
                     compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value()};
    g::GraphOutput old;
    c::ReadLease escaped;
    std::vector<std::uint16_t> old_bits;
    for (unsigned phase : {0u, 1u}) {
      auto &slot = slots[phase];
      const auto version = layout(owned, phase);
      const std::array versions{gb::RowCacheBinding{model.rows.reader(), version}};
      auto prepared = gb::prepare(slot, model.graph, {}, {}, versions,
                                  {.host_workspace = route.staging.ref()});
      ASSERT_TRUE(prepared.ok()) << prepared.message();
      ASSERT_TRUE(gb::bind(slot, model.rows.bank_input(),
          arena.read(version, std::array{pages[0], pages[3], pages[1]}).value()).ok());
      const auto visible = ranges(phase);
      bind_inputs(slot, model, ids, route.load(std::as_bytes(std::span(visible))), query, denominator);
      ASSERT_TRUE(slot.prepare_replay({}).ok());
      { auto done = take(slot.submit(), "ordinary ready Graph submit"); ASSERT_TRUE(done.wait().ok()); }
      const auto wanted = expected(reference, phase);
      verify(route, slot, model, wanted);
      ASSERT_TRUE(unused_writer.valid());
      if (!phase) {
        old = slot.result(model.rows.output()).value();
        escaped = old.storage_lease().value();
        old_bits = wanted.rows;
        EXPECT_TRUE(gb::prepare(slot, model.graph, {}, {}, versions,
            {.host_workspace = route.staging.ref()}).IsUnavailable());
      } else {
        EXPECT_EQ(read<std::uint16_t>(route, old), old_bits);
        EXPECT_NE(wanted.rows, old_bits);
      }
    }
    EXPECT_FALSE(slots[0].submit());
    old.reset();
    const auto first_layout = layout(owned, 0);
    const std::array first_versions{gb::RowCacheBinding{model.rows.reader(), first_layout}};
    // A cold reuse check reaches the actual native owner before any split
    // artifact can submit a new prefix. A ReadLease alone still protects it.
    EXPECT_TRUE(gb::prepare(slots[0], model.graph, {}, {}, first_versions,
        {.host_workspace = route.staging.ref()}).IsUnavailable());
    {
      const auto description = model.graph.value_description(model.rows.output()).value();
      const auto retained = route.device.binding(escaped, description).value();
      std::vector<std::uint16_t> values(old_bits.size());
      ASSERT_TRUE(route.context.download(retained, std::as_writable_bytes(std::span(values))).ok());
      EXPECT_EQ(values, old_bits);
    }
    escaped.reset();
    // Reuse the same device mirror with a different request grouping and page
    // order. A stale metadata upload or stale native layout cannot pass this.
    const auto changed_layout = layout(owned, 1);
    const std::array changed_versions{gb::RowCacheBinding{model.rows.reader(), changed_layout}};
    const auto prepared = gb::prepare(slots[0], model.graph, {}, {}, changed_versions,
                                       {.host_workspace = route.staging.ref()});
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    ASSERT_TRUE(gb::bind(slots[0], model.rows.bank_input(),
        arena.read(changed_layout, std::array{pages[1], pages[0], pages[3]}).value()).ok());
    const auto visible = ranges(1);
    bind_inputs(slots[0], model, ids, route.load(std::as_bytes(std::span(visible))), query, denominator);
    ASSERT_TRUE(slots[0].prepare_replay({}).ok());
    { auto replay = take(slots[0].submit(), "original slot replay after observers release"); ASSERT_TRUE(replay.wait().ok()); }
    verify(route, slots[0], model, expected(reference, 1));
  }
}

TEST(VulkanProjectedRowGraph, TrueEmptyHistoryRunsWhileEveryPageWriterIsHeld) {
  Route route;
  const auto owned = schema();
  const auto model = recipe(owned);
  const auto compiled = compile(route, model);
  auto slot = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  auto arena = c::PagedArena::create(owned, route.device.ref(),
      {.max_backing_bytes = 8 * owned.capacity().encoding.row_bytes() + 1}).value();
  std::array<c::PagedPageWriter, 4> writers;
  for (unsigned page = 0; page < writers.size(); ++page) writers[page] = arena.acquire(page).value();
  const auto version = layout(owned, 0, true);
  const std::array versions{gb::RowCacheBinding{model.rows.reader(), version}};
  auto prepared = gb::prepare(slot, model.graph, {}, {}, versions,
                              {.host_workspace = route.staging.ref()});
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  ASSERT_TRUE(gb::bind(slot, model.rows.bank_input(), arena.read(version, {}).value()).ok());
  const std::array<std::int32_t, 6> visibility{};
  const std::vector<float> zero(queries * heads * depth), sinks(heads, .25F);
  bind_inputs(slot, model, route.load(std::as_bytes(std::span(logical_ids))),
      route.load(std::as_bytes(std::span(visibility))), route.load(std::as_bytes(std::span(zero))),
      route.load(std::as_bytes(std::span(sinks))));
  ASSERT_TRUE(slot.prepare_replay({}).ok());
  { auto done = take(slot.submit(), "true-empty projected Graph"); ASSERT_TRUE(done.wait().ok()); }
  verify(route, slot, model, expected({}, 0, true));
  for (const auto &writer : writers) EXPECT_TRUE(writer.valid());
}

TEST(VulkanProjectedRowGraph, ExtraBankReaderAndPublishedPhysicalIdsAreRejected) {
  Route route;
  const auto owned = schema();
  const auto version = layout(owned, 0);
  for (unsigned invalid : {1u, 2u}) {
    SCOPED_TRACE(invalid == 1 ? "unqualified bank Copy" : "private mapped IDs published");
    const auto model = recipe(owned, invalid);
    const auto compiled = compile(route, model);
    auto slot = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
    const std::array versions{gb::RowCacheBinding{model.rows.reader(), version}};
    const auto denied = gb::prepare(slot, model.graph, {}, {}, versions,
                                    {.host_workspace = route.staging.ref()});
    EXPECT_TRUE(denied.IsInvalidArgument()) << denied.message();
    EXPECT_FALSE(slot.submit());
    EXPECT_FALSE(slot.result(model.rows.output()));
  }
}
} // namespace
