#include <gtest/gtest.h>
#include <mlx/memory.h>

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
#include "hyle/compute/apple_memory.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/apple.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/result.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace mx = mlx::core;
namespace g = hyle::graph;
namespace gb = g::batch;
using row_cache_native_test::take;

constexpr unsigned queries = 3, selections = 3, heads = 64;
constexpr std::array<std::int32_t, 9> logical_ids{-1, 0, 2, 0, 1, 2, 0, 1, 2};

void check(hyle::Status status, const char *what) {
  if (!status.ok())
    throw std::runtime_error(std::string(what) + ": " + std::string(status.message()));
}
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, std::span(shape.begin(), shape.size())).value();
}
c::RowCacheBatchSchema schema(c::RowCacheEncoding encoding) {
  return c::RowCacheBatchSchema::create({
      .encoding = encoding,
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

// Actual MLX shared Metal storage. These writes initialize exclusive cold
// writers; every result read follows its original Graph Completion::wait.
struct Route final {
  a::Context context = a::Context::create().value();
  a::MemoryResource device = a::MemoryResource::create(context).value();
  a::MemoryResource host_resource = a::MemoryResource::create(context).value();
  c::LeasePool pool = c::LeasePool::create(device.ref(), 2u << 20).value();

  c::WriteLease upload(c::WriteLease destination, std::span<const std::byte> bytes) {
    check(destination.view()->write(bytes), "initialize exclusive shared Metal writer");
    return destination;
  }
  c::ReadLease load(std::span<const std::byte> bytes) {
    return upload(pool.allocate(bytes.size()).value(), bytes).publish().value();
  }
};

struct Recipe final {
  g::Graph graph;
  gb::PagedRowCacheGatherPadded rows;
  g::ValueId ids, visibility, cast, query, sinks, attention, negative;
  bool has_sink = false;
};
Recipe recipe(const c::RowCacheBatchSchema &owned, unsigned invalid = 0) {
  auto builder = g::GraphBuilder::create().value();
  const auto depth = owned.capacity().encoding.depth();
  Recipe result;
  result.has_sink = depth == 512;
  result.ids = builder.input(dense(c::DataType::kInt32, {queries, selections})).value();
  result.visibility = builder.input(dense(c::DataType::kInt32, {queries, 2})).value();
  result.rows = gb::add_row_cache_gather_padded(builder, owned, result.ids,
                                              result.visibility, 4).value();
  result.cast = builder.add(c::Cast{c::DataType::kFloat32}, result.rows.output()).value();
  for (auto output : {result.rows.output(), result.rows.mask(), result.cast})
    check(builder.publish(output), "publish exact decoded rows, mask and F32 cast");
  if (result.has_sink) {
    result.query = builder.input(dense(c::DataType::kFloat32, {queries, 1, heads, depth})).value();
    result.sinks = builder.input(dense(c::DataType::kFloat32, {heads})).value();
    const auto reshape = [&](g::ValueId value, std::initializer_list<std::int64_t> shape) {
      return builder.add(c::View::reshape(c::TensorShape::create(
          std::span(shape.begin(), shape.size())).value()).value(), value).value();
    };
    const auto kv = reshape(result.cast, {queries, selections, 1, depth});
    const auto mask = reshape(result.rows.mask(), {queries, 1, selections});
    result.attention = builder.result(builder.add_node(c::model_ops::sink_attention(.0625F),
        std::array{result.query, kv, kv, mask, result.sinks}).value()).value();
    result.negative = builder.add(c::Unary{c::UnaryKind::kNeg}, result.attention).value();
    check(builder.publish(result.attention), "publish original D512 Sink");
    check(builder.publish(result.negative), "publish original Sink Neg");
  }
  if (invalid == 1) {
    const auto unqualified = builder.add(c::Copy{}, result.rows.bank_input()).value();
    check(builder.publish(unqualified), "publish unqualified capacity reader");
  } else if (invalid == 2) {
    const auto map = gb::detail::RowCacheAccess::map(result.rows.reader());
    check(builder.publish(builder.result(map).value()), "publish private mapped IDs");
  }
  result.graph = std::move(builder).finish().value();
  return result;
}

g::CompiledGraph compile(Route &route, const Recipe &model, bool split = false) {
  const auto source = g::apple::execution_source(route.context).value();
  const std::array domains{g::StorageDomain{route.device.ref()}, g::StorageDomain{route.host_resource.ref()}};
  if (!split) {
    const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
    return take(gb::compile(model.graph, {.storage_domains = domains, .targets = targets}),
                "ordinary projected row Graph");
  }
  // Two required placements split only the selected artifacts. Both sources
  // still use the one original Context/Runtime, kernels and completion owner.
  const g::apple::CustomLowering configuration = [](const c::Operation &,
      std::span<const mx::array>, mx::Stream)
      -> hyle::Result<std::vector<mx::array>> {
    return std::unexpected(hyle::Status::Unimplemented(
        "projected row test uses only original native mathematics"));
  };
  const auto other = g::apple::execution_source(route.context,
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
template<class T>std::vector<T> read(Route &, const g::GraphOutput &output) {
  const auto description = output.description().value();
  std::vector<T> result(description.element_count().value());
  check(output.storage_lease().value().view()->read(std::as_writable_bytes(std::span(result))),
        "read successfully retired shared Metal result");
  return result;
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
Expected expected(const row_cache_test::Reference &bank, unsigned phase, unsigned depth, bool empty = false) {
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
void verify(Route &route, g::Invocation &slot, const Recipe &model, const Expected &wanted, unsigned depth) {
  EXPECT_EQ(read<std::uint16_t>(route, slot.result(model.rows.output()).value()), wanted.rows);
  EXPECT_EQ(read<float>(route, slot.result(model.rows.mask()).value()), wanted.mask);
  std::vector<std::uint32_t> widened;
  widened.reserve(wanted.rows.size());
  for (const auto bits : wanted.rows) widened.push_back(std::uint32_t(bits) << 16);
  EXPECT_EQ(read<std::uint32_t>(route, slot.result(model.cast).value()), widened);
  if (!model.has_sink) return;
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
  if (!model.has_sink) return;
  check(slot.bind(model.query, model.graph.value_description(model.query).value(), std::move(query)), "bind Sink query");
  check(slot.bind(model.sinks, model.graph.value_description(model.sinks).value(), std::move(sinks)), "bind Sink denominator");
}

void ready(c::RowCacheEncoding encoding) {
  SCOPED_TRACE(encoding.depth());
  const auto depth = encoding.depth();
  Route route;
  ::testing::Test::RecordProperty("device", route.context.environment()->device_name);
  const auto owned = schema(encoding);
  const auto model = recipe(owned);
  const auto reference = data(encoding);
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
  c::ReadLease query, denominator;
  if (model.has_sink) {
    const std::vector<float> zero(queries * heads * depth), sinks(heads, .25F);
    query = route.load(std::as_bytes(std::span(zero)));
    denominator = route.load(std::as_bytes(std::span(sinks)));
  }
  for (bool split : {false, true}) {
    if (split && !model.has_sink) continue;
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
                                  {.host_workspace = route.host_resource.ref()});
      ASSERT_TRUE(prepared.ok()) << prepared.message();
      ASSERT_TRUE(gb::bind(slot, model.rows.bank_input(),
          arena.read(version, std::array{pages[0], pages[3], pages[1]}).value()).ok());
      const auto visible = ranges(phase);
      bind_inputs(slot, model, ids, route.load(std::as_bytes(std::span(visible))), query, denominator);
      ASSERT_TRUE(slot.prepare_replay({}).ok());
      { auto done = take(slot.submit(), "ordinary ready Graph submit"); ASSERT_TRUE(done.wait().ok()); }
      const auto wanted = expected(reference, phase, depth);
      verify(route, slot, model, wanted, depth);
      ASSERT_TRUE(unused_writer.valid());
      if (!phase) {
        old = slot.result(model.rows.output()).value();
        escaped = old.storage_lease().value();
        old_bits = wanted.rows;
        EXPECT_TRUE(gb::prepare(slot, model.graph, {}, {}, versions,
            {.host_workspace = route.host_resource.ref()}).IsUnavailable());
      } else {
        EXPECT_EQ(read<std::uint16_t>(route, old), old_bits);
        EXPECT_NE(wanted.rows, old_bits);
      }
    }
    EXPECT_FALSE(slots[0].submit());
    old.reset();
    // The escaped readonly MLX array owns its backing independently. It may
    // survive Graph slot reuse; only GraphOutput retains this Invocation slot.
    // Reuse the same device mirror with a different request grouping and page
    // order. A stale metadata upload or stale native layout cannot pass this.
    const auto changed_layout = layout(owned, 1);
    const std::array changed_versions{gb::RowCacheBinding{model.rows.reader(), changed_layout}};
    const auto prepared = gb::prepare(slots[0], model.graph, {}, {}, changed_versions,
                                       {.host_workspace = route.host_resource.ref()});
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    ASSERT_TRUE(gb::bind(slots[0], model.rows.bank_input(),
        arena.read(changed_layout, std::array{pages[1], pages[0], pages[3]}).value()).ok());
    const auto visible = ranges(1);
    bind_inputs(slots[0], model, ids, route.load(std::as_bytes(std::span(visible))), query, denominator);
    ASSERT_TRUE(slots[0].prepare_replay({}).ok());
    { auto replay = take(slots[0].submit(), "original slot replay after observers release"); ASSERT_TRUE(replay.wait().ok()); }
    verify(route, slots[0], model, expected(reference, 1, depth), depth);
    {
      std::vector<std::uint16_t> values(old_bits.size());
      ASSERT_TRUE(escaped.view()->read(std::as_writable_bytes(std::span(values))).ok());
      EXPECT_EQ(values, old_bits);
    }
    escaped.reset();
  }
}

TEST(AppleProjectedRowGraph, Fp8PermutedPagesKeepTwoLayoutsSlotsAndOldResults) {
  ready(c::RowCacheEncoding::float8_block64_tail64());
}
TEST(AppleProjectedRowGraph, Fp4PermutedPagesConsumeExactRowsMaskAndCast) {
  ready(c::RowCacheEncoding::float4_block32());
}

void empty(c::RowCacheEncoding encoding) {
  SCOPED_TRACE(encoding.depth());
  const auto depth = encoding.depth();
  Route route;
  const auto owned = schema(encoding);
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
                              {.host_workspace = route.host_resource.ref()});
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  ASSERT_TRUE(gb::bind(slot, model.rows.bank_input(), arena.read(version, {}).value()).ok());
  const std::array<std::int32_t, 6> visibility{};
  c::ReadLease query, denominator;
  if (model.has_sink) {
    const std::vector<float> zero(queries * heads * depth), sinks(heads, .25F);
    query = route.load(std::as_bytes(std::span(zero)));
    denominator = route.load(std::as_bytes(std::span(sinks)));
  }
  bind_inputs(slot, model, route.load(std::as_bytes(std::span(logical_ids))),
      route.load(std::as_bytes(std::span(visibility))), query, denominator);
  ASSERT_TRUE(slot.prepare_replay({}).ok());
  { auto done = take(slot.submit(), "true-empty projected Graph"); ASSERT_TRUE(done.wait().ok()); }
  verify(route, slot, model, expected({}, 0, depth, true), depth);
  for (const auto &writer : writers) EXPECT_TRUE(writer.valid());
}

TEST(AppleProjectedRowGraph, TrueEmptyHistoryRunsWhileEveryPageWriterIsHeld) {
  empty(c::RowCacheEncoding::float8_block64_tail64());
  empty(c::RowCacheEncoding::float4_block32());
}

TEST(AppleProjectedRowGraph, ExtraBankReaderAndPublishedPhysicalIdsAreRejected) {
  const auto encoding = c::RowCacheEncoding::float8_block64_tail64();
  Route route;
  const auto owned = schema(encoding);
  const auto version = layout(owned, 0);
  for (unsigned invalid : {1u, 2u}) {
    SCOPED_TRACE(invalid == 1 ? "unqualified bank Copy" : "private mapped IDs published");
    const auto model = recipe(owned, invalid);
    const auto compiled = compile(route, model);
    auto slot = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
    const std::array versions{gb::RowCacheBinding{model.rows.reader(), version}};
    const auto denied = gb::prepare(slot, model.graph, {}, {}, versions,
                                    {.host_workspace = route.host_resource.ref()});
    EXPECT_TRUE(denied.IsInvalidArgument()) << denied.message();
    EXPECT_FALSE(slot.submit());
    EXPECT_FALSE(slot.result(model.rows.output()));
  }
}

TEST(AppleProjectedRowGraph, CheckedPlainRowsKeepUnauthorizedPageZeroAndFailureStatus) {
  for (const auto encoding : {c::RowCacheEncoding::float8_block64_tail64(),
                              c::RowCacheEncoding::float4_block32()}) {
    SCOPED_TRACE(encoding.depth());
    Route route;
    const auto owned = schema(encoding);
    const std::array<std::int32_t, 3> q{0, 2, 3}, kv{0, 2, 3}, physical_pages{3, 1, 1};
    const std::array<std::int32_t, 2> tails{1, 1};
    const auto version = owned.version({q, kv, physical_pages, tails}).value();
    auto builder = g::GraphBuilder::create().value();
    const auto ids_desc = dense(c::DataType::kInt32, {queries, selections});
    const auto ids_port = builder.input(ids_desc).value();
    const auto rows = gb::add_row_cache_gather(builder, owned, ids_port).value();
    const auto cast = builder.add(c::Cast{c::DataType::kFloat32}, rows.output()).value();
    check(builder.publish(rows.output()), "publish checked plain rows");
    check(builder.publish(cast), "publish checked plain Cast");
    const auto graph = std::move(builder).finish().value();
    const auto source = g::apple::execution_source(route.context).value();
    const std::array domains{g::StorageDomain{route.device.ref()}, g::StorageDomain{route.host_resource.ref()}};
    const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
    const auto compiled = take(gb::compile(graph, {.storage_domains = domains, .targets = targets}),
                               "checked plain Graph");
    auto arena = c::PagedArena::create(owned, route.device.ref(),
        {.max_backing_bytes = 8 * encoding.row_bytes() + 1}).value();
    const auto reference = data(encoding);
    std::array<c::ReadLease, 2> pages;
    for (unsigned i = 0; i < pages.size(); ++i) {
      const unsigned page = i ? 1 : 3;
      auto writer = arena.acquire(page).value();
      writer.lease() = route.upload(std::move(writer.lease()), std::as_bytes(std::span(reference.bytes)
          .subspan(page * arena.page_bytes(), arena.page_bytes())));
      pages[i] = writer.publish(2).value();
    }
    auto page_zero = arena.acquire(0).value();
    auto unused = arena.acquire(2).value();
    const auto frame = arena.read(version, pages).value();
    const std::array<std::int32_t, 9> ids{0, 1, 2, 2, 1, 0, 0, 0, 0};
    std::vector<std::uint16_t> wanted;
    for (unsigned i = 0; i < ids.size(); ++i) {
      const auto id = static_cast<unsigned>(ids[i]);
      const auto physical = i / selections < 2 ? (id < 2 ? 6 + id : 2) : 2;
      const auto begin = reference.decoded.begin() + physical * encoding.depth();
      wanted.insert(wanted.end(), begin, begin + encoding.depth());
    }
    const std::array versions{gb::RowCacheBinding{rows, version}};
    for (bool invalid : {false, true}) {
      auto slot = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
      auto prepared = gb::prepare(slot, graph, {}, {}, versions,
                                  {.host_workspace = route.host_resource.ref()});
      ASSERT_TRUE(prepared.ok()) << prepared.message();
      auto selected = ids;
      if (invalid) selected[0] = -1; // Checked plain IDs cannot use padding.
      ASSERT_TRUE(slot.bind(ids_port, ids_desc, route.load(std::as_bytes(std::span(selected)))).ok());
      const auto bound = gb::bind(slot, rows.bank_input(), frame);
      ASSERT_TRUE(bound.ok()) << bound.message();
      ASSERT_TRUE(slot.prepare_replay({}).ok());
      auto done = take(slot.submit(), "checked plain submit");
      const auto status = done.wait();
      if (invalid) {
        EXPECT_TRUE(status.IsOutOfRange()) << status.message();
        EXPECT_FALSE(slot.result(rows.output()));
        EXPECT_FALSE(slot.result(cast));
      } else {
        ASSERT_TRUE(status.ok()) << status.message();
        EXPECT_EQ(read<std::uint16_t>(route, slot.result(rows.output()).value()), wanted);
        const auto wide = read<float>(route, slot.result(cast).value());
        ASSERT_EQ(wide.size(), wanted.size());
        for (std::size_t i = 0; i < wide.size(); ++i)
          EXPECT_EQ(wide[i], row_cache_test::decode(wanted[i])) << i;
      }
      EXPECT_TRUE(page_zero.valid());
      EXPECT_TRUE(unused.valid());
    }
  }
}

TEST(AppleProjectedRowGraph, FusedColdPreparationDoesNotAllocateTheLogicalBank) {
  Route route;
  const auto encoding = c::RowCacheEncoding::float8_block64_tail64();
  auto capacity = schema(encoding).capacity();
  capacity.max_pages = 1u << 21;
  const auto owned = c::RowCacheBatchSchema::create(capacity).value();
  const auto model = recipe(owned);
  const auto logical_bytes = model.graph.value_description(model.rows.bank_input())->required_bytes().value();
  constexpr std::size_t limit = 128u << 20;
  ASSERT_GT(logical_bytes, 4 * limit);
  mx::clear_cache();
  mx::reset_peak_memory();
  struct RestoreLimit final {
    std::size_t prior;
    ~RestoreLimit() { mx::set_memory_limit(prior); }
  } restore{mx::set_memory_limit(limit)};
  const auto compiled = compile(route, model);
  EXPECT_EQ(selected_step(compiled.report, gb::detail::RowCacheAccess::map(model.rows.reader())),
            selected_step(compiled.report, model.rows.node()));
  EXPECT_LT(mx::get_peak_memory(), limit);
  ::testing::Test::RecordProperty("logical_bank_bytes", std::to_string(logical_bytes));
  ::testing::Test::RecordProperty("cold_peak_bytes", std::to_string(mx::get_peak_memory()));
  ::testing::Test::RecordProperty("native_memory_limit", std::to_string(limit));
  // No arena/real bank or Invocation is allocated here. The other tests bind
  // real Metal banks and independently verify the original numeric outputs.
}
} // namespace
