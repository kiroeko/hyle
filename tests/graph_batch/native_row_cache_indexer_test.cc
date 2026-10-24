#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../compute_models/row_cache_indexer_fixture.h"
#include "../compute_models/row_cache_native_fixture.h"
#include "../../src/graph_batch/row_cache_node.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/result.h"
#if defined(HYLE_NATIVE_ROW_INDEXER_APPLE)
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#else
#include "hyle/compute/vulkan_memory.h"
#include "hyle/graph/vulkan.h"
#endif

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gb = g::batch;
#if defined(HYLE_NATIVE_ROW_INDEXER_APPLE)
namespace native = c::apple;
namespace ng = g::apple;
#else
namespace native = c::vulkan;
namespace ng = g::vulkan;
#endif
using row_cache_native_test::take;
constexpr unsigned tokens = 3, heads = 64, depth = 512, selected = 3;

void check(hyle::Status status, const char *what) {
  if (!status.ok()) throw std::runtime_error(std::string(what) + ": " + std::string(status.message()));
}
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, std::span(shape.begin(), shape.size())).value();
}
struct Route final {
  native::Context context = native::Context::create().value();
#if defined(HYLE_NATIVE_ROW_INDEXER_APPLE)
  native::MemoryResource device = native::MemoryResource::create(context).value();
  native::MemoryResource staging = native::MemoryResource::create(context).value();
#else
  native::MemoryResource device = native::MemoryResource::create(context, native::MemoryKind::kDevice).value();
  native::MemoryResource staging = native::MemoryResource::create(context, native::MemoryKind::kStaging).value();
#endif
  c::LeasePool host = c::LeasePool::create(staging.ref(), 2u << 20).value();
  c::LeasePool pool = c::LeasePool::create(device.ref(), 4u << 20).value();
  native::CopyBatchProgram copy_program = native::CopyBatchProgram::prepare(context,
      std::array{c::CopyResourceSpec{staging.ref(), 2u << 20},
                 c::CopyResourceSpec{device.ref(), 2u << 20}}).value();
  native::CopyBatchInvocation copy = native::CopyBatchInvocation::create(copy_program).value();
  c::WriteLease upload(c::WriteLease destination, std::span<const std::byte> bytes) {
    auto writer = host.allocate(bytes.size()).value();
    check(writer.view()->write(bytes), "write source staging");
    auto source = writer.publish().value();
    check(copy.reset(), "reset upload");
    check(copy.bind_source({0}, source), "bind upload source");
    check(copy.bind_mutable({1}, std::move(destination)), "bind upload destination");
    const std::array<c::CopyPhase, 1> phases{{{}}};
    const std::array<c::CopyRange, 1> ranges{{{{0}, {1}, {0}, 0, 0, bytes.size()}}};
    check(copy.prepare_replay(phases, ranges), "prepare upload");
    { auto done = take(copy.submit(), "submit upload"); check(done.wait(), "wait upload"); }
    auto result = take(copy.unbind_mutable({1}), "return destination");
    check(copy.reset(), "release upload");
    return result;
  }
  template<class T> c::ReadLease load(const std::vector<T> &values) {
    return upload(pool.allocate(values.size() * sizeof(T)).value(), std::as_bytes(std::span(values))).publish().value();
  }
};
template<class T> std::vector<T> read(Route &route, const g::GraphOutput &output) {
  const auto description = output.description().value();
  std::vector<T> values(description.element_count().value());
#if defined(HYLE_NATIVE_ROW_INDEXER_APPLE)
  (void)route;
  check(output.storage_lease().value().view()->read(std::as_writable_bytes(std::span(values))), "read Metal result");
#else
  const auto tensor = route.device.binding(output.storage_lease().value(), description).value();
  check(route.context.download(tensor, std::as_writable_bytes(std::span(values))), "download native result");
#endif
  return values;
}

struct Geometry final {
  unsigned max_requests = 2, page_size = 2, max_pages = 4, columns = 4;
  std::vector<std::int32_t> q{0, 2, 3}, kv{0, 2, 3}, pages{3, 1, 0}, tails{1, 1};
  std::vector<std::int32_t> ranges{0, 0, 0, 3, 0, 1};
  c::RowCacheBatchSchema schema(c::RowCacheEncoding encoding) const {
    return c::RowCacheBatchSchema::create({.encoding = encoding, .max_requests = max_requests,
        .max_query_rows = tokens, .max_kv_rows = page_size * max_pages,
        .max_pages = max_pages, .max_page_refs = 4, .page_size = page_size}).value();
  }
  c::RowCacheLayoutVersion version(const c::RowCacheBatchSchema &schema) const {
    return schema.version({q, kv, pages, tails}).value();
  }
  unsigned physical(unsigned query, unsigned logical) const {
    unsigned request = 0;
    while (query >= static_cast<unsigned>(q[request + 1])) ++request;
    return unsigned(pages[unsigned(kv[request]) + logical / page_size]) * page_size + logical % page_size;
  }
};
struct Bank final {
  c::PagedArena arena;
  std::vector<c::ReadLease> ready;
  std::vector<c::PagedPageWriter> unused;
  Bank(Route &route, const Geometry &geometry, const c::RowCacheBatchSchema &schema,
       std::span<const std::uint8_t> bytes)
      : arena(c::PagedArena::create(schema, route.device.ref(),
          {.max_backing_bytes = geometry.max_pages * geometry.page_size * schema.capacity().encoding.row_bytes() + 1}).value()) {
    for (unsigned page = 0; page < geometry.max_pages; ++page) {
      auto writer = arena.acquire(page).value();
      if (std::find(geometry.pages.begin(), geometry.pages.end(), page) == geometry.pages.end()) {
        unused.push_back(std::move(writer));
      } else {
        writer.lease() = route.upload(std::move(writer.lease()), std::as_bytes(bytes.subspan(
            page * arena.page_bytes(), arena.page_bytes())));
        ready.push_back(writer.publish(geometry.page_size).value());
      }
    }
    // Deliberately distinct from the request reference order.
    std::reverse(ready.begin(), ready.end());
  }
  c::PagedRowCacheRead frame(c::RowCacheLayoutVersion version) {
    return arena.read(version, ready).value();
  }
};
struct Recipe final {
  g::Graph graph;
  gb::PagedRowCacheIndexer scorer;
  gb::PagedRowCacheGatherPadded gather;
  g::ValueId q, w, ranges, ids, top_values, sink_q, sinks, attention, negative;
  bool chain = false;
};
Recipe recipe(const c::RowCacheBatchSchema &index, const c::RowCacheBatchSchema &values,
              unsigned columns, bool chain = true, unsigned invalid = 0) {
  auto builder = g::GraphBuilder::create().value();
  Recipe model;
  model.chain = chain;
  model.q = builder.input(dense(c::DataType::kBFloat16, {tokens, heads, 128})).value();
  model.w = builder.input(dense(c::DataType::kBFloat16, {tokens, heads})).value();
  model.ranges = builder.input(dense(c::DataType::kInt32, {tokens, 2})).value();
  model.scorer = gb::add_row_cache_indexer(builder, index, model.q, model.w, model.ranges, columns).value();
  check(builder.publish(model.scorer.output()), "publish checked scores");
  if (chain) {
    const auto top = builder.add_node(c::Operation(c::TopK{.count = selected}), std::array{model.scorer.output()}).value();
    model.ids = builder.result(top, 1).value();
    model.top_values = builder.result(top).value();
    model.gather = gb::add_row_cache_gather_padded(builder, values, model.ids, model.ranges, columns).value();
    const auto reshape = [&](g::ValueId value, std::initializer_list<std::int64_t> shape) {
      return builder.add(c::View::reshape(c::TensorShape::create(std::span(shape.begin(), shape.size())).value()).value(), value).value();
    };
    const auto floats = builder.add(c::Cast{c::DataType::kFloat32}, model.gather.output()).value();
    const auto kv = reshape(floats, {tokens, selected, 1, depth});
    const auto mask = reshape(model.gather.mask(), {tokens, 1, selected});
    model.sink_q = builder.input(dense(c::DataType::kFloat32, {tokens, 1, heads, depth})).value();
    model.sinks = builder.input(dense(c::DataType::kFloat32, {heads})).value();
    model.attention = builder.result(builder.add_node(c::model_ops::sink_attention(.0625F),
        std::array{model.sink_q, kv, kv, mask, model.sinks}).value()).value();
    model.negative = builder.add(c::Unary{c::UnaryKind::kNeg}, model.attention).value();
    for (const auto output : {model.ids, model.top_values, model.gather.output(), model.gather.mask(), model.attention, model.negative})
      check(builder.publish(output), "publish chain result");
  }
  if (invalid) {
    const auto extra = invalid == 1 ? builder.add(c::Copy{}, model.scorer.bank_input()).value()
                                    : gb::detail::RowCacheAccess::metadata(model.scorer.reader());
    check(builder.publish(extra), "publish illegal private capacity reader");
  }
  model.graph = std::move(builder).finish().value();
  return model;
}
g::CompiledGraph compile(Route &route, const Recipe &model, bool split = false) {
  const auto source = ng::execution_source(route.context).value();
  const std::array domains{g::StorageDomain{route.device.ref()}, g::StorageDomain{route.staging.ref()}};
  if (!split) return take(gb::compile(model.graph, {.storage_domains = domains,
      .targets = std::array{g::ExecutionTarget{source, g::StorageDomainId{0}}}}), "native scorer Graph compile");
#if defined(HYLE_NATIVE_ROW_INDEXER_APPLE)
  const ng::CustomLowering custom = [](const c::Operation &, std::span<const mlx::core::array>, mlx::core::Stream)
      -> hyle::Result<std::vector<mlx::core::array>> {
    return std::unexpected(hyle::Status::Unimplemented("test uses original native math"));
  };
#else
  const ng::CustomLowering custom = [](const c::Operation &, std::span<const c::TensorDesc>, const c::TensorDesc &)
      -> hyle::Result<ng::CustomDispatch> {
    return std::unexpected(hyle::Status::Unimplemented("test uses original native math"));
  };
#endif
  const auto other = ng::execution_source(route.context, {.custom = custom}).value();
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}},
                           g::ExecutionTarget{other, g::StorageDomainId{0}}};
  const std::array placements{
      g::NodePlacement{model.scorer.node(), g::ExecutionTargetId{0}, g::PlacementStrength::kRequired},
      g::NodePlacement{gb::detail::RowCacheAccess::map(model.gather.reader()), g::ExecutionTargetId{1}, g::PlacementStrength::kRequired},
      g::NodePlacement{model.gather.node(), g::ExecutionTargetId{1}, g::PlacementStrength::kRequired}};
  return take(gb::compile(model.graph, {.storage_domains = domains, .targets = targets},
      {.node_placements = placements}), "split scorer/gather Graph compile");
}
std::size_t step(const g::CompileReport &report, g::NodeId node) {
  for (std::size_t i = 0; i < report.step_count(); ++i)
    for (std::size_t j = 0; j < report.step(i)->node_count; ++j)
      if (report.step_node(i, j).value() == node) return i;
  throw std::runtime_error("native node absent from compile report");
}
std::vector<gb::RowCacheBinding> versions(const Recipe &model, const Geometry &geometry,
    const c::RowCacheBatchSchema &index, const c::RowCacheBatchSchema &values) {
  std::vector<gb::RowCacheBinding> result{{model.scorer.reader(), geometry.version(index)}};
  if (model.chain) result.push_back({model.gather.reader(), geometry.version(values)});
  return result;
}
void bind_inputs(Route &route, g::Invocation &slot, const Recipe &model,
    const row_cache_indexer_test::Fixture &data, const std::vector<std::int32_t> &ranges) {
  const auto bind = [&](g::ValueId port, c::ReadLease lease) {
    check(slot.bind(port, model.graph.value_description(port).value(), std::move(lease)), "bind ordinary input");
  };
  bind(model.q, route.load(data.query));
  bind(model.w, route.load(data.weights));
  bind(model.ranges, route.load(ranges));
  if (model.chain) {
    bind(model.sink_q, route.load(std::vector<float>(tokens * heads * depth)));
    bind(model.sinks, route.load(std::vector<float>(heads, .25F)));
  }
}
row_cache_test::Reference attention_data(unsigned rows) {
  // The original FP8 codec has 5-row groups. Encode complete groups, then
  // retain exactly the physical bank's rows; never use a different codec.
  const auto padded = (rows + 4) / 5 * 5;
  row_cache_test::Reference result;
  for (unsigned first = 0; first < padded; first += 5) {
    std::vector<std::uint16_t> source(5 * depth);
    for (unsigned i = 0; i < source.size(); ++i)
      source[i] = row_cache_test::bfloat(double(int(((first * depth + i) * 7) % 31) - 15) / 8);
    const auto encoded = row_cache_test::reference(c::RowCacheEncoding::float8_block64_tail64(), source);
    result.bytes.insert(result.bytes.end(), encoded.bytes.begin(), encoded.bytes.end());
    result.decoded.insert(result.decoded.end(), encoded.decoded.begin(), encoded.decoded.end());
  }
  result.bytes.resize(rows * 584);
  result.decoded.resize(rows * depth);
  return result;
}
void verify(Route &route, g::Invocation &slot, const Recipe &model, const Geometry &geometry,
    row_cache_indexer_test::Fixture data, const row_cache_test::Reference &values) {
  data.ranges = {0, std::int32_t(data.rows), 0, std::int32_t(data.rows), 0, std::int32_t(data.rows)};
  const auto full = data.expected();
  std::vector<float> wanted(tokens * geometry.columns, -std::numeric_limits<float>::infinity());
  for (unsigned q = 0; q < tokens; ++q)
    for (auto key = geometry.ranges[q * 2]; key < geometry.ranges[q * 2 + 1]; ++key)
      wanted[q * geometry.columns + unsigned(key)] = full[q * unsigned(data.rows) + geometry.physical(q, unsigned(key))];
  const auto actual = read<float>(route, slot.result(model.scorer.output()).value());
  ASSERT_EQ(actual.size(), wanted.size());
  for (unsigned i = 0; i < actual.size(); ++i) ASSERT_FLOAT_EQ(actual[i], wanted[i]) << i;
  if (!model.chain) return;
  const auto ids = read<std::int32_t>(route, slot.result(model.ids).value());
  const auto top = read<float>(route, slot.result(model.top_values).value());
  const auto decoded = read<std::uint16_t>(route, slot.result(model.gather.output()).value());
  const auto mask = read<float>(route, slot.result(model.gather.mask()).value());
  const auto attention = read<float>(route, slot.result(model.attention).value());
  const auto negative = read<float>(route, slot.result(model.negative).value());
  for (unsigned q = 0; q < tokens; ++q) {
    std::vector<float> sorted(wanted.begin() + q * geometry.columns, wanted.begin() + (q + 1) * geometry.columns);
    std::sort(sorted.begin(), sorted.end(), std::greater<float>{});
    unsigned count = 0;
    for (unsigned k = 0; k < selected; ++k) {
      const auto id = ids[q * selected + k];
      ASSERT_GE(id, 0); ASSERT_LT(id, std::int32_t(geometry.columns));
      for (unsigned previous = 0; previous < k; ++previous) ASSERT_NE(id, ids[q * selected + previous]);
      ASSERT_FLOAT_EQ(top[q * selected + k], sorted[k]);
      ASSERT_FLOAT_EQ(wanted[q * geometry.columns + unsigned(id)], sorted[k]);
      const bool visible = id >= geometry.ranges[q * 2] && id < geometry.ranges[q * 2 + 1];
      count += visible;
      ASSERT_EQ(mask[q * selected + k], visible ? 0.F : -std::numeric_limits<float>::infinity());
      for (unsigned d = 0; d < depth; ++d)
        ASSERT_EQ(decoded[(q * selected + k) * depth + d], visible ? values.decoded[geometry.physical(q, unsigned(id)) * depth + d] : 0u);
    }
    for (unsigned d = 0; d < depth; ++d) {
      double wanted_value = 0;
      for (unsigned k = 0; k < selected; ++k)
        if (mask[q * selected + k] == 0) wanted_value += row_cache_test::decode(decoded[(q * selected + k) * depth + d]);
      wanted_value /= std::exp(.25) + count;
      for (unsigned h = 0; h < heads; ++h) {
        const auto i = (q * heads + h) * depth + d;
        ASSERT_NEAR(attention[i], wanted_value, 2e-5 + 2e-5 * std::abs(wanted_value)) << i;
        ASSERT_EQ(negative[i], -attention[i]);
      }
    }
  }
}

TEST(NativePagedRowIndexer, CompleteChainReusesSlotsAndChangedLayoutWithExactPages) {
  Route route;
  Geometry geometry;
  const auto index = geometry.schema(c::RowCacheEncoding::float4_block32());
  const auto values = geometry.schema(c::RowCacheEncoding::float8_block64_tail64());
  row_cache_indexer_test::Fixture data(8, tokens);
  const auto original_bank = data.bank;
  const auto decoded = attention_data(8);
  Bank score_bank(route, geometry, index, data.bank), value_bank(route, geometry, values, decoded.bytes);
  const auto model = recipe(index, values, geometry.columns);
  for (bool split : {false, true}) {
    SCOPED_TRACE(split);
    auto compiled = compile(route, model, split);
    const auto score_step = step(compiled.report, model.scorer.node());
    const auto decode_step = step(compiled.report, model.gather.node());
    if (split) {
      ASSERT_NE(score_step, decode_step);
      EXPECT_EQ(compiled.report.step(score_step)->target, g::ExecutionTargetId{0});
      EXPECT_EQ(compiled.report.step(decode_step)->target, g::ExecutionTargetId{1});
    } else EXPECT_EQ(score_step, decode_step);
    std::array slots{compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value(),
                     compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value()};
    g::GraphOutput old;
    std::vector<float> old_values;
    for (unsigned phase : {0u, 1u, 2u}) {
      auto &slot = slots[phase == 1 ? 1 : 0];
      Geometry current;
      if (phase) { current.q = {0, 1, 3}; current.pages = {1, 3, 0}; current.ranges = {0, 3, 0, 1, 0, 1}; }
      data.update(int(phase)); data.bank = original_bank;
      const auto layouts = versions(model, current, index, values);
      if (phase == 2) {
        EXPECT_EQ(read<float>(route, old), old_values);
        EXPECT_TRUE(gb::prepare(slot, model.graph, {}, {}, layouts, {.host_workspace = route.staging.ref()}).IsUnavailable());
        old.reset();
      }
      check(gb::prepare(slot, model.graph, {}, {}, layouts, {.host_workspace = route.staging.ref()}), "cold score/gather prepare");
      check(gb::bind(slot, model.scorer.bank_input(), score_bank.frame(layouts[0].layout)), "bind bank1 exact pages");
      check(gb::bind(slot, model.gather.bank_input(), value_bank.frame(layouts[1].layout)), "bind bank0 exact pages");
      bind_inputs(route, slot, model, data, current.ranges);
      check(slot.prepare_replay({}), "prepare original replay");
      { auto done = take(slot.submit(), "submit original scorer chain"); check(done.wait(), "wait scorer chain"); }
      verify(route, slot, model, current, data, decoded);
      if (!phase) { old = slot.result(model.scorer.output()).value(); old_values = read<float>(route, old); }
      for (const auto &writer : score_bank.unused) EXPECT_TRUE(writer.valid());
      for (const auto &writer : value_bank.unused) EXPECT_TRUE(writer.valid());
    }
  }
}

TEST(NativePagedRowIndexer, EmptyHistoryHasNoBankReadWhileAllWritersAreHeld) {
  Route route;
  Geometry geometry; geometry.kv = {0, 0, 0}; geometry.pages.clear(); geometry.tails = {0, 0}; geometry.ranges.assign(6, 0);
  const auto index = geometry.schema(c::RowCacheEncoding::float4_block32());
  const auto values = geometry.schema(c::RowCacheEncoding::float8_block64_tail64());
  row_cache_indexer_test::Fixture data(8, tokens);
  const auto decoded = attention_data(8);
  Bank score_bank(route, geometry, index, data.bank), value_bank(route, geometry, values, decoded.bytes);
  const auto model = recipe(index, values, geometry.columns);
  auto compiled = compile(route, model);
  auto slot = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  const auto layouts = versions(model, geometry, index, values);
  check(gb::prepare(slot, model.graph, {}, {}, layouts,
      {.host_workspace = route.staging.ref()}), "prepare truly empty scorer");
  check(gb::bind(slot, model.scorer.bank_input(), score_bank.frame(layouts[0].layout)), "empty scorer frame");
  check(gb::bind(slot, model.gather.bank_input(), value_bank.frame(layouts[1].layout)), "empty gather frame");
  bind_inputs(route, slot, model, data, geometry.ranges);
  check(slot.prepare_replay({}), "prepare empty replay");
  { auto done = take(slot.submit(), "submit empty chain"); check(done.wait(), "wait empty chain"); }
  verify(route, slot, model, geometry, data, decoded);
  ASSERT_EQ(score_bank.unused.size(), 4); ASSERT_EQ(value_bank.unused.size(), 4);
  for (const auto &writer : score_bank.unused) EXPECT_TRUE(writer.valid());
  for (const auto &writer : value_bank.unused) EXPECT_TRUE(writer.valid());
}

TEST(NativePagedRowIndexer, TileBoundaryAndSparseRequestGroupsUseLogicalColumns) {
  Route route;
  Geometry geometry;
  geometry.max_requests = 5; geometry.page_size = 128; geometry.max_pages = 5; geometry.columns = 258;
  geometry.q = {0, 0, 2, 2, 3}; geometry.kv = {0, 0, 3, 3, 4};
  geometry.pages = {3, 1, 2, 0}; geometry.tails = {0, 1, 0, 17}; geometry.ranges = {0, 0, 0, 257, 2, 17};
  const auto index = geometry.schema(c::RowCacheEncoding::float4_block32());
  row_cache_indexer_test::Fixture data(640, tokens);
  Bank bank(route, geometry, index, data.bank);
  const auto model = recipe(index, {}, geometry.columns, false);
  auto compiled = compile(route, model);
  auto slot = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  const auto layouts = versions(model, geometry, index, {});
  check(gb::prepare(slot, model.graph, {}, {}, layouts,
      {.host_workspace = route.staging.ref()}), "prepare cross-tile scorer");
  check(gb::bind(slot, model.scorer.bank_input(), bank.frame(layouts[0].layout)), "bind cross-tile exact pages");
  bind_inputs(route, slot, model, data, geometry.ranges);
  check(slot.prepare_replay({}), "prepare cross-tile replay");
  { auto done = take(slot.submit(), "submit cross-tile scorer"); check(done.wait(), "wait cross-tile scorer"); }
  verify(route, slot, model, geometry, data, {});
  ASSERT_EQ(bank.unused.size(), 1); EXPECT_TRUE(bank.unused.front().valid());
}

TEST(NativePagedRowIndexer, PrivateCapacityAndMetadataCannotEscape) {
  Route route;
  Geometry geometry;
  const auto index = geometry.schema(c::RowCacheEncoding::float4_block32());
  for (unsigned invalid : {1u, 2u}) {
    SCOPED_TRACE(invalid);
    const auto model = recipe(index, {}, geometry.columns, false, invalid);
    auto compiled = compile(route, model);
    auto slot = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
    const auto denied = gb::prepare(slot, model.graph, {}, {}, versions(model, geometry, index, {}),
        {.host_workspace = route.staging.ref()});
    EXPECT_TRUE(denied.IsInvalidArgument()) << denied.message();
    EXPECT_FALSE(slot.submit()); EXPECT_FALSE(slot.result(model.scorer.output()));
  }
}

TEST(NativePagedRowIndexer, InvalidLastQueryRangeFailsWithoutPublishedScores) {
  Route route;
  Geometry geometry;
  const auto index = geometry.schema(c::RowCacheEncoding::float4_block32());
  row_cache_indexer_test::Fixture data(8, tokens);
  Bank bank(route, geometry, index, data.bank);
  const auto model = recipe(index, {}, geometry.columns, false);
  auto compiled = compile(route, model);
  for (const std::int32_t bad_end : {2, 5}) {
    auto slot = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
    const auto layouts = versions(model, geometry, index, {});
    check(gb::prepare(slot, model.graph, {}, {}, layouts,
        {.host_workspace = route.staging.ref()}), "prepare checked invalid-range scorer");
    check(gb::bind(slot, model.scorer.bank_input(), bank.frame(layouts[0].layout)), "bind valid pages before bad range");
    auto invalid = geometry.ranges; invalid.back() = bad_end;
    bind_inputs(route, slot, model, data, invalid);
    check(slot.prepare_replay({}), "prepare invalid-range replay");
    { auto done = take(slot.submit(), "submit checked invalid range"); EXPECT_TRUE(done.wait().IsOutOfRange()); }
    EXPECT_FALSE(slot.result(model.scorer.output()));
  }
}
} // namespace

#include "native_row_cache_indexer_pending.inc"
