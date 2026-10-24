#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../compute_models/row_cache_native_fixture.h"
#include "../../src/compute_route_apple/apple_file_stream_access.h"
#include "../../src/compute_route_apple/apple_runtime_internal.h"
#include "../../src/graph_batch/row_cache_node.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/apple.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/region_provider.h"
#include "hyle/graph/result.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace ad = a::detail;
namespace mx = mlx::core;
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

// Actual MLX shared Metal storage. These writes initialize exclusive cold
// writers; every result read follows its original Graph Completion::wait.
struct Route final {
  a::Context context = a::Context::create().value();
  a::MemoryResource device = a::MemoryResource::create(context).value();
  a::MemoryResource host_resource = a::MemoryResource::create(context).value();
  c::LeasePool host = c::LeasePool::create(host_resource.ref(), 1u << 20).value();
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
Recipe recipe(const c::RowCacheBatchSchema &owned) {
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
template<class T>std::vector<T> read(Route &, const g::GraphOutput &output) {
  const auto description = output.description().value();
  std::vector<T> result(description.element_count().value());
  check(output.storage_lease().value().view()->read(std::as_writable_bytes(std::span(result))),
        "read successfully retired shared Metal result");
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
// Delay only the original producer FIFO. The actual graph jobs, native
// arrays, copy recording and Completion remain untouched. Every exit opens
// the gate before its stack storage can disappear; the body also times out.
class FifoGate final : public ad::Job {
public:
  FifoGate() {
    execute = [](ad::Job *job) noexcept {
      auto &gate = *static_cast<FifoGate *>(job);
      std::unique_lock lock(gate.mutex_);
      gate.entered_ = true;
      gate.changed_.notify_all();
      gate.timed_out_ = !gate.changed_.wait_for(lock, std::chrono::seconds(5),
          [&] { return gate.open_; });
      gate.finished_ = true;
      gate.changed_.notify_all();
    };
  }
  ~FifoGate() { open_and_join(); }
  hyle::Status enqueue(const std::shared_ptr<ad::Runtime> &runtime) noexcept {
    const auto status = runtime->enqueue(*this);
    admitted_ = status.ok();
    return status;
  }
  bool wait_entered() {
    std::unique_lock lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(5), [&] { return entered_; });
  }
  void open_and_join() {
    if (!admitted_) return;
    std::unique_lock lock(mutex_);
    open_ = true;
    changed_.notify_all();
    changed_.wait(lock, [&] { return finished_; });
    admitted_ = false;
  }
  bool timed_out() {
    std::lock_guard lock(mutex_);
    return timed_out_;
  }
private:
  std::mutex mutex_;
  std::condition_variable changed_;
  bool admitted_ = false, entered_ = false, open_ = false;
  bool finished_ = false, timed_out_ = false;
};

void exercise(c::RowCacheEncoding encoding, bool split = false) {
  SCOPED_TRACE(split ? "split artifacts" : "ordinary fusion");
  SCOPED_TRACE(encoding.depth());
  Route route;
  ::testing::Test::RecordProperty("device", route.context.environment()->device_name);
  const auto owned = schema(encoding);
  const auto version = layout(owned);
  const auto model = recipe(owned);
  const auto compiled = compile(route, model, split);
  const auto map_step = selected_step(compiled.report, gb::detail::RowCacheAccess::map(model.rows.reader()));
  const auto decoder_step = selected_step(compiled.report, model.rows.node());
  if (split) {
    ASSERT_NE(map_step, decoder_step);
    EXPECT_EQ(compiled.report.step(map_step)->target, g::ExecutionTargetId{0});
    EXPECT_EQ(compiled.report.step(decoder_step)->target, g::ExecutionTargetId{1});
  } else {
    ASSERT_EQ(map_step, decoder_step);
  }
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
  c::ReadLease query, denominator;
  if (model.has_sink) {
    const std::vector<float> zero(queries * heads * encoding.depth()), sinks(heads, .25F);
    query = route.load(std::as_bytes(std::span(zero)));
    denominator = route.load(std::as_bytes(std::span(sinks)));
  }
  const std::array versions{gb::RowCacheBinding{model.rows.reader(), version}};
  const auto prepare = [&] {
    return gb::prepare(slot, model.graph, {}, {}, versions,
        {.host_workspace = route.host_resource.ref(), .max_source_completions = 2});
  };
  auto prepared = prepare();
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  for (const auto &[port, input] : std::array{
           std::pair{model.ids, ids}, std::pair{model.visibility, ranges},
           std::pair{model.query, query}, std::pair{model.sinks, denominator}}) {
    if (!port.valid()) continue;
    ASSERT_TRUE(slot.bind(port, model.graph.value_description(port).value(), input).ok());
  }

  auto host_writer = route.host.allocate(copy_bytes).value();
  ASSERT_TRUE(host_writer.view()->write(std::as_bytes(std::span(original.bytes)).first(copy_bytes)).ok());
  const auto payload = host_writer.publish().value();
  const std::array specs{c::CopyResourceSpec{route.host_resource.ref(), copy_bytes},
                         c::CopyResourceSpec{route.device.ref(), arena.page_bytes()}};
  const auto program = a::CopyBatchProgram::prepare(route.context, specs).value();
  auto copy = a::CopyBatchInvocation::create(program).value();
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
  short_copy[0].bytes = row_bytes;
  ASSERT_TRUE(copy.prepare_replay(phases, short_copy).ok());
  auto partial = take(copy.submit(), "original short native page copy");
  ASSERT_TRUE(partial.wait().ok());
  auto update = take(a::CopyBatchUpdate::take(std::move(copy), std::move(partial)), "original short Apple Update");
  const auto rejected = g::apple::submit(slot, {}, std::move(update));
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsOutOfRange()) << rejected.error().message();
  ASSERT_TRUE(update.valid());
  EXPECT_FALSE(slot.result(model.rows.output()));
  EXPECT_FALSE(slot.result(model.rows.mask()));
  EXPECT_FALSE(slot.result(model.cast));
  if (model.has_sink) EXPECT_FALSE(slot.result(model.attention));
  EXPECT_FALSE(g::apple::take_update(slot));
  EXPECT_FALSE(arena.acquire(3));
  ASSERT_TRUE(unused_writer.valid());
  page.lease() = take(update.take_mutable({1}), "return rejected original page writer");
  copy = take(update.release_invocation(), "return rejected original copy invocation");
  ASSERT_TRUE(copy.reset().ok());
  ASSERT_TRUE(copy.bind_source({0}, payload).ok());
  ASSERT_TRUE(copy.bind_mutable({1}, std::move(page.lease())).ok());
  ASSERT_TRUE(copy.prepare_replay(phases, full).ok());

  // Every owner that may wait is declared before the gate. An assertion exit
  // opens the original FIFO before destroying any accepted native work owner.
  a::CopyBatchSubmission copied;
  c::Completion done;
  FifoGate gate;
  const auto runtime = ad::ContextAccess::runtime(route.context);
  ASSERT_TRUE(gate.enqueue(runtime).ok());
  ASSERT_TRUE(gate.wait_entered());
  // All native compilation, metadata copies and uploads finished before this
  // original CPU FIFO gate. It does not assert a physical GPU stall.
  copied = take(copy.submit(), "original full native page copy");
  update = take(a::CopyBatchUpdate::take(std::move(copy), std::move(copied)), "original pending Apple Update");
  std::array<c::Completion *, 2> original_sources{};
  const auto source_count = a::detail::CopyStreamAccess::retirement_sources(update, original_sources);
  ASSERT_TRUE(source_count) << source_count.error().message();
  ASSERT_EQ(*source_count, original_sources.size());
  ASSERT_NE(original_sources.back(), nullptr);
  struct SourceObservation final {
    c::Completion *completion;
    hyle::Result<c::CompletionPoll> polled = std::unexpected(hyle::Status::Unknown());
  } observation{original_sources.back()};
  // The original copy FIFO preflight shares this observation lock. Keep its
  // real Completion as the sole source, rather than relabeling a ready update.
  ad::CopyStreamAccess::serialize(update, [](void *raw) noexcept {
    auto &probe = *static_cast<SourceObservation *>(raw);
    probe.polled = probe.completion->poll();
  }, &observation);
  ASSERT_TRUE(observation.polled) << observation.polled.error().message();
  ASSERT_EQ(*observation.polled, c::CompletionPoll::kPending);
  {
    std::lock_guard lock(runtime->queue_mutex);
    EXPECT_NE(runtime->head, nullptr); // Actual native copy admitted behind gate.
  }
  EXPECT_TRUE(update.in_flight());

  auto accepted = g::apple::submit(slot, {}, std::move(update));
  ASSERT_TRUE(accepted) << accepted.error().message();
  done = std::move(*accepted);
  EXPECT_FALSE(update.valid());
  const auto pending = done.poll();
  ASSERT_TRUE(pending) << pending.error().message();
  EXPECT_EQ(*pending, c::CompletionPoll::kPending);
  EXPECT_FALSE(slot.result(model.rows.output()));
  EXPECT_FALSE(slot.result(model.rows.mask()));
  EXPECT_FALSE(slot.result(model.cast));
  if (model.has_sink) EXPECT_FALSE(slot.result(model.attention));
  if (model.has_sink) EXPECT_FALSE(slot.result(model.negative));
  EXPECT_FALSE(g::apple::take_update(slot));
  EXPECT_FALSE(arena.acquire(3));
  EXPECT_TRUE(unused_writer.valid());
  EXPECT_TRUE(prepare().IsUnavailable());

  gate.open_and_join();
  EXPECT_FALSE(gate.timed_out());
  ASSERT_TRUE(done.wait().ok()); // Original FIFO copy and native suffix retire.
  verify(route, slot, model, wanted, encoding.depth());
  auto output = slot.result(model.rows.output()).value();
  auto old = output;
  done = c::Completion{};
  EXPECT_FALSE(g::apple::take_update(slot));
  output.reset();
  EXPECT_FALSE(g::apple::take_update(slot));
  EXPECT_EQ(read<std::uint16_t>(route, old), wanted.rows);
  EXPECT_FALSE(arena.acquire(3));
  EXPECT_TRUE(unused_writer.valid());
  old.reset();

  update = take(g::apple::take_update(slot), "recover original native page Update");
  EXPECT_FALSE(update.in_flight());
  // Recovering source controls does not turn the consumed frame into ready
  // permission or authorize an ordinary replay without the original Update.
  EXPECT_FALSE(slot.submit({}));
  EXPECT_TRUE(slot.prepare_replay({}).IsFailedPrecondition());
  page.lease() = take(update.take_mutable({1}), "recover original initialized page writer");
  auto published = take(page.publish(2), "publish exact two-row initialized page");
  const std::array complete{published, ready[1], ready[0]};
  EXPECT_TRUE(arena.read(version, complete));
  copy = take(update.release_invocation(), "recover original Apple copy invocation");
  ASSERT_TRUE(copy.reset().ok());
  frame = {};
  EXPECT_TRUE(unused_writer.valid());
  const auto reused = prepare();
  EXPECT_TRUE(reused.ok()) << reused.message();
}

TEST(AppleProjectedRowGraph, PendingNativeFp8CopyKeepsPageSourceAndOldResults) {
  exercise(c::RowCacheEncoding::float8_block64_tail64());
}
TEST(AppleProjectedRowGraph, PendingNativeFp8SplitArtifactsKeepExactSource) {
  exercise(c::RowCacheEncoding::float8_block64_tail64(), true);
}
TEST(AppleProjectedRowGraph, PendingNativeFp4CopyKeepsPageSourceAndOldResults) {
  exercise(c::RowCacheEncoding::float4_block32());
}

// Inspect the actual selected native decoder, independently of the outer
// DependentCompletion. The original prepared artifact and Invocation stay
// alive and are never reset/closed before this borrowed probe is consumed.
struct NativeDecoderProbe final {
  std::uint32_t step = 0;
  std::size_t port = 0;
  const g::ProviderRegionPreparedVTable *table = nullptr;
  void *prepared = nullptr, *invocation = nullptr;
};

void late_source_failure(bool independent_latest) {
  SCOPED_TRACE(independent_latest ? "failed page phase0, successful independent latest phase1"
                                 : "cancel the last writing phase after Graph acceptance");
  Route route;
  ::testing::Test::RecordProperty("device", route.context.environment()->device_name);
  const auto encoding = c::RowCacheEncoding::float8_block64_tail64();
  const auto owned = schema(encoding);
  const auto version = layout(owned);
  const auto model = recipe(owned);
  const auto compiled = compile(route, model);
  const auto decoder_step = selected_step(compiled.report, model.rows.node());
  ASSERT_EQ(decoder_step, selected_step(compiled.report,
      gb::detail::RowCacheAccess::map(model.rows.reader())));
  auto slot = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  auto arena = c::PagedArena::create(owned, route.device.ref(),
      {.max_backing_bytes = 8 * encoding.row_bytes() + 1}).value();
  const auto reference = data(encoding);
  std::array<c::ReadLease, 2> ready;
  for (unsigned page : {0u, 1u}) {
    auto writer = arena.acquire(page).value();
    writer.lease() = route.upload(std::move(writer.lease()), std::as_bytes(std::span(reference.bytes)
        .subspan(page * arena.page_bytes(), arena.page_bytes())));
    ready[page] = writer.publish(1).value();
  }
  auto unused = arena.acquire(2).value();
  auto page = arena.acquire(3).value();
  // Define the physical bytes with a valid, different encoded sentinel. They
  // remain unpublished and have no initialized-row permission. A bad decoder
  // therefore exposes deterministic stale content rather than random bytes.
  page.lease() = route.upload(std::move(page.lease()), std::as_bytes(std::span(reference.bytes)
      .subspan(3 * arena.page_bytes(), arena.page_bytes())));
  const auto payload_bytes = std::as_bytes(std::span(reference.bytes)).first(arena.page_bytes());
  EXPECT_FALSE(std::equal(payload_bytes.begin(), payload_bytes.end(),
      std::as_bytes(std::span(reference.bytes)).begin() + 3 * arena.page_bytes()));
  auto staging = route.host.allocate(payload_bytes.size()).value();
  ASSERT_TRUE(staging.view()->write(payload_bytes).ok());
  const auto payload = staging.publish().value();

  const auto ids = route.load(std::as_bytes(std::span(logical_ids)));
  const auto ranges = route.load(std::as_bytes(std::span(visibility)));
  const std::vector<float> zero(queries * heads * encoding.depth()), sinks(heads, .25F);
  const auto query = route.load(std::as_bytes(std::span(zero)));
  const auto denominator = route.load(std::as_bytes(std::span(sinks)));
  const std::array versions{gb::RowCacheBinding{model.rows.reader(), version}};
  const auto prepared = gb::prepare(slot, model.graph, {}, {}, versions,
      {.host_workspace = route.host_resource.ref(), .max_source_completions = 2});
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  for (const auto &[port, input] : std::array{
           std::pair{model.ids, ids}, std::pair{model.visibility, ranges},
           std::pair{model.query, query}, std::pair{model.sinks, denominator}})
    ASSERT_TRUE(slot.bind(port, model.graph.value_description(port).value(), input).ok());

  const std::array specs{c::CopyResourceSpec{route.host_resource.ref(), payload_bytes.size()},
                         c::CopyResourceSpec{route.device.ref(), arena.page_bytes()},
                         c::CopyResourceSpec{route.device.ref(), 4}};
  const auto program = a::CopyBatchProgram::prepare(route.context, specs).value();
  auto copy = a::CopyBatchInvocation::create(program).value();
  ASSERT_TRUE(copy.bind_source({0}, payload).ok());
  ASSERT_TRUE(copy.bind_mutable({1}, std::move(page.lease())).ok());
  if (independent_latest) {
    auto unrelated = route.pool.allocate(4).value();
    ASSERT_TRUE(unrelated.view()->fill(std::byte{0}).ok());
    ASSERT_TRUE(copy.bind_mutable({2}, std::move(unrelated)).ok());
  }
  const auto writing_phase = independent_latest ? 0u : 1u;
  const std::array<c::CopyPhase, 2> phases{{{},
      {.dependency_mask = independent_latest ? 0u : 1u}}};
  const std::array<c::CopyRange, 2> ranges_to_copy{{
      {{0}, {1}, {writing_phase}, 0, 0, payload_bytes.size()},
      {{0}, {2}, {1}, 0, 0, 4}}};
  ASSERT_TRUE(copy.prepare_replay(phases,
      std::span(ranges_to_copy).first(independent_latest ? 2 : 1)).ok());
  const std::array<c::PagedPageUpdate, 1> updates{{{&page, {1}}}};
  const auto frame = take(arena.pending_read(version, ready, updates), "late-source pending page");
  ASSERT_TRUE(gb::bind(slot, model.rows.bank_input(), frame).ok());
  ASSERT_TRUE(slot.prepare_replay({},
      {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());

  NativeDecoderProbe probe{.step = static_cast<std::uint32_t>(decoder_step)};
  bool found_output = false;
  for (std::size_t port = 0; port < compiled.report.step(decoder_step)->output_count; ++port) {
    const auto physical = compiled.report.step_output(decoder_step, port).value();
    if (compiled.report.physical_value(physical)->value == model.rows.output()) {
      probe.port = port;
      found_output = true;
      break;
    }
  }
  ASSERT_TRUE(found_output);
  // A failed Graph cannot call visit again. Copy only these stable identities
  // while idle, never the callback's borrowed spans or RegionAccessView itself.
  ASSERT_TRUE(g::detail::RegionAccess::visit(slot,
      [](void *raw, const g::detail::RegionAccessView &view) noexcept {
        auto &selected = *static_cast<NativeDecoderProbe *>(raw);
        if (view.step == selected.step) {
          selected.table = view.prepared_vtable;
          selected.prepared = view.prepared;
          selected.invocation = view.invocation;
        }
        return hyle::Status::Ok();
      }, &probe).ok());
  ASSERT_NE(probe.table, nullptr);
  ASSERT_NE(probe.table->output_value, nullptr);
  ASSERT_NE(probe.prepared, nullptr);
  ASSERT_NE(probe.invocation, nullptr);

  a::CopyBatchUpdate update;
  a::CopyBatchSubmission copied;
  c::Completion done;
  FifoGate gate; // Every potentially waiting owner is destroyed after this gate.
  const auto runtime = ad::ContextAccess::runtime(route.context);
  ASSERT_TRUE(gate.enqueue(runtime).ok());
  ASSERT_TRUE(gate.wait_entered());
  copied = take(copy.submit(), "late-source original copy submit");
  update = take(a::CopyBatchUpdate::take(std::move(copy), std::move(copied)), "late-source original Update");
  std::array<c::Completion *, 2> sources{};
  const auto source_count = ad::CopyStreamAccess::retirement_sources(update, sources);
  ASSERT_TRUE(source_count) << source_count.error().message();
  ASSERT_EQ(*source_count, sources.size());
  ASSERT_NE(sources[writing_phase], nullptr);
  const auto source_pending = sources[writing_phase]->poll();
  ASSERT_TRUE(source_pending) << source_pending.error().message();
  ASSERT_EQ(*source_pending, c::CompletionPoll::kPending);
  auto accepted = g::apple::submit(slot, {}, std::move(update));
  ASSERT_TRUE(accepted) << accepted.error().message();
  done = std::move(*accepted);
  EXPECT_FALSE(update.valid());
  EXPECT_EQ(done.poll().value(), c::CompletionPoll::kPending);
  {
    std::lock_guard lock(runtime->queue_mutex);
    std::size_t jobs = 0;
    for (auto *job = runtime->head; job; job = job->next) ++jobs;
    EXPECT_GE(jobs, independent_latest ? 3u : 2u);
  }
  // The gate prevents worker/preflight observation races. Cancel only the
  // original writing phase after the native Graph suffix has been accepted;
  // cancelling done itself would hide the missing source failure check.
  ASSERT_TRUE(sources[writing_phase]->request_cancel().ok());
  gate.open_and_join();
  EXPECT_FALSE(gate.timed_out());
  const auto terminal = done.wait();
  EXPECT_TRUE(terminal.IsCancelled()) << terminal.message();
  EXPECT_FALSE(slot.result(model.rows.output()));
  EXPECT_FALSE(slot.result(model.attention));

  const auto native = probe.table->output_value(probe.prepared, probe.invocation, probe.port);
  ::testing::Test::RecordProperty("native_decoder_has_result", native.has_value() ? "true" : "false");
  EXPECT_FALSE(native) << "outer source failure masked a successful original native decoder";
  if (!native) EXPECT_TRUE(native.error().IsCancelled()) << native.error().message();
  {
    std::lock_guard lock(runtime->queue_mutex);
    EXPECT_TRUE(runtime->failure.ok()) << runtime->failure.message();
  }
  EXPECT_TRUE(unused.valid());
  done = {};
  update = take(g::apple::take_update(slot), "recover failed original source controls");
  EXPECT_FALSE(update.in_flight());
  EXPECT_EQ(update.report().native_calls, independent_latest ? 1u : 0u);
  struct SourceOutcomes final {
    std::array<c::Completion *, 2> sources;
    std::array<hyle::Status, 2> terminal;
  } outcomes{sources, {}};
  ad::CopyStreamAccess::serialize(update, [](void *raw) noexcept {
    auto &observed = *static_cast<SourceOutcomes *>(raw);
    for (std::size_t phase = 0; phase < observed.sources.size(); ++phase) {
      const auto polled = observed.sources[phase]->poll();
      if (!polled) observed.terminal[phase] = polled.error();
      else if (*polled != c::CompletionPoll::kComplete)
        observed.terminal[phase] = hyle::Status::Unavailable("original source phase still pending");
    }
  }, &outcomes);
  EXPECT_TRUE(outcomes.terminal[writing_phase].IsCancelled());
  if (independent_latest) EXPECT_TRUE(outcomes.terminal[1].ok());
  EXPECT_TRUE(update.observe().IsCancelled());
  EXPECT_FALSE(update.take_mutable({1})); // Failed content is never published.
}

TEST(AppleProjectedRowGraph, LateCancelledWritingPhaseMustFailNativeDecoder) {
  late_source_failure(false);
}
TEST(AppleProjectedRowGraph, SuccessfulIndependentLatestPhaseCannotMaskFailedPageSource) {
  late_source_failure(true);
}
} // namespace
