#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../../engine/inference/append_transactions.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/custom_kernel.h"
#include "hyle/compute/providers/ggml_cpu.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace b = g::batch;
namespace i = hyle::inference;
namespace m = hyle::mem;

void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
template<class T> T take(hyle::Result<T> value) {
  if (!value) throw std::runtime_error(std::string(value.error().message()));
  return std::forward<T>(*value);
}

// A real Graph effect refuses after both original append effects. It does not
// inject a CUDA/device failure or replace the episode's terminal observation.
struct Refuse final {
  static constexpr c::OperationId operation_id{0x48594c4554455354ULL, 0x494e464150504641ULL};
  static constexpr std::string_view operation_name = "inference-refuse-after-appends";
  constexpr c::OperationTraits operation_traits() const noexcept {
    return {{0, 0}, 0, c::AliasPolicy::kNone, c::EffectKind::kExternalSideEffect};
  }
  hyle::Status infer_output_descriptions(std::span<const c::TensorDesc> input,
      std::span<c::TensorDesc> output) const noexcept {
    return input.empty() && output.empty() ? hyle::Status::Ok()
        : hyle::Status::InvalidArgument("refusal effect has no tensors");
  }
};
constexpr c::ProviderRuntimeProperties runtime{.mode = c::ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = c::ProviderWorkspaceOwner::kCaller, .replay_class = c::ProviderReplayClass::kPrepared};
struct RefuseKernel final {
  struct Invocation final {};
  hyle::Result<Invocation> create_invocation(m::BufferView, m::AllocatorRef) const noexcept { return Invocation{}; }
  c::KernelCommandResult enqueue(Invocation &, c::detail::QueueSequence &,
      std::span<const c::ConstTensorView>, std::span<const c::TensorView>) const noexcept {
    return c::KernelCommandResult::rejected(hyle::Status::Internal("intentional refusal after original appends"));
  }
};
struct RefuseFactory final {
  hyle::Result<c::KernelProposal> propose(const c::KernelRequest &request,
      std::span<c::ResourceRequirement>, std::span<c::ResourceRequirement>) const noexcept {
    if (!request.operation.custom_as<Refuse>()) return std::unexpected(hyle::Status::Unimplemented("not this effect"));
    return c::KernelProposal{.runtime = runtime};
  }
  c::KernelPreparation prepare(const c::KernelPrepareRequest &request, m::AllocatorRef allocator) const noexcept {
    if (!request.operation.custom_as<Refuse>())
      return c::KernelPreparation::unsupported(hyle::Status::Unimplemented("not this effect"));
    auto kernel = c::make_custom_kernel(RefuseKernel{}, {.workspace = {0, 1}, .runtime = runtime,
        .persistent_bytes = 0, .invocation_private_bytes = 0}, allocator);
    return kernel ? c::KernelPreparation::prepared(std::move(*kernel))
        : c::KernelPreparation::error(kernel.error());
  }
};

struct Program final {
  c::Context context;
  c::Queue queue;
  c::BatchSchema schema;
  c::BatchLayoutVersion before, after;
  std::array<c::PagedArena, 2> arenas;
  std::array<c::ReadLease, 2> old_pages;
  std::array<c::PagedArenaRead, 2> old_frames;
  std::array<b::PagedAppendPages, 2> appends;
  std::array<g::NodeId, 2> nodes;
  std::array<g::ValueId, 2> inputs, cache_inputs{};
  std::array<std::uint8_t, 2> bound{};
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation run;
  c::LeasePool input_pool;
  c::ReadLease input;
  c::TensorDesc rows;
  std::array<std::uint16_t, 64> original{};
  const std::uint32_t page_rows;

  explicit Program(bool refuse = false, bool partial_tail = false) : page_rows(partial_tail ? 4u : 2u) {
    auto refusal = take(c::make_custom_provider({{0x48594c45, 0x49415050}, "inference-append-refusal", "1"},
        c::cpu::kBackendId, RefuseFactory{}));
    const std::array providers{c::ProviderBinding{c::ggml_cpu::provider(), 0}, c::ProviderBinding{refusal, 0}};
    context = take(c::Context::create(c::cpu::device(), {.providers = providers}));
    queue = take(context.create_queue());
    const auto plain = take(c::CacheStorage::plain(c::DataType::kBFloat16));
    schema = take(c::BatchSchema::create({.query_type = c::DataType::kBFloat16, .output_type = c::DataType::kBFloat16,
        .key_storage = plain, .value_storage = plain, .max_requests = 1, .max_query_rows = 1, .max_kv_rows = page_rows * 2,
        .max_pages = 2, .max_page_refs = 2, .query_heads = 1, .kv_heads = 1, .qk_head_dim = 64,
        .value_head_dim = 64, .page_size = page_rows}));
    const std::array<std::int32_t, 2> query{0, 1}, ptr{0, 1};
    const std::array<std::int32_t, 1> old_id{0}, new_id{1}, old_last{1}, new_last{2};
    before = take(schema.version({query, ptr, old_id, old_last}));
    after = take(schema.version({query, ptr, new_id, new_last}));
    original.fill(0x3e80); // BF16 0.25; the uninitialized old tail stays NaN.
    for (std::size_t bank = 0; bank < 2; ++bank) {
      arenas[bank] = take(c::PagedArena::create(schema, bank ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey,
          m::system_resource(), {.max_backing_bytes = 4096}));
      auto initial = take(arenas[bank].acquire(0));
      std::vector<std::uint16_t> bytes(page_rows * 64, 0x7fc1);
      std::copy(original.begin(), original.end(), bytes.begin());
      check(initial.lease().view()->write(std::as_bytes(std::span(bytes))));
      old_pages[bank] = take(initial.publish(1));
      old_frames[bank] = take(arenas[bank].read(before, std::span(&old_pages[bank], 1)));
      std::array dirty{take(arenas[bank].acquire(1))};
      appends[bank] = partial_tail
          ? take(b::PagedAppendPages::create(arenas[bank], after, old_frames[bank], dirty,
                std::span<c::PagedPageAppender>{}))
          : take(b::PagedAppendPages::create(arenas[bank], after, old_frames[bank], dirty));
    }
    rows = take(c::TensorDesc::dense(c::DataType::kBFloat16, std::array<std::int64_t, 3>{1, 1, 64}));
    auto builder = take(g::GraphBuilder::create());
    for (std::size_t bank = 0; bank < 2; ++bank) {
      inputs[bank] = take(builder.input(rows));
      auto append = take(b::paged_append(schema, bank ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey));
      nodes[bank] = take(builder.add_node(c::Operation::custom(append), std::array{inputs[bank]}));
    }
    check(builder.depends_on(nodes[1], nodes[0]));
    if (refuse) {
      const auto stop = take(builder.add_node(c::Operation::custom(Refuse{}), {}));
      check(builder.depends_on(stop, nodes[1]));
    }
    graph = take(std::move(builder).finish());
    const std::array domains{g::StorageDomain{m::system_resource()}};
    auto source = take(g::ExecutionSource::acquire(context));
    const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
    compiled = take(b::compile(graph, {.storage_domains = domains, .targets = targets}));
    run = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
    input_pool = take(c::LeasePool::create(m::system_resource(), 128, {.max_blocks = 1}));
    auto writer = take(input_pool.allocate(128, 64));
    std::array<std::uint16_t, 64> incoming; incoming.fill(0x3f40); // BF16 0.75.
    check(writer.view()->write(std::as_bytes(std::span(incoming))));
    input = take(writer.publish());
    for (const auto port : inputs) check(run.bind(port, rows, input));
    check(b::prepare(run, queues(), std::span<const b::NodeLayout>{}));
  }
  ~Program() { context.drain().IgnoreError(); }
  std::array<g::QueueBinding, 1> queues() { return {{{g::ExecutionLaneId{0}, &queue}}}; }
  i::AppendTransactions transactions() { return {nodes, appends, bound}; }
  void old_unchanged() {
    for (const auto &page : old_pages) {
      std::vector<std::uint16_t> actual(page_rows * 64), expected(page_rows * 64, 0x7fc1);
      std::copy(original.begin(), original.end(), expected.begin());
      check(page.view()->read(std::as_writable_bytes(std::span(actual))));
      EXPECT_EQ(actual, expected);
    }
  }
  std::array<c::PagedArenaRead, 2> finish_partial_tails() {
    check(transactions().bind(run, graph, cache_inputs));
    check(run.prepare_replay(queues()));
    auto completion = take(run.submit(queues()));
    std::size_t premature_destinations = 0;
    const std::array owners{i::AppendTailOwner{1, 0}};
    const auto premature_return = transactions().return_tails(owners,
        [&](std::size_t, std::size_t) noexcept -> i::AppendTailDestination { ++premature_destinations; return {}; });
    EXPECT_TRUE(premature_return.IsFailedPrecondition()); EXPECT_EQ(premature_destinations, 0u);
    // A fast CPU kernel does not bypass the original Completion observation.
    const auto premature = transactions().recover(run);
    EXPECT_TRUE(premature);
    if (premature) EXPECT_FALSE(*premature);
    EXPECT_EQ(bound, (std::array<std::uint8_t, 2>{1, 1}));
    check(completion.wait()); completion = {};
    check(transactions().take(run));
    EXPECT_EQ(bound, (std::array<std::uint8_t, 2>{0, 0}));
    std::array<c::PagedArenaRead, 2> ready;
    for (std::size_t bank = 0; bank != ready.size(); ++bank) {
      ready[bank] = take(appends[bank].ready());
      EXPECT_EQ(appends[bank].appender_count(), 1u);
      const auto &tail = take(appends[bank].appender(0));
      EXPECT_TRUE(tail.valid()); EXPECT_EQ(tail.page(), 1u);
      EXPECT_EQ(tail.initialized_rows(), 2u); EXPECT_EQ(tail.reserved_rows(), 0u);
    }
    return ready;
  }
  void ready_unchanged(const std::array<c::PagedArenaRead, 2> &ready) {
    for (std::size_t bank = 0; bank != ready.size(); ++bank) {
      auto prefix = take(ready[bank].page_prefix(1));
      std::array<std::uint16_t, 128> actual{}, expected;
      expected.fill(0x3f40);
      std::copy(original.begin(), original.end(), expected.begin());
      EXPECT_EQ(prefix.size(), sizeof(actual));
      check(take(prefix.view()).read(std::as_writable_bytes(std::span(actual))));
      EXPECT_EQ(actual, expected);
      EXPECT_FALSE(prefix.subview(0, sizeof(actual) + sizeof(original)));
      auto old_prefix = take(old_frames[bank].page_prefix(0));
      EXPECT_EQ(old_prefix.size(), sizeof(original));
      EXPECT_FALSE(old_prefix.subview(0, sizeof(actual)));
    }
    old_unchanged();
  }
  void discard_and_reacquire() {
    auto discarded = transactions().discard(); ASSERT_TRUE(discarded) << discarded.error().message();
    ASSERT_TRUE(*discarded);
    appends = {};
    for (auto &arena : arenas) {
      auto writer = arena.acquire(1); ASSERT_TRUE(writer) << writer.error().message();
      EXPECT_TRUE(writer->lease().valid());
      EXPECT_EQ(writer->initialized_rows(), 0u);
    }
    old_unchanged();
  }
};

TEST(InferenceAppendTransactions, LateBindFailureRemembersActualTransferredPrefix) {
  Program p;
  auto withheld = std::move(p.appends[1]); // Deliberately empty second original transaction.
  EXPECT_FALSE(p.transactions().bind(p.run, p.graph, p.cache_inputs).ok());
  EXPECT_EQ(p.bound, (std::array<std::uint8_t, 2>{1, 0}));
  EXPECT_FALSE(p.appends[0].valid()); EXPECT_TRUE(withheld.valid());
  auto premature = p.transactions().discard(); ASSERT_TRUE(premature); EXPECT_FALSE(*premature);
  auto recovered = p.transactions().recover(p.run); ASSERT_TRUE(recovered) << recovered.error().message();
  ASSERT_TRUE(*recovered); EXPECT_EQ(p.bound, (std::array<std::uint8_t, 2>{0, 0}));
  ASSERT_TRUE(p.appends[0].valid());
  EXPECT_FALSE(p.appends[0].ready()); // No submission is no proof of success.
  auto repeated = p.transactions().recover(p.run); ASSERT_TRUE(repeated); EXPECT_TRUE(*repeated);
  p.appends[1] = std::move(withheld);
  p.discard_and_reacquire();
}

TEST(InferenceAppendTransactions, WholeFailureReturnsReservationsWithoutReadyAuthority) {
  Program p(true);
  check(p.transactions().bind(p.run, p.graph, p.cache_inputs));
  check(p.run.prepare_replay(p.queues()));
  auto completion = take(p.run.submit(p.queues()));
  // Even a fast CPU completion retains its original observation gate until wait.
  auto pending = p.transactions().recover(p.run); ASSERT_TRUE(pending); EXPECT_FALSE(*pending);
  const auto terminal = completion.wait(); completion = {};
  ASSERT_FALSE(terminal.ok());
  EXPECT_NE(terminal.message().find("intentional refusal"), std::string_view::npos);
  check(p.transactions().subspan(0, 1).take(p.run));
  EXPECT_EQ(p.bound, (std::array<std::uint8_t, 2>{0, 1}));
  auto recovered = p.transactions().recover(p.run); ASSERT_TRUE(recovered) << recovered.error().message();
  ASSERT_TRUE(*recovered); EXPECT_EQ(p.bound, (std::array<std::uint8_t, 2>{0, 0}));
  EXPECT_FALSE(p.appends[0].ready()); EXPECT_FALSE(p.appends[1].ready());
  p.discard_and_reacquire();
}

TEST(InferenceAppendTransactions, ReturnedTailOwnerAndFrontierRefusalPreservesLateBankForRetry) {
  Program p(false, true);
  auto ready = p.finish_partial_tails();
  std::array<c::PagedPageAppender, 2> returned;
  std::array<std::uint32_t, 2> frontiers{2, 2};
  std::array<std::size_t, 2> visits{};
  bool wrong_destination = false;
  const auto resolve = [&](std::size_t bank, std::size_t member) noexcept -> i::AppendTailDestination {
    if (bank >= returned.size() || member != 7) { wrong_destination = true; return {}; }
    ++visits[bank]; return {frontiers[bank], &returned[bank]};
  };
  const std::array wrong_owner{i::AppendTailOwner{0, 7}};
  EXPECT_TRUE(p.transactions().return_tails(wrong_owner, resolve).IsFailedPrecondition());
  EXPECT_EQ(visits, (std::array<std::size_t, 2>{0, 0}));
  for (std::size_t bank = 0; bank != returned.size(); ++bank) {
    EXPECT_TRUE(take(p.appends[bank].appender(0)).valid()); EXPECT_FALSE(returned[bank].valid());
  }
  const std::array owners{i::AppendTailOwner{1, 7}};
  frontiers[0] = 1;
  EXPECT_TRUE(p.transactions().return_tails(owners, resolve).IsFailedPrecondition());
  EXPECT_EQ(visits, (std::array<std::size_t, 2>{1, 0}));
  for (std::size_t bank = 0; bank != returned.size(); ++bank) {
    const auto &tail = take(p.appends[bank].appender(0));
    EXPECT_TRUE(tail.valid()); EXPECT_EQ(tail.initialized_rows(), 2u); EXPECT_EQ(tail.reserved_rows(), 0u);
    EXPECT_FALSE(returned[bank].valid());
  }
  frontiers = {2, 1};
  EXPECT_TRUE(p.transactions().return_tails(owners, resolve).IsFailedPrecondition());
  EXPECT_TRUE(returned[0].valid()); EXPECT_FALSE(take(p.appends[0].appender(0)).valid());
  EXPECT_FALSE(returned[1].valid()); EXPECT_TRUE(take(p.appends[1].appender(0)).valid());
  EXPECT_EQ(returned[0].page(), 1u); EXPECT_EQ(returned[0].initialized_rows(), 2u);
  EXPECT_EQ(visits, (std::array<std::size_t, 2>{2, 1}));
  p.ready_unchanged(ready);
  frontiers[1] = 2;
  check(p.transactions().return_tails(owners, resolve));
  EXPECT_FALSE(wrong_destination);
  EXPECT_EQ(visits, (std::array<std::size_t, 2>{2, 2}));
  for (std::size_t bank = 0; bank != returned.size(); ++bank) {
    EXPECT_FALSE(take(p.appends[bank].appender(0)).valid());
    EXPECT_TRUE(returned[bank].valid()); EXPECT_EQ(returned[bank].page(), 1u);
    EXPECT_EQ(returned[bank].initialized_rows(), 2u); EXPECT_EQ(returned[bank].reserved_rows(), 0u);
  }
  check(p.transactions().return_tails(owners, resolve));
  EXPECT_EQ(visits, (std::array<std::size_t, 2>{2, 2})); // No duplicate movement or destination resolution.
  p.ready_unchanged(ready);
  returned = {}; ready = {};
  p.discard_and_reacquire();
}

TEST(InferenceAppendTransactions, WithdrawnReturnedTailsDropCapabilityWithoutWideningReadyPrefixes) {
  Program p(false, true);
  auto ready = p.finish_partial_tails();
  const std::array owners{i::AppendTailOwner{1, 5}};
  std::array<std::size_t, 2> visits{};
  bool wrong_destination = false;
  const auto withdrawn = [&](std::size_t bank, std::size_t member) noexcept -> i::AppendTailDestination {
    if (bank >= visits.size() || member != 5) wrong_destination = true;
    else ++visits[bank];
    return {}; // The real participant withdrew; no new mutable destination exists.
  };
  check(p.transactions().return_tails(owners, withdrawn));
  EXPECT_FALSE(wrong_destination);
  EXPECT_EQ(visits, (std::array<std::size_t, 2>{1, 1}));
  for (auto &transaction : p.appends) EXPECT_FALSE(take(transaction.appender(0)).valid());
  p.ready_unchanged(ready);
  check(p.transactions().return_tails(owners, withdrawn));
  EXPECT_EQ(visits, (std::array<std::size_t, 2>{1, 1}));
  // Immutable ready readers still prevent reuse after mutable-tail withdrawal.
  for (auto &arena : p.arenas) EXPECT_FALSE(arena.acquire(1));
  ready = {};
  p.discard_and_reacquire();
}

struct TailPair final {
  static constexpr std::uint32_t page_rows = 4;
  c::BatchSchema schema;
  std::array<c::PagedArena, 2> arenas;
  std::array<c::ReadLease, 2> prefixes;
  std::array<c::PagedPageAppender, 2> tails;
  std::array<std::vector<std::byte>, 2> before;
  std::size_t row_bytes = 0;

  explicit TailPair(std::uint32_t pages = 2) {
    const auto plain = take(c::CacheStorage::plain(c::DataType::kBFloat16));
    schema = take(c::BatchSchema::create({.query_type = c::DataType::kBFloat16, .output_type = c::DataType::kBFloat16,
        .key_storage = plain, .value_storage = plain, .max_requests = 1, .max_query_rows = 1, .max_kv_rows = pages * page_rows,
        .max_pages = pages, .max_page_refs = pages, .query_heads = 1, .kv_heads = 1, .qk_head_dim = 64,
        .value_head_dim = 64, .page_size = page_rows}));
    for (std::size_t bank = 0; bank != arenas.size(); ++bank) {
      arenas[bank] = take(c::PagedArena::create(schema, bank ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey,
          m::system_resource(), {.max_backing_bytes = 4096}));
      auto writer = take(arenas[bank].acquire(0));
      row_bytes = arenas[bank].page_bytes() / page_rows;
      check(take(writer.lease().view()).fill(std::byte{0x6d}));
      check(take(take(writer.lease().view()).subview(0, row_bytes)).fill(static_cast<std::byte>(0x20 + bank)));
      auto page = take(writer.publish_appendable(1));
      prefixes[bank] = std::move(page.prefix); tails[bank] = std::move(page.appender);
      before[bank].resize(row_bytes);
      check(take(prefixes[bank].view()).read(before[bank]));
    }
  }

  void old_unchanged() {
    for (std::size_t bank = 0; bank != prefixes.size(); ++bank) {
      std::vector<std::byte> actual(row_bytes);
      check(take(prefixes[bank].view()).read(actual));
      EXPECT_EQ(actual, before[bank]);
      EXPECT_EQ(prefixes[bank].size(), row_bytes);
      EXPECT_FALSE(prefixes[bank].subview(0, row_bytes * 2));
    }
  }
};

// Policy IDs only; all actual permissions remain in the original arena's
// dirty writers, appenders and ready prefixes. No alternative page owner or
// native executor is introduced by this acquisition fixture.
struct PhysicalPage final {
  std::uint32_t id;
  std::uint32_t page() const noexcept { return id; }
};
struct PagePreparation final {
  TailPair pair{3};
  const std::array<PhysicalPage, 1> old{{{0}}};
  std::vector<PhysicalPage> next;
  std::array<std::vector<c::PagedPageWriter>, 2> dirty;
  std::array<std::vector<c::PagedPageAppender>, 2> returned;
  std::uint32_t next_id = 1;
  std::size_t acquisitions = 0;

  PagePreparation() { for (auto &bank : returned) bank.reserve(1); }
  hyle::Result<PhysicalPage> acquire() {
    const auto id = next_id++; ++acquisitions;
    auto key = pair.arenas[0].acquire(id);
    if (!key) return std::unexpected(key.error());
    auto value = pair.arenas[1].acquire(id);
    if (!value) return std::unexpected(value.error());
    // Both are real original writers from this one physical page. The common
    // preparation has already reserved each destination bank's capacity.
    dirty[0].push_back(std::move(*key)); dirty[1].push_back(std::move(*value));
    return PhysicalPage{id};
  }
};

TEST(InferenceAppendTransactions, AppendExtentChecksPageBoundariesAndOverflow) {
  struct Case { std::size_t before, added, after, prefix, pages, dirty; std::uint32_t tail, reserved; };
  const std::array cases{Case{0, 1, 1, 0, 1, 1, 0, 0}, Case{15, 1, 16, 0, 1, 1, 15, 1},
      Case{16, 1, 17, 1, 2, 1, 0, 0}, Case{17, 32, 49, 1, 4, 3, 1, 15}};
  for (const auto &test : cases) {
    const auto result = i::append_extent(test.before, test.added, 64, 16);
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(result->after_rows, test.after); EXPECT_EQ(result->prefix_pages, test.prefix);
    EXPECT_EQ(result->page_count, test.pages); EXPECT_EQ(result->dirty_count, test.dirty);
    EXPECT_EQ(result->tail_rows, test.tail); EXPECT_EQ(result->tail_append_rows, test.reserved);
  }
  const auto maximum = (std::numeric_limits<std::size_t>::max)();
  const auto exact = i::append_extent(maximum - 1, 1, maximum, 1);
  ASSERT_TRUE(exact) << exact.error().message();
  EXPECT_EQ(exact->after_rows, maximum); EXPECT_EQ(exact->page_count, maximum); EXPECT_EQ(exact->dirty_count, 1u);
  const auto overflow = i::append_extent(maximum, 1, maximum, 16);
  ASSERT_FALSE(overflow); EXPECT_TRUE(overflow.error().IsOutOfRange());
  const auto over_capacity = i::append_extent(7, 2, 8, 4);
  ASSERT_FALSE(over_capacity); EXPECT_TRUE(over_capacity.error().IsOutOfRange());
  const auto invalid_base = i::append_extent(9, 1, 8, 4);
  ASSERT_FALSE(invalid_base); EXPECT_TRUE(invalid_base.error().IsOutOfRange());
  const auto no_page = i::append_extent(0, 1, 8, 0), no_rows = i::append_extent(0, 0, 8, 4);
  ASSERT_FALSE(no_page); EXPECT_TRUE(no_page.error().IsInvalidArgument());
  ASSERT_FALSE(no_rows); EXPECT_TRUE(no_rows.error().IsInvalidArgument());
}

TEST(InferenceAppendTransactions, TailCohortReservesOriginalBanksWithoutWideningOldReaders) {
  TailPair pair;
  const auto extent = take(i::append_extent(1, 2, 8, TailPair::page_rows));
  EXPECT_FALSE(i::tail_compatible(pair.tails, 0, extent, false));
  EXPECT_FALSE(i::tail_compatible(pair.tails, 1, extent, true));
  EXPECT_FALSE(i::tail_compatible(pair.tails, 0, take(i::append_extent(2, 1, 8, TailPair::page_rows)), true));
  ASSERT_TRUE(i::tail_compatible(pair.tails, 0, extent, true));
  for (const auto &tail : pair.tails) EXPECT_EQ(tail.reserved_rows(), 0u); // Compatibility is read-only.
  const auto reserved = i::reserve_tail(pair.tails, extent);
  ASSERT_TRUE(reserved) << reserved.error().message(); ASSERT_TRUE(*reserved);
  for (std::size_t bank = 0; bank != pair.tails.size(); ++bank) {
    EXPECT_EQ(pair.tails[bank].reserved_rows(), 2u);
    check(take(pair.tails[bank].view()).fill(std::byte{0x55}));
    const auto ready = take(pair.tails[bank].publish());
    EXPECT_EQ(ready.size(), pair.row_bytes * 3);
    EXPECT_EQ(ready.generation(), pair.prefixes[bank].generation());
    EXPECT_EQ(pair.tails[bank].initialized_rows(), 3u);
    std::vector<std::byte> actual(ready.size()), expected(pair.row_bytes * 3, std::byte{0x55});
    std::copy(pair.before[bank].begin(), pair.before[bank].end(), expected.begin());
    check(take(ready.view()).read(actual)); EXPECT_EQ(actual, expected);
  }
  pair.old_unchanged();
}

TEST(InferenceAppendTransactions, LateTailUnavailableRollsBackEarlierBankBeforeCow) {
  TailPair pair;
  const auto extent = take(i::append_extent(1, 2, 8, TailPair::page_rows));
  auto uses = take(c::LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}));
  auto second_bank = take(uses.begin());
  check(second_bank.read(pair.prefixes[1]));
  // A real SDK admission holds the second block. It is not accepted native
  // work or a mocked Completion; dropping this unsubmitted use returns it.
  ASSERT_TRUE(i::tail_compatible(pair.tails, 0, extent, true));
  const auto refused = i::reserve_tail(pair.tails, extent);
  ASSERT_TRUE(refused) << refused.error().message(); EXPECT_FALSE(*refused);
  EXPECT_EQ(pair.tails[0].reserved_rows(), 0u); EXPECT_EQ(pair.tails[1].reserved_rows(), 0u);
  pair.old_unchanged();
  second_bank = {};
  const auto retry = i::reserve_tail(pair.tails, extent);
  ASSERT_TRUE(retry) << retry.error().message(); ASSERT_TRUE(*retry);
  for (auto &tail : pair.tails) check(tail.rollback());
  pair.old_unchanged();
}

TEST(InferenceAppendTransactions, LateTailErrorReturnsOriginalStatusAfterEarlierRollback) {
  TailPair pair;
  const auto extent = take(i::append_extent(1, 1, 8, TailPair::page_rows));
  check(pair.arenas[1].close());
  ASSERT_TRUE(i::tail_compatible(pair.tails, 0, extent, true));
  const auto original = pair.tails[1].reserve_rows(extent.tail_append_rows);
  ASSERT_TRUE(original.IsFailedPrecondition());
  const auto refused = i::reserve_tail(pair.tails, extent);
  ASSERT_FALSE(refused);
  EXPECT_TRUE(refused.error().IsFailedPrecondition()); EXPECT_EQ(refused.error().message(), original.message());
  EXPECT_EQ(pair.tails[0].reserved_rows(), 0u); EXPECT_EQ(pair.tails[1].reserved_rows(), 0u);
  pair.old_unchanged();
}

TEST(InferenceAppendTransactions, CompletePreparationMovesOnlyActuallyReservedTailCapabilities) {
  PagePreparation prepared;
  const auto extent = take(i::append_extent(1, 2, 12, TailPair::page_rows));
  const auto reused = i::prepare_append_pages(extent, std::span<const PhysicalPage>(prepared.old), prepared.next,
      prepared.pair.tails, true, prepared.dirty, prepared.returned, [&] { return prepared.acquire(); });
  ASSERT_TRUE(reused) << reused.error().message(); ASSERT_TRUE(*reused);
  ASSERT_EQ(prepared.next.size(), 1u); EXPECT_EQ(prepared.next[0].page(), 0u);
  EXPECT_EQ(prepared.acquisitions, 0u);
  for (std::size_t bank = 0; bank != prepared.dirty.size(); ++bank) {
    EXPECT_TRUE(prepared.dirty[bank].empty()); EXPECT_FALSE(prepared.pair.tails[bank].valid());
    ASSERT_EQ(prepared.returned[bank].size(), 1u);
    auto &tail = prepared.returned[bank][0];
    EXPECT_EQ(tail.page(), 0u); EXPECT_EQ(tail.initialized_rows(), 1u); EXPECT_EQ(tail.reserved_rows(), 2u);
    check(i::AppendTransactions::rollback_tail(tail));
    EXPECT_EQ(tail.reserved_rows(), 0u); EXPECT_EQ(tail.initialized_rows(), 1u);
    prepared.pair.tails[bank] = std::move(tail);
  }
  prepared.pair.old_unchanged();
}

TEST(InferenceAppendTransactions, CompletePreparationCrossesPageWithOriginalBankWriters) {
  PagePreparation prepared;
  const auto extent = take(i::append_extent(1, 5, 12, TailPair::page_rows));
  const auto reused = i::prepare_append_pages(extent, std::span<const PhysicalPage>(prepared.old), prepared.next,
      prepared.pair.tails, true, prepared.dirty, prepared.returned, [&] { return prepared.acquire(); });
  ASSERT_TRUE(reused) << reused.error().message(); ASSERT_TRUE(*reused);
  ASSERT_EQ(prepared.next.size(), 2u); EXPECT_EQ(prepared.next[0].page(), 0u); EXPECT_EQ(prepared.next[1].page(), 1u);
  EXPECT_EQ(prepared.acquisitions, 1u);
  for (std::size_t bank = 0; bank != prepared.dirty.size(); ++bank) {
    ASSERT_EQ(prepared.returned[bank].size(), 1u); ASSERT_EQ(prepared.dirty[bank].size(), 1u);
    auto &tail = prepared.returned[bank][0]; auto &page = prepared.dirty[bank][0];
    EXPECT_EQ(tail.page(), 0u); EXPECT_EQ(tail.initialized_rows(), 1u); EXPECT_EQ(tail.reserved_rows(), 3u);
    EXPECT_EQ(page.page(), 1u); EXPECT_EQ(page.initialized_rows(), 0u); EXPECT_TRUE(page.lease().valid());
    const auto blocked = prepared.pair.arenas[bank].acquire(1);
    ASSERT_FALSE(blocked); EXPECT_TRUE(blocked.error().IsUnavailable());
    check(i::AppendTransactions::rollback_tail(tail)); prepared.pair.tails[bank] = std::move(tail);
    check(i::AppendTransactions::discard_page(page)); EXPECT_FALSE(page.valid());
    auto recovered = take(prepared.pair.arenas[bank].acquire(1)); EXPECT_EQ(recovered.initialized_rows(), 0u);
  }
  prepared.pair.old_unchanged();
}

TEST(InferenceAppendTransactions, CompletePreparationUsesCowAfterRealLateTailAdmissionRefusal) {
  PagePreparation prepared;
  auto uses = take(c::LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}));
  auto second_bank = take(uses.begin()); check(second_bank.read(prepared.pair.prefixes[1]));
  const auto extent = take(i::append_extent(1, 1, 12, TailPair::page_rows));
  const auto reused = i::prepare_append_pages(extent, std::span<const PhysicalPage>(prepared.old), prepared.next,
      prepared.pair.tails, true, prepared.dirty, prepared.returned, [&] { return prepared.acquire(); });
  ASSERT_TRUE(reused) << reused.error().message(); EXPECT_FALSE(*reused);
  ASSERT_EQ(prepared.next.size(), 1u); EXPECT_EQ(prepared.next[0].page(), 1u); EXPECT_EQ(prepared.acquisitions, 1u);
  for (std::size_t bank = 0; bank != prepared.dirty.size(); ++bank) {
    EXPECT_FALSE(prepared.pair.tails[bank].valid()); EXPECT_TRUE(prepared.returned[bank].empty());
    ASSERT_EQ(prepared.dirty[bank].size(), 1u); auto &page = prepared.dirty[bank][0];
    EXPECT_EQ(page.page(), 1u); EXPECT_TRUE(page.lease().valid()); EXPECT_EQ(page.initialized_rows(), 0u);
    EXPECT_FALSE(prepared.pair.arenas[bank].acquire(0)); // The original ready prefix remains retained.
    check(i::AppendTransactions::discard_page(page));
  }
  second_bank = {}; prepared.pair.old_unchanged();
}

TEST(InferenceAppendTransactions, LateActualPageAcquireRefusalPreservesEarlierStagingPermissions) {
  PagePreparation prepared;
  auto occupied = take(prepared.pair.arenas[1].acquire(2));
  const auto original = prepared.pair.arenas[1].acquire(2);
  ASSERT_FALSE(original); ASSERT_TRUE(original.error().IsUnavailable());
  const auto extent = take(i::append_extent(1, 8, 12, TailPair::page_rows));
  const auto refused = i::prepare_append_pages(extent, std::span<const PhysicalPage>(prepared.old), prepared.next,
      prepared.pair.tails, true, prepared.dirty, prepared.returned, [&] { return prepared.acquire(); });
  ASSERT_FALSE(refused); EXPECT_EQ(refused.error(), original.error()); EXPECT_EQ(prepared.acquisitions, 2u);
  ASSERT_EQ(prepared.next.size(), 2u); EXPECT_EQ(prepared.next[0].page(), 0u); EXPECT_EQ(prepared.next[1].page(), 1u);
  for (std::size_t bank = 0; bank != prepared.dirty.size(); ++bank) {
    EXPECT_FALSE(prepared.pair.tails[bank].valid()); ASSERT_EQ(prepared.returned[bank].size(), 1u);
    const auto &tail = prepared.returned[bank][0];
    EXPECT_TRUE(tail.valid()); EXPECT_EQ(tail.page(), 0u); EXPECT_EQ(tail.initialized_rows(), 1u); EXPECT_EQ(tail.reserved_rows(), 3u);
    ASSERT_EQ(prepared.dirty[bank].size(), 1u); EXPECT_TRUE(prepared.dirty[bank][0].lease().valid());
    EXPECT_EQ(prepared.dirty[bank][0].page(), 1u); EXPECT_FALSE(prepared.pair.arenas[bank].acquire(1));
  }
  // The failed second page's earlier key acquisition was never moved into
  // staging. Its actual original writer returned through normal scope exit.
  { auto unconsumed_key = take(prepared.pair.arenas[0].acquire(2)); EXPECT_TRUE(unconsumed_key.lease().valid()); }
  EXPECT_TRUE(occupied.lease().valid()); prepared.pair.old_unchanged();
  for (std::size_t bank = 0; bank != prepared.dirty.size(); ++bank) {
    auto &tail = prepared.returned[bank][0]; check(i::AppendTransactions::rollback_tail(tail));
    EXPECT_EQ(tail.reserved_rows(), 0u); prepared.pair.tails[bank] = std::move(tail);
    check(i::AppendTransactions::discard_page(prepared.dirty[bank][0]));
    auto recovered = take(prepared.pair.arenas[bank].acquire(1)); EXPECT_EQ(recovered.initialized_rows(), 0u);
  }
  prepared.pair.old_unchanged();
}
} // namespace
