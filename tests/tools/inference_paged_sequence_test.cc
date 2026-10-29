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

#include "../../engine/inference/append_transactions.h"
#include "../../engine/inference/paged_sequence.h"
#include "hyle/compute/backends/cpu.h"
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

c::BatchSchema schema(std::uint32_t physical_pages) {
  const auto plain = take(c::CacheStorage::plain(c::DataType::kBFloat16));
  return take(c::BatchSchema::create({.query_type = c::DataType::kBFloat16,
      .output_type = c::DataType::kBFloat16, .key_storage = plain, .value_storage = plain,
      .max_requests = 1, .max_query_rows = 1, .max_kv_rows = 4, .max_pages = physical_pages,
      .max_page_refs = 2, .query_heads = 1, .kv_heads = 1, .qk_head_dim = 64,
      .value_head_dim = 64, .page_size = 2}));
}

std::vector<std::uint16_t> words(const c::PagedArenaFrame &frame, std::uint32_t page) {
  const auto source = take(frame.view());
  const auto prefix = take(source.page_prefix(page));
  std::vector<std::uint16_t> result(prefix.size() / sizeof(std::uint16_t));
  check(take(prefix.view()).read(std::as_writable_bytes(std::span(result))));
  return result;
}

struct Program final {
  c::Context context;
  c::Queue queue;
  c::BatchSchema layout_schema;
  i::PagedPageInventory inventory;
  c::TensorDesc rows;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation run;
  g::ValueId input;
  std::array<g::NodeId, 1> nodes;
  c::LeasePool inputs;

  Program() {
    const std::array providers{c::ProviderBinding{c::ggml_cpu::provider(), 0}};
    context = take(c::Context::create(c::cpu::device(), {.providers = providers}));
    queue = take(context.create_queue());
    layout_schema = schema(4);
    inventory.banks.push_back(take(c::PagedArena::create(layout_schema, c::BatchCacheRole::kKey,
        m::system_resource(), {.max_backing_bytes = 4096})));
    check(inventory.prepare(4, 1));
    rows = take(c::TensorDesc::dense(c::DataType::kBFloat16, std::array<std::int64_t, 3>{1, 1, 64}));
    auto builder = take(g::GraphBuilder::create());
    input = take(builder.input(rows));
    const auto append = take(b::paged_append(layout_schema, c::BatchCacheRole::kKey));
    nodes[0] = take(builder.add_node(c::Operation::custom(append), std::array{input}));
    graph = take(std::move(builder).finish());
    const std::array domains{g::StorageDomain{m::system_resource()}};
    const std::array targets{g::ExecutionTarget{take(g::ExecutionSource::acquire(context)), g::StorageDomainId{0}}};
    compiled = take(b::compile(graph, {.storage_domains = domains, .targets = targets}));
    run = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
    check(b::prepare(run, queues(), std::span<const b::NodeLayout>{}));
    inputs = take(c::LeasePool::create(m::system_resource(), 128, {.max_blocks = 1}));
  }
  ~Program() { context.drain().IgnoreError(); }
  std::array<g::QueueBinding, 1> queues() { return {{{g::ExecutionLaneId{0}, &queue}}}; }

  i::PagedSequenceState empty() {
    const std::array<std::int32_t, 2> zero{0, 0};
    const std::array<std::int32_t, 1> tail{0};
    const auto layout = take(layout_schema.version({zero, zero, {}, tail}));
    const auto ready = take(inventory.banks[0].read(layout, {}));
    const auto original = take(c::PagedArenaFootprint::from_read(ready));
    i::PagedSequenceState state;
    if (!state.frames.try_resize(1)) throw std::bad_alloc{};
    state.frames[0] = take(c::PagedArenaFrame::fork(original));
    return state;
  }

  // The same actual frame supplies both the logical boundary and source
  // permissions. No model length, K/V pair or alternative executor is stored.
  bool append(i::PagedSequenceState &state, std::array<c::PagedPageAppender, 1> &reusable,
      std::uint16_t value, bool allow_tail = true) {
    const auto before = state.frames[0].layout();
    const auto extent = take(i::append_extent(before.facts().kv_rows, 1, 4, 2));
    i::PagedSequenceState candidate;
    if (!candidate.frames.try_resize(1)) throw std::bad_alloc{};
    std::array<c::PagedArenaCommit, 1> commits;
    i::PagedSequenceEdit edit(state, candidate, commits);
    const auto source = take(edit.source(0));
    std::array<std::vector<c::PagedPageWriter>, 1> dirty;
    std::array<std::vector<c::PagedPageAppender>, 1> tails;
    tails[0].reserve(1);
    const auto reused = take(i::prepare_append_pages(extent, std::span<const i::PagedPageRef>(state.pages),
        candidate.pages, reusable, allow_tail, dirty, tails,
        [&] { return inventory.acquire_into(dirty); }));
    std::vector<std::int32_t> ids;
    for (const auto &page : candidate.pages) ids.push_back(static_cast<std::int32_t>(page.page()));
    const std::array<std::int32_t, 2> query{0, 1}, ptr{0, static_cast<std::int32_t>(ids.size())};
    const std::array<std::int32_t, 1> last{static_cast<std::int32_t>((extent.after_rows - 1) % 2 + 1)};
    const auto after = take(layout_schema.version({query, ptr, ids, last}));
    const std::array sources{source};
    std::array<b::PagedAppendPages, 1> appends;
    appends[0] = take(b::PagedAppendPages::create(inventory.banks[0], after, before, sources, dirty[0], tails[0]));
    const auto premature = appends[0].ready_delta(source, after);
    EXPECT_FALSE(premature);
    if (!premature) EXPECT_TRUE(premature.error().IsFailedPrecondition());
    std::array<std::uint8_t, 1> bound{};
    const std::array<g::ValueId, 1> cache_inputs{};
    i::AppendTransactions transactions(nodes, appends, bound);
    auto writer = take(inputs.allocate(128, 64));
    std::array<std::uint16_t, 64> incoming; incoming.fill(value);
    check(take(writer.view()).write(std::as_bytes(std::span(incoming))));
    auto input_ready = take(writer.publish());
    check(run.bind(input, rows, input_ready));
    check(transactions.bind(run, graph, cache_inputs));
    check(run.prepare_replay(queues()));
    auto completion = take(run.submit(queues()));
    check(completion.wait()); completion = {};
    check(transactions.take(run));
    EXPECT_EQ(bound[0], 0u);
    check(edit.prepare_commit(0, take(appends[0].ready_delta(source, after))));
    EXPECT_EQ(edit.selected(0).layout(), before);
    check(edit.preflight());
    edit.publish(&state);
    EXPECT_FALSE(source.valid());
    EXPECT_EQ(candidate.frames[0].layout(), after);
    const auto last_page = candidate.pages.back().page();
    const std::array owners{i::AppendTailOwner{last_page, 0}};
    check(transactions.return_tails(owners,
        [&](std::size_t bank, std::size_t member) noexcept -> i::AppendTailDestination {
          if (bank != 0 || member != 0 || extent.after_rows % 2 == 0) return {};
          return {static_cast<std::uint32_t>(extent.after_rows % 2), &reusable[0]};
        }));
    appends[0] = {}; // The original retired transaction must release its COW source.
    state = std::move(candidate);
    check(run.unbind(input));
    EXPECT_EQ(state.frames[0].layout().facts().kv_rows, extent.after_rows);
    return reused;
  }
};

TEST(InferencePagedSequence, SingleBankActualAppendForkCowAndTailReuse) {
  Program program;
  {
    auto state = program.empty();
    std::array<c::PagedPageAppender, 1> tails;
    EXPECT_EQ(state.frames[0].layout().facts().kv_rows, 0u);
    ASSERT_FALSE(program.append(state, tails, 0x3e80)); // BF16 0.25.
    ASSERT_TRUE(tails[0].valid());
    check(state.can_seal()); state.seal();
    auto frozen = take(state.frozen_alias());
    auto branch = take(state.fork());
    const auto original_page = frozen.pages.front().page();
    const auto original = words(frozen.frames[0], original_page);
    ASSERT_EQ(original, std::vector<std::uint16_t>(64, 0x3e80));
    EXPECT_TRUE(frozen.frames[0].sealed()); EXPECT_FALSE(branch.frames[0].sealed());
    state.clear(); tails = {}; // Immutable state/forks never inherit a tail capability.
    ASSERT_FALSE(program.append(branch, tails, 0x3f40, false)); // BF16 0.75, real COW.
    EXPECT_NE(branch.pages.front().page(), original_page);
    std::vector<std::uint16_t> first_page(128, 0x3f40);
    std::fill_n(first_page.begin(), 64, 0x3e80);
    EXPECT_EQ(words(branch.frames[0], branch.pages.front().page()), first_page);
    EXPECT_EQ(words(frozen.frames[0], original_page), original);
    const auto blocked = program.inventory.banks[0].acquire(original_page);
    ASSERT_FALSE(blocked); EXPECT_TRUE(blocked.error().IsUnavailable());
    ASSERT_FALSE(program.append(branch, tails, 0x3f00)); // BF16 0.5, crosses to a new page.
    ASSERT_TRUE(tails[0].valid());
    const auto continued_page = branch.pages.back().page();
    ASSERT_TRUE(program.append(branch, tails, 0x3f80)); // BF16 1.0, same original tail.
    EXPECT_EQ(branch.pages.back().page(), continued_page);
    std::vector<std::uint16_t> last_page(128, 0x3f80);
    std::fill_n(last_page.begin(), 64, 0x3f00);
    EXPECT_EQ(words(branch.frames[0], branch.pages.front().page()), first_page);
    EXPECT_EQ(words(branch.frames[0], continued_page), last_page);
    EXPECT_EQ(words(frozen.frames[0], original_page), original);
    EXPECT_EQ(branch.frames[0].layout().facts().kv_rows, 4u);
    EXPECT_FALSE(tails[0].valid());
    branch.clear(); frozen.clear();
  }
  EXPECT_EQ(program.inventory.free_ids.size(), 4u);
  EXPECT_TRUE(std::ranges::all_of(program.inventory.owners, [](auto count) { return count == 0; }));
  auto free = program.inventory.free_ids; std::ranges::sort(free);
  EXPECT_EQ(free, (std::vector<std::uint32_t>{0, 1, 2, 3}));
  // Actual original arena reservations, not policy counts, prove reuse.
  for (std::uint32_t page = 0; page != 4; ++page) {
    auto writer = take(program.inventory.banks[0].acquire(page));
    EXPECT_TRUE(writer.lease().valid()); EXPECT_EQ(writer.initialized_rows(), 0u);
  }
}

TEST(InferencePagedSequence, LatePhysicalBankRefusalKeepsExistingCohortAndReleasesPartialAcquire) {
  const auto layout_schema = schema(2);
  i::PagedPageInventory inventory;
  // Deliberately reverse the physical roles: writer order follows banks.
  for (const auto role : {c::BatchCacheRole::kValue, c::BatchCacheRole::kKey})
    inventory.banks.push_back(take(c::PagedArena::create(layout_schema, role,
        m::system_resource(), {.max_backing_bytes = 4096})));
  check(inventory.prepare(2, 2));
  std::array<std::vector<c::PagedPageWriter>, 2> dirty;
  for (auto &bank : dirty) bank.reserve(2);
  auto existing = take(inventory.acquire_into(dirty));
  const auto before = inventory.free_ids;
  ASSERT_EQ(before.size(), 1u);
  const auto blocked_page = before.back();
  auto occupied = take(inventory.banks[1].acquire(blocked_page));
  const auto unavailable = inventory.acquire_into(dirty);
  ASSERT_FALSE(unavailable); EXPECT_TRUE(unavailable.error().IsResourceExhausted());
  EXPECT_EQ(inventory.free_ids, before);
  for (auto &bank : dirty) {
    ASSERT_EQ(bank.size(), 1u); EXPECT_TRUE(bank[0].lease().valid()); EXPECT_EQ(bank[0].page(), existing.page());
  }
  { auto released = take(inventory.banks[0].acquire(blocked_page)); EXPECT_TRUE(released.lease().valid()); }
  EXPECT_TRUE(occupied.lease().valid()); occupied = {};
  check(inventory.banks[1].close());
  const auto original = inventory.banks[1].acquire(blocked_page);
  ASSERT_FALSE(original); ASSERT_TRUE(original.error().IsFailedPrecondition());
  const auto refused = inventory.acquire_into(dirty);
  ASSERT_FALSE(refused); EXPECT_EQ(refused.error(), original.error());
  EXPECT_EQ(inventory.free_ids, before);
  { auto released = take(inventory.banks[0].acquire(blocked_page)); EXPECT_TRUE(released.lease().valid()); }
  for (auto &bank : dirty) {
    ASSERT_EQ(bank.size(), 1u); EXPECT_TRUE(bank[0].lease().valid());
    // These writers were never submitted. Their original destructors release
    // the reservations, including the bank explicitly closed above.
    bank.clear();
  }
  existing = {};
  EXPECT_EQ(inventory.free_ids.size(), 2u);
  EXPECT_EQ(inventory.owners, (std::vector<std::uint32_t>{0, 0}));
}
} // namespace
