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

#include "../../engine/inference/paged_append.h"
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

c::BatchSchema schema(std::uint32_t physical_pages, std::uint32_t members = 1) {
  const auto plain = take(c::CacheStorage::plain(c::DataType::kBFloat16));
  return take(c::BatchSchema::create({.query_type = c::DataType::kBFloat16,
      .output_type = c::DataType::kBFloat16, .key_storage = plain, .value_storage = plain,
      .max_requests = members, .max_query_rows = members, .max_kv_rows = 4 * members, .max_pages = physical_pages,
      .max_page_refs = 2 * members, .query_heads = 1, .kv_heads = 1, .qk_head_dim = 64,
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
  i::PagedAppendScratch scratch;

  explicit Program(std::uint32_t members = 1) {
    const std::array providers{c::ProviderBinding{c::ggml_cpu::provider(), 0}};
    context = take(c::Context::create(c::cpu::device(), {.providers = providers}));
    queue = take(context.create_queue());
    layout_schema = schema(4, members);
    inventory.banks.push_back(take(c::PagedArena::create(layout_schema, c::BatchCacheRole::kKey,
        m::system_resource(), {.max_backing_bytes = 4096})));
    check(inventory.prepare(4, 1));
    rows = take(c::TensorDesc::dense(c::DataType::kBFloat16, std::array<std::int64_t, 3>{members, 1, 64}));
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
    inputs = take(c::LeasePool::create(m::system_resource(), 128 * members, {.max_blocks = 1}));
    scratch.prepare(members, layout_schema.capacity().max_page_refs);
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
    const auto source = take(state.frames[0].view());
    std::array<std::vector<c::PagedPageWriter>, 1> dirty;
    std::array<std::vector<c::PagedPageAppender>, 1> tails;
    tails[0].reserve(1);
    std::array<b::PagedAppendPages, 1> appends;
    c::BatchLayoutVersion after;
    const bool had_tail = reusable[0].valid();
    const auto original_tail_page = had_tail ? reusable[0].page() : 0u;
    i::PagedAppendBanks banks{inventory, std::span<std::vector<c::PagedPageWriter>>(dirty),
        std::span<std::vector<c::PagedPageAppender>>(tails), std::span<b::PagedAppendPages>(appends)};
    const auto aggregate = take(i::prepare_paged_append(layout_schema, 1, banks, scratch,
        [&](std::size_t) noexcept -> i::PagedAppendMember {
          return {state, &candidate, after, before.facts().kv_rows, 1, reusable, allow_tail};
        }, {.capacity_rows = 4}));
    EXPECT_EQ(aggregate, after); // R1 reuses its original member version.
    EXPECT_EQ(aggregate.facts().requests, 1u);
    const bool reused = had_tail && candidate.pages.back().page() == original_tail_page;
    if (reused) {
      EXPECT_TRUE(dirty[0].empty()); EXPECT_EQ(appends[0].appender_count(), 1u);
      const auto &tail = take(appends[0].appender(0));
      EXPECT_TRUE(tail.valid()); EXPECT_EQ(tail.reserved_rows(), 1u);
      EXPECT_EQ(tail.page(), original_tail_page);
    }
    const auto finish = [&](std::size_t) noexcept -> i::PagedAppendFinishMember {
      return {state, candidate, commits, after, candidate.pages.back().page(),
          static_cast<std::uint32_t>(extent.after_rows % 2), reusable, true, extent.after_rows % 2 != 0};
    };
    const auto premature = i::prepare_paged_append_commits(appends, 1, finish);
    EXPECT_TRUE(premature.IsFailedPrecondition()); EXPECT_FALSE(commits[0].valid());
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
    check(i::prepare_paged_append_commits(appends, 1, finish));
    EXPECT_EQ(edit.selected(0).layout(), before);
    check(i::return_paged_append_tails(transactions, 1, 1, scratch, finish));
    check(edit.preflight());
    edit.publish(&state);
    EXPECT_FALSE(source.valid());
    EXPECT_EQ(candidate.frames[0].layout(), after);
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

TEST(InferencePagedSequence, ZeroBankSharedPolicyPageHasNoPhysicalTailToReturn) {
  i::PagedPageInventory inventory;
  check(inventory.prepare(2, 0));
  {
    auto page = take(inventory.acquire());
    std::array<i::PagedSequenceState, 2> base, candidate;
    base[0].pages.push_back(page.reference); base[1].pages.push_back(page.reference);
    std::array<c::BatchLayoutVersion, 2> layouts;
    i::PagedAppendScratch scratch; scratch.prepare(2, 4);
    std::span<std::vector<c::PagedPageWriter>> dirty;
    std::span<std::vector<c::PagedPageAppender>> tails;
    std::span<b::PagedAppendPages> appends;
    const auto aggregate = take(i::prepare_paged_append(schema(2, 2), 2,
        i::PagedAppendBanks{inventory, dirty, tails, appends}, scratch,
        [&](std::size_t member) noexcept -> i::PagedAppendMember {
          return {base[member], &candidate[member], layouts[member], 1, 1, {}, true};
        }, {.capacity_rows = 4}));
    EXPECT_EQ(aggregate.facts().requests, 2u);
    ASSERT_EQ(candidate[0].pages[0].page(), candidate[1].pages[0].page());
    const auto finish = [&](std::size_t member) noexcept -> i::PagedAppendFinishMember {
      return {base[member], candidate[member], {}, layouts[member], candidate[member].pages[0].page(), 0};
    };
    check(i::prepare_paged_append_commits(appends, 2, finish));
    const i::AppendTransactions transactions{{}, appends, {}};
    check(i::return_paged_append_tails(transactions, 0, 2, scratch, finish));
    EXPECT_TRUE(scratch.sources.empty()); EXPECT_TRUE(scratch.tail_owners.empty());
    // Empty banks convey policy identity only, never native write authority.
    EXPECT_TRUE(page.writers.empty());
  }
  EXPECT_EQ(inventory.free_ids.size(), 2u);
  EXPECT_EQ(inventory.owners, (std::vector<std::uint32_t>{0, 0}));
}

TEST(InferencePagedSequence, LateMemberExhaustionRetainsOriginalEpisodeThenPublishesActualR2) {
  {
    Program narrow;
    // Prepare enough scratch independently: the R1 schema itself, rather than
    // scratch storage or a forged source, must reject two real members.
    narrow.scratch.prepare(2, 4);
    std::array<i::PagedSequenceState, 2> base, candidate;
    std::array<c::BatchLayoutVersion, 2> layouts;
    for (std::size_t member = 0; member != base.size(); ++member) {
      base[member] = narrow.empty();
      if (!candidate[member].frames.try_resize(1)) throw std::bad_alloc{};
    }
    const auto free = narrow.inventory.free_ids;
    std::array<std::vector<c::PagedPageWriter>, 1> dirty;
    std::array<std::vector<c::PagedPageAppender>, 1> tails;
    std::array<b::PagedAppendPages, 1> appends;
    i::PagedAppendBanks banks{narrow.inventory, std::span<std::vector<c::PagedPageWriter>>(dirty),
        std::span<std::vector<c::PagedPageAppender>>(tails), std::span<b::PagedAppendPages>(appends)};
    const auto rejected = i::prepare_paged_append(narrow.layout_schema, 2, banks, narrow.scratch,
        [&](std::size_t member) noexcept -> i::PagedAppendMember {
          return {base[member], &candidate[member], layouts[member], 0, 1};
        }, {.capacity_rows = 4});
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsInvalidArgument());
    EXPECT_EQ(narrow.inventory.free_ids, free); EXPECT_TRUE(dirty[0].empty());
    EXPECT_FALSE(appends[0].valid());
    for (std::size_t member = 0; member != base.size(); ++member) {
      EXPECT_TRUE(candidate[member].pages.empty()); EXPECT_FALSE(layouts[member].valid());
      EXPECT_EQ(base[member].frames[0].layout().facts().kv_rows, 0u);
      EXPECT_EQ(base[member].frames[0].page_count(), 0u);
    }
  }
  Program program(2);
  // One policy page is insufficient for two independent members. The actual
  // arena/schema still support both; no provider or Completion is simulated.
  check(program.inventory.prepare(1, 1));
  std::array<i::PagedSequenceState, 2> base, candidate;
  std::array<c::BatchLayoutVersion, 2> layouts;
  for (std::size_t member = 0; member != base.size(); ++member) {
    base[member] = program.empty();
    if (!candidate[member].frames.try_resize(1)) throw std::bad_alloc{};
  }
  const std::array old_layouts{base[0].frames[0].layout(), base[1].frames[0].layout()};
  const std::array old_epochs{base[0].frames[0].epoch(), base[1].frames[0].epoch()};
  std::array<std::vector<c::PagedPageWriter>, 1> dirty;
  std::array<std::vector<c::PagedPageAppender>, 1> tails;
  tails[0].reserve(2);
  std::array<b::PagedAppendPages, 1> appends;
  i::PagedAppendBanks banks{program.inventory, std::span<std::vector<c::PagedPageWriter>>(dirty),
      std::span<std::vector<c::PagedPageAppender>>(tails), std::span<b::PagedAppendPages>(appends)};
  const auto member_at = [&](std::size_t member) noexcept -> i::PagedAppendMember {
    return {base[member], &candidate[member], layouts[member], base[member].frames[0].layout().facts().kv_rows, 1};
  };
  const auto failed = i::prepare_paged_append(program.layout_schema, 2, banks, program.scratch,
      member_at, {.capacity_rows = 4});
  ASSERT_FALSE(failed); EXPECT_TRUE(failed.error().IsResourceExhausted());
  ASSERT_EQ(candidate[0].pages.size(), 1u); EXPECT_TRUE(candidate[1].pages.empty());
  ASSERT_EQ(dirty[0].size(), 1u); EXPECT_TRUE(dirty[0][0].lease().valid());
  EXPECT_EQ(dirty[0][0].page(), candidate[0].pages[0].page());
  EXPECT_FALSE(appends[0].valid()); EXPECT_TRUE(program.inventory.free_ids.empty());
  EXPECT_EQ(program.inventory.owners, (std::vector<std::uint32_t>{1}));
  const auto retained = program.inventory.banks[0].acquire(candidate[0].pages[0].page());
  ASSERT_FALSE(retained); EXPECT_TRUE(retained.error().IsUnavailable());
  for (std::size_t member = 0; member != base.size(); ++member) {
    EXPECT_EQ(base[member].frames[0].layout(), old_layouts[member]);
    EXPECT_EQ(base[member].frames[0].epoch(), old_epochs[member]);
    EXPECT_TRUE(base[member].pages.empty());
  }
  // Original unsubmitted recovery releases the actual writer before its ID.
  check(i::AppendTransactions::discard_page(dirty[0][0])); dirty[0].clear();
  for (auto &state : candidate) {
    state.clear(); if (!state.frames.try_resize(1)) throw std::bad_alloc{};
  }
  ASSERT_EQ(program.inventory.free_ids.size(), 1u);
  check(program.inventory.prepare(4, 1));
  const auto aggregate = take(i::prepare_paged_append(program.layout_schema, 2, banks, program.scratch,
      member_at, {.capacity_rows = 4}));
  EXPECT_EQ(aggregate.facts().requests, 2u); EXPECT_EQ(aggregate.facts().query_rows, 2u);
  EXPECT_EQ(aggregate.facts().kv_rows, 2u);
  for (const auto &layout : layouts) {
    EXPECT_EQ(layout.facts().requests, 1u); EXPECT_EQ(layout.facts().query_rows, 1u);
    EXPECT_EQ(layout.facts().kv_rows, 1u); EXPECT_NE(layout, aggregate);
  }
  EXPECT_NE(candidate[0].pages[0].page(), candidate[1].pages[0].page());
  auto writer = take(program.inputs.allocate(256, 64));
  std::array<std::uint16_t, 128> incoming; incoming.fill(0x3f40);
  std::fill_n(incoming.begin(), 64, 0x3e80);
  check(take(writer.view()).write(std::as_bytes(std::span(incoming))));
  auto input_ready = take(writer.publish());
  check(program.run.bind(program.input, program.rows, input_ready));
  std::array<std::uint8_t, 1> bound{};
  const std::array<g::ValueId, 1> cache_inputs{};
  i::AppendTransactions transactions(program.nodes, appends, bound);
  check(transactions.bind(program.run, program.graph, cache_inputs));
  check(program.run.prepare_replay(program.queues()));
  auto completion = take(program.run.submit(program.queues()));
  check(completion.wait()); completion = {};
  check(transactions.take(program.run));
  std::array<std::array<c::PagedArenaCommit, 1>, 2> commits;
  std::array<std::array<c::PagedPageAppender, 1>, 2> returned_tails;
  const auto finish = [&](std::size_t member) noexcept -> i::PagedAppendFinishMember {
    return {base[member], candidate[member], commits[member], layouts[member],
        candidate[member].pages.back().page(), 1, returned_tails[member], true, true};
  };
  check(i::prepare_paged_append_commits(appends, 2, finish));
  check(i::return_paged_append_tails(transactions, 1, 2, program.scratch, finish));
  for (std::size_t member = 0; member != base.size(); ++member)
    check(i::PagedSequenceEdit(base[member], candidate[member], commits[member]).preflight());
  for (std::size_t member = 0; member != base.size(); ++member) {
    i::PagedSequenceEdit(base[member], candidate[member], commits[member]).publish(&base[member]);
    base[member] = std::move(candidate[member]);
    EXPECT_EQ(base[member].frames[0].layout(), layouts[member]);
    const auto expected = member ? std::uint16_t{0x3f40} : std::uint16_t{0x3e80};
    EXPECT_EQ(words(base[member].frames[0], base[member].pages[0].page()),
        std::vector<std::uint16_t>(64, expected));
    const auto mapping = base[member].frames[0].layout().host().page_indices;
    ASSERT_EQ(mapping.size(), 1u);
    EXPECT_EQ(mapping[0], static_cast<std::int32_t>(base[member].pages[0].page()));
    EXPECT_TRUE(returned_tails[member][0].valid());
  }
  appends[0] = {};
  check(program.run.unbind(program.input));
  returned_tails = {};
  for (auto &state : base) state.clear();
  EXPECT_EQ(program.inventory.free_ids.size(), 4u);
  EXPECT_EQ(program.inventory.owners, (std::vector<std::uint32_t>{0, 0, 0, 0}));
  for (std::uint32_t page = 0; page != 4; ++page) {
    auto released = take(program.inventory.banks[0].acquire(page)); EXPECT_TRUE(released.lease().valid());
  }
}
} // namespace
