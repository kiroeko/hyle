// Included after the ready scorer fixture, outside its anonymous namespace.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>

#if defined(HYLE_NATIVE_ROW_INDEXER_APPLE)
#include "hyle/compute/detail/apple_copy_stream_access.h"
#include "../../src/compute_route_apple/apple_runtime_internal.h"
#else
#include "../compute_batch/native_vulkan_failure_fixture.h"
#include "hyle/compute/detail/vulkan_copy_stream_access.h"
#endif

namespace {
#if defined(HYLE_NATIVE_ROW_INDEXER_APPLE)
// The existing original CPU FIFO gate. This does not claim a GPU stall.
class IndexerPendingGate final : public native::detail::Job {
public:
  explicit IndexerPendingGate(const native::Context &context)
      : runtime_(native::detail::ContextAccess::runtime(context)) {
    execute = [](native::detail::Job *job) noexcept {
      auto &gate = *static_cast<IndexerPendingGate *>(job);
      std::unique_lock lock(gate.mutex_);
      gate.entered_ = true;
      gate.changed_.notify_all();
      gate.timed_out_ = !gate.changed_.wait_for(lock, std::chrono::seconds(5),
          [&] { return gate.open_; });
      gate.finished_ = true;
      gate.changed_.notify_all();
    };
  }
  hyle::Status hold() {
    const auto status = runtime_->enqueue(*this);
    if (!status.ok()) return status;
    admitted_ = true;
    std::unique_lock lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(5), [&] { return entered_; })
        ? hyle::Status::Ok() : hyle::Status::Unavailable("original FIFO gate did not enter");
  }
  void release() {
    if (!admitted_) return;
    std::unique_lock lock(mutex_);
    open_ = true;
    changed_.notify_all();
    changed_.wait(lock, [&] { return finished_; });
    admitted_ = false;
  }
  bool timed_out() { std::lock_guard lock(mutex_); return timed_out_; }
  ~IndexerPendingGate() { release(); }
private:
  std::shared_ptr<native::detail::Runtime> runtime_;
  std::mutex mutex_;
  std::condition_variable changed_;
  bool admitted_ = false, entered_ = false, open_ = false;
  bool finished_ = false, timed_out_ = false;
};
#else
// Reuse the native pending fixture's retirement-observation boundary. Every
// poll still queries the actual original fence; this proves pending owners,
// rather than claiming that the GPU has not already executed the copy.
struct IndexerPendingGate final {
  hyle_f5_vk_failure_fixture::Boundary boundary;
  static inline std::atomic<bool> held{false};
  explicit IndexerPendingGate(const native::Context &context) : boundary(context) {}
  hyle::Status hold() {
    held = true;
    boundary.device.fn.GetFenceStatus = [](VkDevice device, VkFence fence) -> VkResult {
      auto &hook = *hyle_f5_vk_failure_fixture::Boundary::active;
      ++hook.polls;
      const auto actual = hook.original.GetFenceStatus(device, fence);
      return held.load() && (actual == VK_SUCCESS || actual == VK_NOT_READY) ? VK_NOT_READY : actual;
    };
    return hyle::Status::Ok();
  }
  void release() { held = false; }
  bool timed_out() const { return false; }
  ~IndexerPendingGate() { release(); }
};
#endif

TEST(NativePagedRowIndexer, PendingBankOneFeedsReadyBankZeroThroughOriginalChain) {
  for (const bool split : {false, true}) {
    SCOPED_TRACE(split ? "required scorer/gather split" : "ordinary fusion");
    Route route;
#if !defined(HYLE_NATIVE_ROW_INDEXER_APPLE)
    ASSERT_FALSE(route.context.environment().software_device);
#endif
    Geometry geometry;
    const auto index = geometry.schema(c::RowCacheEncoding::float4_block32());
    const auto values = geometry.schema(c::RowCacheEncoding::float8_block64_tail64());
    row_cache_indexer_test::Fixture data(8, tokens);
    const auto decoded = attention_data(8);
    Bank value_bank(route, geometry, values, decoded.bytes);
    auto score_arena = c::PagedArena::create(index, route.device.ref(),
        {.max_backing_bytes = 8 * index.capacity().encoding.row_bytes() + 1}).value();
    std::array<c::ReadLease, 2> ready;
    // Logical references are pages3,1,0. Only the two tail pages are ready;
    // page3's two visible rows must come from the original native copy.
    for (unsigned physical : {0u, 1u}) {
      auto writer = score_arena.acquire(physical).value();
      writer.lease() = route.upload(std::move(writer.lease()), std::as_bytes(std::span(data.bank)
          .subspan(physical * score_arena.page_bytes(), score_arena.page_bytes())));
      ready[physical] = writer.publish(1).value();
    }
    auto unused_writer = score_arena.acquire(2).value();
    auto page = score_arena.acquire(3).value();
    const auto copy_bytes = score_arena.page_bytes();
    auto changed = data;
    std::copy_n(data.bank.begin(), copy_bytes, changed.bank.begin() + 3 * copy_bytes);
    auto original_reference = data, changed_reference = changed;
    original_reference.ranges.assign(tokens * 2, 0);
    changed_reference.ranges.assign(tokens * 2, 0);
    for (unsigned query = 0; query < tokens; ++query) {
      original_reference.ranges[query * 2 + 1] = 8;
      changed_reference.ranges[query * 2 + 1] = 8;
    }
    const auto original_scores = original_reference.expected();
    const auto changed_scores = changed_reference.expected();
    ASSERT_TRUE(original_scores[8 + 6] != changed_scores[8 + 6] ||
                original_scores[8 + 7] != changed_scores[8 + 7]);

    const auto model = recipe(index, values, geometry.columns);
    auto compiled = compile(route, model, split);
    const auto scorer_step = step(compiled.report, model.scorer.node());
    const auto gather_step = step(compiled.report, model.gather.node());
    if (split) {
      ASSERT_NE(scorer_step, gather_step);
      EXPECT_EQ(compiled.report.step(scorer_step)->target, g::ExecutionTargetId{0});
      EXPECT_EQ(compiled.report.step(gather_step)->target, g::ExecutionTargetId{1});
    } else {
      ASSERT_EQ(scorer_step, gather_step);
    }
    auto slot = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
    const auto layouts = versions(model, geometry, index, values);
    const auto prepare = [&] {
      return gb::prepare(slot, model.graph, {}, {}, layouts,
          {.host_workspace = route.staging.ref(), .max_source_completions = 2});
    };
    ASSERT_TRUE(prepare().ok());
    bind_inputs(route, slot, model, data, geometry.ranges);
    ASSERT_TRUE(gb::bind(slot, model.gather.bank_input(), value_bank.frame(layouts[1].layout)).ok());

    auto host_writer = route.host.allocate(copy_bytes).value();
    ASSERT_TRUE(host_writer.view()->write(std::as_bytes(std::span(data.bank)).first(copy_bytes)).ok());
    const auto payload = host_writer.publish().value();
    const std::array specs{c::CopyResourceSpec{route.staging.ref(), copy_bytes},
                           c::CopyResourceSpec{route.device.ref(), copy_bytes}};
    const auto program = native::CopyBatchProgram::prepare(route.context, specs).value();
    auto copy = native::CopyBatchInvocation::create(program).value();
    ASSERT_TRUE(copy.bind_source({0}, payload).ok());
    ASSERT_TRUE(copy.bind_mutable({1}, std::move(page.lease())).ok());
    const std::array<c::CopyPhase, 2> phases{{{}, {.dependency_mask = 1}}};
    const std::array<c::CopyRange, 1> ranges{{{{0}, {1}, {1}, 0, 0, copy_bytes}}};
    ASSERT_TRUE(copy.prepare_replay(phases, ranges).ok());
    const std::array<c::PagedPageUpdate, 1> updates{{{&page, {1}}}};
    auto frame = take(score_arena.pending_read(layouts[0].layout, ready, updates), "pending bank1 scorer frame");
    ASSERT_TRUE(gb::bind(slot, model.scorer.bank_input(), frame).ok());
    ASSERT_TRUE(slot.prepare_replay({},
        {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());

    // Every potentially waiting owner precedes the gate. Assertion unwinding
    // opens the original FIFO/observation gate before destroying these owners.
    native::CopyBatchUpdate update;
    native::CopyBatchSubmission copied;
    c::Completion done;
    IndexerPendingGate gate(route.context);
    ASSERT_TRUE(gate.hold().ok());
    copied = take(copy.submit(), "original native pending scorer page copy");
    update = take(native::CopyBatchUpdate::take(std::move(copy), std::move(copied)), "original pending scorer Update");
    std::array<c::Completion *, 2> sources{};
    const auto source_count = native::detail::CopyStreamAccess::retirement_sources(update, sources);
    ASSERT_TRUE(source_count) << source_count.error().message();
    ASSERT_EQ(*source_count, sources.size());
    ASSERT_NE(sources.back(), nullptr);
    struct SourceObservation final {
      c::Completion *source;
      hyle::Result<c::CompletionPoll> result = std::unexpected(hyle::Status::Unknown());
    } observed{sources.back()};
#if defined(HYLE_NATIVE_ROW_INDEXER_APPLE)
    native::detail::CopyStreamAccess::serialize(update, [](void *raw) noexcept {
      auto &probe = *static_cast<SourceObservation *>(raw);
      probe.result = probe.source->poll();
    }, &observed);
#else
    observed.result = observed.source->poll();
    EXPECT_GT(gate.boundary.accepted, 0u);
    EXPECT_GT(gate.boundary.polls.load(), 0u);
#endif
    ASSERT_TRUE(observed.result) << observed.result.error().message();
    ASSERT_EQ(*observed.result, c::CompletionPoll::kPending);
    EXPECT_TRUE(update.in_flight());
    auto accepted = ng::submit(slot, {}, std::move(update));
    ASSERT_TRUE(accepted) << accepted.error().message();
    done = std::move(*accepted);
    EXPECT_FALSE(update.valid());
    const auto pending = done.poll();
    ASSERT_TRUE(pending) << pending.error().message();
    EXPECT_EQ(*pending, c::CompletionPoll::kPending);
    for (const auto output : {model.scorer.output(), model.ids, model.gather.output(), model.gather.mask(), model.attention})
      EXPECT_FALSE(slot.result(output));
    EXPECT_FALSE(ng::take_update(slot));
    EXPECT_FALSE(score_arena.acquire(3));
    EXPECT_TRUE(unused_writer.valid());
    for (const auto &writer : value_bank.unused) EXPECT_TRUE(writer.valid());
    EXPECT_TRUE(prepare().IsUnavailable());

    gate.release();
    EXPECT_FALSE(gate.timed_out());
    ASSERT_TRUE(done.wait().ok());
    // The same independent FP4 scorer, exact BF16 gather/mask and original
    // FP64 Sink budget as the ready chain, now using the copied page contents.
    verify(route, slot, model, geometry, changed, decoded);
    auto output = slot.result(model.scorer.output()).value();
    auto old = output;
    const auto old_scores = read<float>(route, old);
    done = {};
    EXPECT_FALSE(ng::take_update(slot));
    output.reset();
    EXPECT_FALSE(ng::take_update(slot));
    EXPECT_EQ(read<float>(route, old), old_scores);
    EXPECT_FALSE(score_arena.acquire(3));
    old.reset();
    update = take(ng::take_update(slot), "recover original scorer Update");
    EXPECT_FALSE(update.in_flight());
    EXPECT_FALSE(slot.submit({}));
    EXPECT_TRUE(slot.prepare_replay({}).IsFailedPrecondition());
    page.lease() = take(update.take_mutable({1}), "recover scorer bank1 writer");
    const auto published = take(page.publish(2), "publish exact initialized scorer page");
    const std::array complete{ready[1], published, ready[0]};
    EXPECT_TRUE(score_arena.read(layouts[0].layout, complete));
    copy = take(update.release_invocation(), "recover original scorer copy invocation");
    ASSERT_TRUE(copy.reset().ok());
    frame = {};
    EXPECT_TRUE(unused_writer.valid());
    EXPECT_TRUE(prepare().ok());
  }
}
} // namespace
