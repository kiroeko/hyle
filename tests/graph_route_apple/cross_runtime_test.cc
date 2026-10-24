#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <span>
#include <thread>

#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/progress.h"
#include "hyle/graph/result.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/scheduler.h"

#include "../../src/compute_route_apple/apple_runtime_internal.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace ad = a::detail;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace s = hyle::sched;
using namespace std::chrono_literals;
constexpr auto timeout = 5s;
constexpr g::ExecutionTargetId first_target{0}, second_target{1};
constexpr g::StorageDomainId first_storage{0}, second_storage{1};

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
      gate.timed_out_ = !gate.changed_.wait_for(lock, timeout,
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
    return changed_.wait_for(lock, timeout, [&] { return entered_; });
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

template<class Predicate> bool eventually(Predicate predicate) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    if (predicate()) return true;
    std::this_thread::sleep_for(1ms);
  } while (std::chrono::steady_clock::now() < deadline);
  return predicate();
}

hyle::Status observe(c::Completion &completion) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    auto polled = completion.poll();
    if (!polled) return polled.error();
    if (*polled == c::CompletionPoll::kComplete) return hyle::Status::Ok();
    std::this_thread::sleep_for(1ms);
  } while (std::chrono::steady_clock::now() < deadline);
  return hyle::Status::DeadlineExceeded("cross-Runtime graph did not retire");
}

std::array<float, 4> values(unsigned slot, unsigned phase) {
  const float seed = float(1 + slot + 3 * phase);
  return {-1.25f * seed, 2.5f + seed, -0.125f * seed, 0.5f * seed};
}

void expect_values(c::ConstTensorView tensor, unsigned slot, unsigned phase) {
  std::array<float, 4> actual{};
  ASSERT_TRUE(tensor.storage().read(std::as_writable_bytes(std::span(actual))).ok());
  const auto input = values(slot, phase);
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
              std::bit_cast<std::uint32_t>(-std::abs(input[i]))) << i;
}

TEST(AppleCrossRuntime, ProducerCopyUsesOriginalFrontierAndKeepsTwoSlots) {
  auto first_context = a::Context::create();
  ASSERT_TRUE(first_context) << first_context.error().message();
  auto second_context = a::Context::create();
  ASSERT_TRUE(second_context) << second_context.error().message();
  RecordProperty("device", first_context->environment()->device_name);
  auto first_memory = a::MemoryResource::create(*first_context);
  auto second_memory = a::MemoryResource::create(*second_context);
  ASSERT_TRUE(first_memory) << first_memory.error().message();
  ASSERT_TRUE(second_memory) << second_memory.error().message();
  ASSERT_NE(first_memory->ref(), second_memory->ref());
  auto first_source = g::apple::execution_source(*first_context);
  auto second_source = g::apple::execution_source(*second_context);
  ASSERT_TRUE(first_source) << first_source.error().message();
  ASSERT_TRUE(second_source) << second_source.error().message();
  ASSERT_FALSE(first_source->same_owner(*second_source));
  const auto first_runtime = ad::ContextAccess::runtime(*first_context);
  const auto second_runtime = ad::ContextAccess::runtime(*second_context);
  ASSERT_NE(first_runtime, second_runtime);
  s::Scheduler scheduler{1, m::system_allocator(), 64};
  s::BlockingPool pool{m::system_allocator(), 1};

  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(description).value();
  const auto absolute = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),
      std::array{input}).value();
  const auto intermediate = builder.result(absolute).value();
  const auto negate = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
      std::array{intermediate}).value();
  const auto output = builder.result(negate).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{first_memory->ref()},
                           g::StorageDomain{second_memory->ref()}};
  const std::array targets{g::ExecutionTarget{*first_source, first_storage},
                           g::ExecutionTarget{*second_source, second_storage}};
  const std::array placements{
      g::NodePlacement{absolute, first_target, g::PlacementStrength::kRequired},
      g::NodePlacement{negate, second_target, g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{input, first_storage},
                            g::ExternalResourceContract{output, second_storage}};
  auto compiled = g::compile(graph, {domains, targets},
      {.node_placements = placements, .external_resources = external});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->executable.region_count(), 2U);
  unsigned copies = 0;
  for (std::size_t index = 0; index < compiled->report.step_count(); ++index) {
    const auto step = compiled->report.step(index).value();
    if (step.kind != g::CompileStepKind::kCopy) continue;
    ++copies;
    // A foreign Runtime cannot read the late native handle directly. The
    // producer's existing FIFO resolves it into a real shared backing that
    // B qualifies. Its storage need not be B's default domain.
    EXPECT_EQ(step.target, first_target);
    EXPECT_EQ(step.source_storage, first_storage);
    EXPECT_TRUE(step.destination_storage == first_storage ||
                step.destination_storage == second_storage);
    EXPECT_EQ(step.copy_bytes, 4 * sizeof(float));
  }
  EXPECT_EQ(copies, 1U);
  for (std::size_t index = 0; index < compiled->report.lane_count(); ++index)
    EXPECT_FALSE(compiled->report.lane(index)->queue_required);

  std::array<m::Buffer, 2> inputs;
  std::array<g::Invocation, 2> slots;
  for (std::size_t slot = 0; slot < slots.size(); ++slot) {
    inputs[slot] = m::Buffer::create(first_memory->ref(), 4 * sizeof(float), 64).value();
    const auto data = values(unsigned(slot), 0);
    ASSERT_TRUE(inputs[slot].view().write(std::as_bytes(std::span(data))).ok());
    auto created = compiled->executable.create_invocation();
    ASSERT_TRUE(created) << created.error().message();
    slots[slot] = std::move(*created);
    ASSERT_TRUE(slots[slot].bind(input,
        c::ConstTensorView::create(description, inputs[slot].view()).value()).ok());
    EXPECT_TRUE(slots[slot].prepare_replay({},
        {.host_wait = g::HostWaitPolicy::kReject}).IsUnimplemented());
    auto prepared = slots[slot].prepare_replay({}, {.host_wait = g::HostWaitPolicy::kAllow});
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    EXPECT_GT(slots[slot].replay_report()->host_wait_dependency_count, 0U);
    ASSERT_TRUE(g::prepare_progress(slots[slot], scheduler, pool).ok());
  }

  std::array<c::Completion, 2> completions;
  FifoGate gate; // Released before Completion/Invocation cleanup on assertions.
  ASSERT_TRUE(gate.enqueue(first_runtime).ok());
  ASSERT_TRUE(gate.wait_entered());
  for (std::size_t slot = 0; slot < slots.size(); ++slot) {
    auto accepted = slots[slot].submit();
    ASSERT_TRUE(accepted) << accepted.error().message();
    completions[slot] = std::move(*accepted);
    EXPECT_EQ(completions[slot].poll().value(), c::CompletionPoll::kPending);
    EXPECT_FALSE(slots[slot].result(output));
  }
  // At least one real native producer has been accepted behind the held
  // original Job. The foreign successor cannot enter its FIFO before A's copy.
  ASSERT_TRUE(eventually([&] {
    std::lock_guard lock(first_runtime->queue_mutex);
    return first_runtime->head != nullptr;
  }));
  {
    std::lock_guard lock(second_runtime->queue_mutex);
    EXPECT_EQ(second_runtime->head, nullptr);
  }
  gate.open_and_join();
  EXPECT_FALSE(gate.timed_out());
  std::array<g::GraphOutput, 2> results;
  for (std::size_t slot = 0; slot < slots.size(); ++slot) {
    const auto terminal = observe(completions[slot]);
    ASSERT_TRUE(terminal.ok()) << terminal.message();
    completions[slot] = {};
    auto result = slots[slot].result(output);
    ASSERT_TRUE(result) << result.error().message();
    results[slot] = std::move(*result);
    auto tensor = results[slot].tensor();
    ASSERT_TRUE(tensor) << tensor.error().message();
    expect_values(*tensor, unsigned(slot), 0);
  }
  EXPECT_FALSE(slots[0].prepare_replay({}).ok());
  EXPECT_FALSE(slots[0].reset().ok());
  auto escaped = results[0].storage_lease();
  ASSERT_TRUE(escaped) << escaped.error().message();

  // Reuse B's other original slot while slot0's GraphOutput retains its old
  // native array. Fixed bindings replay with a genuinely new producer epoch.
  results[1].reset();
  auto changed = values(1, 1);
  ASSERT_TRUE(inputs[1].view().write(std::as_bytes(std::span(changed))).ok());
  auto next = slots[1].submit();
  ASSERT_TRUE(next) << next.error().message();
  auto terminal = observe(*next);
  ASSERT_TRUE(terminal.ok()) << terminal.message();
  *next = {};
  auto current = slots[1].result(output);
  ASSERT_TRUE(current) << current.error().message();
  auto tensor = current->tensor();
  ASSERT_TRUE(tensor) << tensor.error().message();
  expect_values(*tensor, 1, 1);
  auto old = results[0].tensor();
  ASSERT_TRUE(old) << old.error().message();
  expect_values(*old, 0, 0);
  current->reset();

  // An exported readonly Array owner is independent of the GraphOutput gate.
  // Reprepare and rerun slot0, then read both its new value and the old owner.
  results[0].reset();
  changed = values(0, 1);
  ASSERT_TRUE(inputs[0].view().write(std::as_bytes(std::span(changed))).ok());
  ASSERT_TRUE(slots[0].prepare_replay({}).ok());
  next = slots[0].submit();
  ASSERT_TRUE(next) << next.error().message();
  terminal = observe(*next);
  ASSERT_TRUE(terminal.ok()) << terminal.message();
  *next = {};
  current = slots[0].result(output);
  ASSERT_TRUE(current) << current.error().message();
  tensor = current->tensor();
  ASSERT_TRUE(tensor) << tensor.error().message();
  expect_values(*tensor, 0, 1);
  auto old_view = escaped->view();
  ASSERT_TRUE(old_view) << old_view.error().message();
  expect_values(c::ConstTensorView::create(description, *old_view).value(), 0, 0);
  current->reset();
  escaped->reset();
  for (auto &slot : slots) ASSERT_TRUE(slot.drain().ok());
  scheduler.drain();
}
} // namespace
