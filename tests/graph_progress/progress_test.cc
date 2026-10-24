#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/progress.h"
#include "hyle/graph/result.h"
#include "hyle/mem/buffer.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/scheduler.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace s = hyle::sched;
using namespace std::chrono_literals;
constexpr auto kTimeout = 5s;
// Native unary routes do not fan out through their executor. These two real
// normalization shards exceed its 16K work threshold, without changing policy.
constexpr std::size_t kRows = 8, kWidth = 4096, kElements = kRows * kWidth;
constexpr float kEpsilon = 1.0e-5F;

template<class Predicate> bool eventually(Predicate predicate) {
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  do {
    if (predicate()) return true;
    std::this_thread::sleep_for(1ms);
  } while (std::chrono::steady_clock::now() < deadline);
  return predicate();
}

hyle::Status observe(c::Completion &completion) {
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  do {
    auto polled = completion.poll();
    if (!polled) return polled.error();
    if (*polled == c::CompletionPoll::kComplete) return hyle::Status::Ok();
    std::this_thread::sleep_for(1ms);
  } while (std::chrono::steady_clock::now() < deadline);
  return hyle::Status::DeadlineExceeded("graph progress observation timed out");
}

// A real borrowed, synchronous executor. It runs every requested native shard;
// its optional gate delays the actual provider body, never a Completion shim.
struct GateExecutor final {
  std::mutex mutex;
  std::condition_variable changed;
  bool armed = false, entered = false, released = true, timed_out = false;
  std::atomic<unsigned> calls{0};
  std::atomic<int> scheduler_worker{-2};

  void arm() {
    std::lock_guard lock(mutex);
    armed = true; entered = false; released = false; timed_out = false;
  }
  void release() {
    std::lock_guard lock(mutex);
    released = true;
    changed.notify_all();
  }
  bool wait_entered() {
    std::unique_lock lock(mutex);
    return changed.wait_for(lock, kTimeout, [&] { return entered; });
  }
  bool held() {
    std::lock_guard lock(mutex);
    return entered && !released;
  }
  bool timeout() {
    std::lock_guard lock(mutex);
    return timed_out;
  }
  c::ParallelExecutorRef ref() noexcept {
    return {this, 2, [](void *opaque, std::size_t count,
                       c::ParallelTaskFn task, void *context) noexcept {
      auto &self = *static_cast<GateExecutor *>(opaque);
      self.calls.fetch_add(1, std::memory_order_relaxed);
      self.scheduler_worker.store(s::current_worker_index(), std::memory_order_relaxed);
      {
        std::unique_lock lock(self.mutex);
        if (self.armed) {
          self.armed = false;
          self.entered = true;
          self.changed.notify_all();
          if (!self.changed.wait_for(lock, 2 * kTimeout, [&] { return self.released; })) {
            self.timed_out = true;
            self.released = true; // failures must not strand a pool worker
          }
        }
      }
      for (std::size_t i = 0; i < count; ++i) task(context, i, count);
    }};
  }
};

struct Tensor final { m::Buffer memory; c::TensorView view; };
hyle::Result<Tensor> tensor(c::TensorDesc description) {
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto memory, m::Buffer::create(m::system_resource(), bytes, 64));
  HYLE_ASSIGN_OR_RETURN(auto view, c::TensorView::create(description, memory.view()));
  return Tensor{std::move(memory), view};
}

class GraphProgress : public ::testing::Test {
protected:
  void SetUp() override {
    auto selected = c::Context::create(c::cpu::device(),
        c::native::provider({.executor = executor.ref()}));
    ASSERT_TRUE(selected) << selected.error().message();
    context = std::move(*selected);
    auto made_queue = context.create_queue();
    ASSERT_TRUE(made_queue) << made_queue.error().message();
    queue = std::move(*made_queue);
    auto desc = c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{kRows, kWidth});
    ASSERT_TRUE(desc) << desc.error().message();
    auto source = tensor(*desc), target = tensor(*desc);
    ASSERT_TRUE(source) << source.error().message();
    ASSERT_TRUE(target) << target.error().message();
    input = std::move(*source); output = std::move(*target);
    const std::vector<float> sentinel(kElements, 99.0F);
    ASSERT_TRUE(output.memory.view().write(std::as_bytes(std::span(sentinel))).ok());
    auto made_builder = g::GraphBuilder::create();
    ASSERT_TRUE(made_builder) << made_builder.error().message();
    auto builder = std::move(*made_builder);
    auto source_id = builder.input(*desc);
    ASSERT_TRUE(source_id) << source_id.error().message();
    input_id = *source_id;
    auto result_id = builder.add(c::RmsNorm{.epsilon = kEpsilon}, input_id);
    ASSERT_TRUE(result_id) << result_id.error().message();
    output_id = *result_id;
    ASSERT_TRUE(builder.publish(output_id).ok());
    auto definition = std::move(builder).finish();
    ASSERT_TRUE(definition) << definition.error().message();
    graph = std::move(*definition);
    auto prepared = g::compile(graph, context,
        {.invocation_storage = g::InvocationStorageLimits{.max_bytes = 2U << 20}});
    ASSERT_TRUE(prepared) << prepared.error().message();
    compiled = std::move(*prepared);
    ASSERT_EQ(compiled.report.step_count(), 1u);
    const auto step = compiled.report.step(0);
    ASSERT_TRUE(step) << step.error().message();
    EXPECT_EQ(step->provider.id, c::native::kProviderId);
    ASSERT_TRUE(step->runtime.blocks_host);
    auto created = compiled.executable.create_invocation();
    ASSERT_TRUE(created) << created.error().message();
    run = std::move(*created);
    ASSERT_TRUE(run.bind(input_id, input.view).ok());
    ASSERT_TRUE(run.bind(output_id, output.view).ok());
    ASSERT_TRUE(run.prepare_replay(queues()).ok());
    ASSERT_TRUE(g::prepare_progress(run, scheduler, pool).ok());
  }
  void TearDown() override {
    executor.release();
    EXPECT_TRUE(scheduler.drain_for(2 * kTimeout));
    if (run.valid()) run.drain().IgnoreError();
    pool.shutdown();
    scheduler.shutdown();
  }
  std::array<g::QueueBinding, 1> queues() {
    return {g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  }
  void write_input(unsigned phase) {
    values.resize(kElements);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = float(int((i + phase * 7) % 61) - 30) / 16.0F;
    ASSERT_TRUE(input.memory.view().write(std::as_bytes(std::span(values))).ok());
  }
  void check_math() {
    std::vector<float> actual(kElements);
    ASSERT_TRUE(output.memory.view().read(std::as_writable_bytes(std::span(actual))).ok());
    for (std::size_t row = 0; row < kRows; ++row) {
      double squared = 0;
      for (std::size_t i = 0; i < kWidth; ++i) {
        const double value = values[row * kWidth + i];
        squared += value * value;
      }
      const double scale = 1.0 / std::sqrt(squared / double(kWidth) + double(kEpsilon));
      for (std::size_t i = 0; i < kWidth; ++i)
        ASSERT_NEAR(actual[row * kWidth + i], values[row * kWidth + i] * scale, 3.0e-6);
    }
  }
  void check_untouched() {
    std::vector<float> actual(kElements);
    ASSERT_TRUE(output.memory.view().read(std::as_writable_bytes(std::span(actual))).ok());
    for (const auto value : actual) ASSERT_EQ(value, 99.0F);
  }

  GateExecutor executor;
  std::atomic<bool> ordinary_done{false}, drain_started{false}, drain_done{false};
  hyle::Status drain_status;
  s::Scheduler scheduler{1, m::system_allocator(), 64};
  s::BlockingPool pool{m::system_allocator(), 1};
  c::Context context;
  c::Queue queue;
  g::Graph graph;
  g::CompiledGraph compiled;
  Tensor input, output;
  g::ValueId input_id, output_id;
  g::Invocation run;
  std::vector<float> values;
};

TEST_F(GraphProgress, OneWorkerReturnsPendingAndKeepsSchedulerAndDrainFiberAvailable) {
  for (unsigned phase = 0; phase != 2; ++phase) {
    SCOPED_TRACE(phase);
    write_input(phase);
    ordinary_done = false;
    executor.arm();
    auto submitted = run.submit(queues());
    ASSERT_TRUE(submitted) << submitted.error().message();
    ASSERT_TRUE(executor.wait_entered());
    EXPECT_EQ(submitted->poll().value(), c::CompletionPoll::kPending);
    EXPECT_EQ(pool.live_threads(), 1u);
    EXPECT_EQ(executor.scheduler_worker.load(), -1); // actual CPU body is offloaded
    if (phase == 1) {
      ASSERT_TRUE(scheduler.spawn_detached([this]() noexcept {
        drain_started.store(true, std::memory_order_release);
        drain_status = run.drain();
        drain_done.store(true, std::memory_order_release);
      }).ok());
      EXPECT_TRUE(eventually([&] { return drain_started.load(std::memory_order_acquire); }));
    }
    ASSERT_TRUE(scheduler.schedule([this]() noexcept {
      ordinary_done.store(true, std::memory_order_release);
    }).ok());
    EXPECT_TRUE(eventually([&] { return ordinary_done.load(std::memory_order_acquire); }));
    EXPECT_TRUE(executor.held()); // ordinary task ran while the real CPU body waited
    executor.release();
    if (phase == 1) {
      ASSERT_TRUE(eventually([&] { return drain_done.load(std::memory_order_acquire); }));
      EXPECT_TRUE(drain_status.ok()) << drain_status.message();
    }
    ASSERT_TRUE(observe(*submitted).ok());
    EXPECT_FALSE(run.in_flight());
    EXPECT_FALSE(executor.timeout());
    check_math();
    auto result = run.result(output_id);
    ASSERT_TRUE(result) << result.error().message();
    auto old = *result;
    if (phase == 0) {
      EXPECT_FALSE(run.submit(queues()));
      EXPECT_TRUE(run.reset().IsUnavailable());
      result->reset();
      EXPECT_FALSE(run.submit(queues()));
      old.reset();
      ASSERT_TRUE(run.bind(input_id, input.view).ok());
      ASSERT_TRUE(run.prepare_replay(queues()).ok());
    }
  }
}

TEST_F(GraphProgress, ClosedSchedulerRejectsBeforeNativeAcceptanceAndLeavesSlotReusable) {
  scheduler.begin_shutdown();
  auto submitted = run.submit(queues());
  ASSERT_FALSE(submitted);
  EXPECT_TRUE(submitted.error().IsCancelled());
  EXPECT_EQ(executor.calls.load(), 0u);
  check_untouched();
  EXPECT_FALSE(run.in_flight());
  EXPECT_FALSE(run.result(output_id));
  EXPECT_TRUE(run.reset().ok());
  EXPECT_TRUE(run.bind(input_id, input.view).ok());
  EXPECT_TRUE(run.close().ok());
  EXPECT_TRUE(run.drain().IsCancelled());
}

TEST_F(GraphProgress, ClosedPoolCancelsBeforeNativeAcceptanceAndLeavesSlotReusable) {
  pool.shutdown();
  auto submitted = run.submit(queues());
  ASSERT_TRUE(submitted) << submitted.error().message();
  EXPECT_TRUE(observe(*submitted).IsCancelled());
  ASSERT_TRUE(eventually([&] { return !run.in_flight(); }));
  EXPECT_EQ(executor.calls.load(), 0u);
  check_untouched();
  EXPECT_FALSE(run.result(output_id));
  EXPECT_TRUE(run.reset().ok());
  EXPECT_TRUE(run.bind(input_id, input.view).ok());
  EXPECT_TRUE(run.close().ok());
  EXPECT_TRUE(run.drain().IsCancelled());
}

TEST_F(GraphProgress, CancellationAndPublicAbandonKeepRealCpuWorkAndOwnersAlive) {
  write_input(3);
  executor.arm();
  auto submitted = run.submit(queues());
  ASSERT_TRUE(submitted) << submitted.error().message();
  ASSERT_TRUE(executor.wait_entered());
  EXPECT_EQ(submitted->poll().value(), c::CompletionPoll::kPending);
  EXPECT_TRUE(submitted->request_cancel().ok());
  EXPECT_TRUE(run.close().IsUnavailable());
  EXPECT_FALSE(run.result(output_id));
  const auto live = compiled.executable.invocation_storage_usage();
  ASSERT_TRUE(live);
  ASSERT_GT(live->live_bytes, 0u);
  run = {};
  EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, live->live_bytes);
  executor.release();
  EXPECT_TRUE(observe(*submitted).IsCancelled());
  ASSERT_TRUE(scheduler.drain_for(kTimeout));
  EXPECT_FALSE(executor.timeout());
  check_math(); // cancellation does not pretend to preempt an admitted CPU body
  EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, 0u);
}

} // namespace
