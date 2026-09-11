#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string_view>

#include "hyle/container/bounded_queue.h"
#include "hyle/error/status.h"
#include "hyle/mem/allocator.h"
#include "hyle/obs/metrics.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/timer.h"

namespace {

using namespace std::chrono_literals;

struct AsyncJourneyState {
  hyle::container::BoundedQueue<int>* queue = nullptr;
  hyle::sched::CancelToken* cancel = nullptr;
  hyle::obs::ConcurrentCounter<>* cancelled = nullptr;
  hyle::Status status{};
  int value = 0;
  std::atomic<bool> finished{false};
};

void consume_after_deadline(void* raw) noexcept {
  auto& state = *static_cast<AsyncJourneyState*>(raw);
  hyle::sched::CancelScope scope{*state.cancel};
  hyle::sched::sleep_for(20ms);
  if (!state.queue->try_pop(state.value)) {
    state.status = hyle::Status::Internal("scheduled input disappeared");
  } else if (hyle::sched::is_cancel_requested()) {
    state.status =
        hyle::Status::Cancelled("request budget ended")
            .WithContext("async inference work item");
    state.cancelled->add();
  }
  state.finished.store(true, std::memory_order_release);
}

void cancel_soon(void* raw) noexcept {
  auto& token = *static_cast<hyle::sched::CancelToken*>(raw);
  hyle::sched::sleep_for(1ms);
  token.cancel();
}

}  // namespace

TEST(CorrectnessJourney, AsyncRuntimePropagatesCancelErrorDrainAndMetrics) {
  hyle::container::BoundedQueue<int> queue(4);
  ASSERT_TRUE(queue.valid());
  ASSERT_TRUE(queue.try_push(42));

  auto token = hyle::sched::CancelToken::try_create_root(
      hyle::mem::system_allocator());
  ASSERT_TRUE(token.has_value()) << token.error().message();
  hyle::obs::ConcurrentCounter<> cancelled;
  AsyncJourneyState state{&queue, &*token, &cancelled};

  hyle::sched::Scheduler scheduler(
      2, hyle::mem::system_allocator(), 64);
  hyle::sched::Fiber consumer(hyle::sched::kDefaultStackSize,
                              &consume_after_deadline, &state);
  hyle::sched::Fiber canceller(hyle::sched::kDefaultStackSize,
                               &cancel_soon, &*token);
  ASSERT_TRUE(scheduler.spawn_on(0, &consumer));
  ASSERT_TRUE(scheduler.spawn_on(1, &canceller));
  scheduler.drain();

  EXPECT_TRUE(state.finished.load(std::memory_order_acquire));
  EXPECT_EQ(state.value, 42);
  EXPECT_TRUE(state.status.IsCancelled()) << state.status.message();
  EXPECT_NE(state.status.message().find("async inference work item"),
            std::string_view::npos);
  EXPECT_EQ(cancelled.value(), 1u);
  int unexpected = 0;
  EXPECT_FALSE(queue.try_pop(unexpected));

  scheduler.shutdown();
  EXPECT_FALSE(scheduler.accepting());
}
