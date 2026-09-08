#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>

#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

// sched-07 consumer: HTTP-request fan-out cancellation. This is the shape io-10
// drives (the reason sched-07 waited for a real consumer): a request handler spawns
// SEVERAL child tasks (upstream calls / sub-queries); cancelling the request (a
// deadline, or the client hanging up) must connectedly cancel the WHOLE subtree the
// handler spawned. We model it directly on the scheduler (no socket needed -- the
// tree/scope mechanics are the point; net_cancel_test covers the io leg): a request
// root token derives one child per fan-out task; each task installs a CancelScope
// and parks; cancelling the request root wakes them to observe cancellation, and a
// when_all over the group returns once all have unwound.

using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::CancelScope;
using hyle::sched::CancelToken;
using hyle::sched::CaptureWaker;
using hyle::sched::is_cancel_requested;
using hyle::sched::JoinHandle;
using hyle::sched::Scheduler;
using hyle::sched::spawn_task;
using hyle::sched::SuspendFiber;
using hyle::sched::Task;
using hyle::sched::Waker;
using hyle::sched::when_all;

namespace {

// One fan-out subtask: install the request-derived cancel scope, publish a waker,
// park, then on wake report whether it observed the request's cancellation. Returns
// 1 if cancellation was observed (the connected-cancel succeeded), else 0.
struct SubBox {
  CancelToken token;             // derived from the request root
  std::atomic<bool> ready{false};
  Waker waker;
  std::atomic<int> observed{-1};
};

Task<int> fanout_subtask(AllocatorRef, SubBox* box) {
  CancelScope scope{box->token};  // opt into the request's cancellation
  box->waker = co_await CaptureWaker{};
  box->ready.store(true, std::memory_order_release);
  co_await SuspendFiber{};                       // simulate an in-flight upstream call
  co_return is_cancel_requested() ? 1 : 0;       // observed the connected cancel?
}

}  // namespace

// A request handler fans out N subtasks (each with a request-derived token), then the
// request is cancelled (deadline/hangup). Every subtask must observe the cancellation
// -- the tree connected one request cancel to the whole spawned subtree.
TEST(HttpCancel, RequestCancelPropagatesToFanoutSubtasks) {
  constexpr int kFanout = 4;
  Scheduler sched(4, system_allocator(), 256);

  // Per-request cancellation root (in a real server this hangs off the connection /
  // is cancelled by the request deadline or client close).
  CancelToken request = CancelToken::root(system_allocator());

  std::unique_ptr<SubBox> boxes[kFanout];
  JoinHandle<int> subs[kFanout];
  for (int i = 0; i < kFanout; ++i) {
    boxes[i] = std::make_unique<SubBox>();
    boxes[i]->token = request.derive();  // child token per fan-out branch
    subs[i] = spawn_task(sched, system_allocator(),
                         fanout_subtask(system_allocator(), boxes[i].get()));
  }
  // Wait until every subtask is parked (in-flight upstream call).
  for (int i = 0; i < kFanout; ++i)
    while (!boxes[i]->ready.load(std::memory_order_acquire)) {}

  // The request is cancelled: ONE cancel cascades to all derived child tokens.
  request.cancel();
  // Wake each parked subtask so it reaches its is_cancel_requested() check + unwinds.
  for (int i = 0; i < kFanout; ++i) boxes[i]->waker.wake();

  // when_all: block until every fan-out branch has finished (connected-cancel done).
  when_all(subs[0], subs[1], subs[2], subs[3]);
  for (int i = 0; i < kFanout; ++i) {
    auto r = subs[i].join();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 1) << "fan-out subtask " << i << " missed the request cancel";
  }
  sched.shutdown();
}

// Cancelling ONE branch's subtree does not disturb sibling branches (a per-upstream
// timeout cancels only that upstream call, not the whole request).
TEST(HttpCancel, PerBranchCancelIsIsolated) {
  Scheduler sched(4, system_allocator(), 256);
  CancelToken request = CancelToken::root(system_allocator());
  auto b0 = std::make_unique<SubBox>();
  auto b1 = std::make_unique<SubBox>();
  b0->token = request.derive();
  b1->token = request.derive();
  auto s0 = spawn_task(sched, system_allocator(), fanout_subtask(system_allocator(), b0.get()));
  auto s1 = spawn_task(sched, system_allocator(), fanout_subtask(system_allocator(), b1.get()));
  while (!b0->ready.load(std::memory_order_acquire)) {}
  while (!b1->ready.load(std::memory_order_acquire)) {}

  b0->token.cancel();       // cancel only branch 0's subtree
  b0->waker.wake();
  b1->waker.wake();
  when_all(s0, s1);
  EXPECT_EQ(*s0.join(), 1);  // branch 0 observed its cancel
  EXPECT_EQ(*s1.join(), 0);  // branch 1 did NOT (isolated)
  sched.shutdown();
}
