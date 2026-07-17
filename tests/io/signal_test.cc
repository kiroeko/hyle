#include <gtest/gtest.h>

#include <atomic>
#include <memory>

#include "hyle/io/io_context.h"
#include "hyle/io/signal.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if !defined(_WIN32)
#  include <csignal>
#endif

// io-09 signal_set test. On POSIX we drive the whole path end-to-end: a fiber
// co_await sigs.wait(), the test raise()s the signal, the process-global handler
// writes the self-pipe, the proactor read completes, and wait() returns the bit.
// On Windows a real console control event would target the whole process (and can
// kill the test runner), so there we only smoke-test that construction/teardown
// installs+removes the handler without crashing -- the real Ctrl-C delivery is an
// integration concern, not a unit test.

using hyle::io::IoContext;
using hyle::io::kSigInt;
using hyle::io::kSigTerm;
using hyle::io::SignalSet;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {
constexpr unsigned kOwner = 0;
struct Drive {
  Task<int> task;
};
void drive_entry(void* p) { static_cast<Drive*>(p)->task.resume(); }

#if !defined(_WIN32)
struct SigR {
  std::atomic<int> got{-999};
  std::atomic<bool> done{false};
};
// Waiter fiber: co_await the signal; record which bit fired.
Task<int> wait_coro(Allocator, SigR* r, SignalSet* s) {
  int bit = co_await s->wait();
  r->got.store(bit, std::memory_order_release);
  r->done.store(true, std::memory_order_release);
  co_return 0;
}
// Raiser fiber: fire SIGTERM (runs after the waiter has armed the pipe read).
Task<int> raise_coro(Allocator) {
  ::raise(SIGTERM);
  co_return 0;
}
#endif
}  // namespace

#if !defined(_WIN32)

// A fiber waits on SIGTERM; another fiber raises it; wait() must return kSigTerm.
TEST(IoSignal, WaitReturnsRaisedSignal) {
  constexpr unsigned kWorkers = 1;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());

  auto r = std::make_unique<SigR>();
  auto sigs = std::make_unique<SignalSet>(io, kOwner, kSigTerm);
  ASSERT_TRUE(sigs->valid());

  // Waiter fiber: arm the wait, then a raiser fiber (spawned after) fires SIGTERM.
  auto dw = std::make_unique<Drive>();
  dw->task = wait_coro(system_allocator(), r.get(), sigs.get());
  Fiber fw(kDefaultStackSize, &drive_entry, dw.get());

  // Raiser fiber: raise SIGTERM. Runs on the same worker (pinned) but after the
  // waiter has suspended on the pipe read (spawned second, FIFO local queue).
  auto dr = std::make_unique<Drive>();
  dr->task = raise_coro(system_allocator());
  Fiber fr(kDefaultStackSize, &drive_entry, dr.get());

  sched.spawn_on(kOwner, &fw);
  sched.spawn_on(kOwner, &fr);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->done.load());
  EXPECT_EQ(r->got.load(), static_cast<int>(kSigTerm));
}

// Registering SIGINT + SIGTERM then destroying restores default disposition
// without leaving a dangling handler (a subsequent default-terminate SIGINT would
// otherwise be swallowed). Smoke: build/destroy several sets, no crash/hang.
TEST(IoSignal, RegisterUnregisterIsClean) {
  constexpr unsigned kWorkers = 1;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());
  {
    SignalSet a(io, kOwner, kSigInt | kSigTerm);
    EXPECT_TRUE(a.valid());
    SignalSet b(io, kOwner, kSigInt);  // overlapping registration (refcounted)
    EXPECT_TRUE(b.valid());
  }
  // Both dropped -> handlers restored. Nothing to assert beyond "no crash".
  SUCCEED();
}

#else  // Windows

// Windows: a real CTRL event targets the whole process group and can terminate the
// test runner, so we only verify the handler install/uninstall path is clean.
TEST(IoSignal, RegisterUnregisterIsCleanWindows) {
  constexpr unsigned kWorkers = 1;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());
  {
    SignalSet a(io, kOwner, kSigInt | kSigTerm);
    EXPECT_TRUE(a.valid());
    SignalSet b(io, kOwner, kSigTerm);
    EXPECT_TRUE(b.valid());
  }
  SUCCEED();
}

#endif
