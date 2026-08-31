#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"
#include "hyle/util/spin_lock.h"

using hyle::mem::system_allocator;
using hyle::sched::Barrier;
using hyle::sched::Channel;
using hyle::sched::ChannelSendFailure;
using hyle::sched::ConditionVariable;
using hyle::sched::Event;
using hyle::sched::EventResetMode;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Latch;
using hyle::sched::make_broadcast;
using hyle::sched::make_oneshot;
using hyle::sched::make_watch;
using hyle::sched::Mutex;
using hyle::sched::RwLock;
using hyle::sched::Scheduler;
using hyle::sched::Semaphore;
using hyle::sched::WaitGroup;
using hyle::util::SpinLock;
using namespace std::chrono_literals;

namespace {
void *fail_alloc(void *, std::size_t, std::size_t) noexcept { return nullptr; }
void fail_free(void *, void *, std::size_t, std::size_t) noexcept {}
void *fail_realloc(void *, void *, std::size_t, std::size_t,
                   std::size_t) noexcept {
  return nullptr;
}

struct ThrowOnCopy {
  explicit ThrowOnCopy(int value = 0) noexcept : value(value) {}
  ThrowOnCopy(const ThrowOnCopy &other) : value(other.value) {
    if (throw_copy)
      throw std::runtime_error("copy failed");
  }
  ThrowOnCopy(ThrowOnCopy &&) noexcept = default;
  ThrowOnCopy &operator=(ThrowOnCopy &&) noexcept = default;

  static inline bool throw_copy = false;
  int value;
};

TEST(Sync, AllocationFailuresAreRecoverable) {
  hyle::mem::Allocator failing(&fail_alloc, &fail_free, &fail_realloc, nullptr);
  hyle::Status channel_status;
  Channel<int> channel(failing, 4, &channel_status);
  EXPECT_FALSE(channel.valid());
  EXPECT_TRUE(channel_status.IsResourceExhausted());
  auto unsent = channel.send(1);
  ASSERT_FALSE(unsent.has_value());
  EXPECT_EQ(unsent.error().reason(), ChannelSendFailure::kInvalid);
  EXPECT_EQ(unsent.error().value(), 1);
  int out = 0;
  EXPECT_FALSE(channel.recv(out));
  EXPECT_FALSE(hyle::sched::try_make_oneshot<int>(failing).has_value());
  EXPECT_FALSE(hyle::sched::try_make_watch<int>(failing, 1).has_value());
  EXPECT_FALSE(hyle::sched::try_make_broadcast<int>(failing, 4).has_value());

  hyle::Status invalid_capacity;
  Channel<int> invalid(system_allocator(), 0, &invalid_capacity);
  EXPECT_FALSE(invalid.valid());
  EXPECT_TRUE(invalid_capacity.IsInvalidArgument());
}

// Spawn `bodies.size()` fibers on a `workers`-wide scheduler and block until
// all of them finish. Fibers outlive the scheduler (drain() returns only once
// every fiber is done), so a stack vector of Fibers is safe.
template <class Fn>
void run_fibers(unsigned workers, int count, Fn entry, void *arg) {
  std::vector<std::unique_ptr<Fiber>> fibers;
  fibers.reserve(count);
  for (int i = 0; i < count; ++i) {
    fibers.push_back(std::make_unique<Fiber>(kDefaultStackSize, entry, arg));
  }
  Scheduler sched(workers, system_allocator(), 1024);
  for (auto &f : fibers)
    sched.spawn(f.get());
  sched.drain();
}

// ---- Mutex ------------------------------------------------------------------

struct MutexArg {
  Mutex m;
  long counter =
      0; // plain (non-atomic): a broken mutex => TSan race + wrong sum
  int iters;
};

void mutex_body(void *p) {
  auto *a = static_cast<MutexArg *>(p);
  for (int i = 0; i < a->iters; ++i) {
    a->m.lock();
    ++a->counter; // critical section
    a->m.unlock();
  }
}

// Mutual exclusion under fiber contention: 8 fibers x N increments of a shared
// non-atomic counter must total exactly 8*N. A lost or overlapping critical
// section both corrupts the count AND trips TSan.
TEST(Sync, MutexMutualExclusion) {
  constexpr int kFibers = 8;
  MutexArg a;
  a.iters = 1000;
  run_fibers(4, kFibers, &mutex_body, &a);
  EXPECT_EQ(a.counter, static_cast<long>(kFibers) * a.iters);
}

// ---- WaitGroup --------------------------------------------------------------

struct WgArg {
  WaitGroup wg;
  std::atomic<int> done_count{0};
  int n;                             // number of worker fibers
  std::atomic<int> seen_at_wait{-1}; // done_count observed when wait() returned
};

void wg_worker(void *p) {
  auto *a = static_cast<WgArg *>(p);
  a->done_count.fetch_add(1, std::memory_order_acq_rel); // "work happened"
  a->wg.done();
}
void wg_coordinator(void *p) {
  auto *a = static_cast<WgArg *>(p);
  a->wg.wait(); // returns only after all n done()
  a->seen_at_wait.store(a->done_count.load(std::memory_order_acquire),
                        std::memory_order_release);
}

// wait() returns only after every done() -- so the coordinator must observe the
// full count. Mixes one coordinator fiber with n worker fibers.
TEST(Sync, WaitGroupWaitsForAll) {
  constexpr int kN = 16;
  WgArg a;
  a.n = kN;
  a.wg.add(kN);
  // Build coordinator + workers as a single fiber set sharing `a`.
  std::vector<std::unique_ptr<Fiber>> fibers;
  fibers.push_back(
      std::make_unique<Fiber>(kDefaultStackSize, &wg_coordinator, &a));
  for (int i = 0; i < kN; ++i) {
    fibers.push_back(
        std::make_unique<Fiber>(kDefaultStackSize, &wg_worker, &a));
  }
  Scheduler sched(4, system_allocator(), 1024);
  for (auto &f : fibers)
    sched.spawn(f.get());
  sched.drain();
  EXPECT_EQ(a.seen_at_wait.load(), kN);
}

// ---- Latch ------------------------------------------------------------------

struct LatchArg {
  Latch *latch;
  std::atomic<int> resumed{0}; // waiters that passed the latch
};

void latch_counter(void *p) {
  auto *a = static_cast<LatchArg *>(p);
  a->latch->count_down();
}
void latch_waiter(void *p) {
  auto *a = static_cast<LatchArg *>(p);
  a->latch->wait();
  a->resumed.fetch_add(1, std::memory_order_acq_rel);
}

// A latch opened by N count_downs releases all M waiters (fiber-aware).
TEST(Sync, LatchReleasesAllWaiters) {
  constexpr int kCount = 8;
  constexpr int kWaiters = 8;
  Latch latch(kCount);
  LatchArg a;
  a.latch = &latch;
  std::vector<std::unique_ptr<Fiber>> fibers;
  for (int i = 0; i < kWaiters; ++i)
    fibers.push_back(
        std::make_unique<Fiber>(kDefaultStackSize, &latch_waiter, &a));
  for (int i = 0; i < kCount; ++i)
    fibers.push_back(
        std::make_unique<Fiber>(kDefaultStackSize, &latch_counter, &a));
  Scheduler sched(4, system_allocator(), 1024);
  for (auto &f : fibers)
    sched.spawn(f.get());
  sched.drain();
  EXPECT_EQ(a.resumed.load(), kWaiters);
}

TEST(Sync, LatchArriveAndWaitCombinesCountdownWithWait) {
  Latch latch(2);
  std::atomic<bool> returned{false};
  std::thread participant([&] {
    latch.arrive_and_wait();
    returned.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(2ms);
  EXPECT_FALSE(returned.load(std::memory_order_acquire));
  latch.count_down();
  participant.join();
  EXPECT_TRUE(returned.load(std::memory_order_acquire));
}

namespace {
struct EventWaitArg {
  Event *event;
  std::atomic<int> *entered;
  std::atomic<int> *returned;
};

void event_waiter(void *raw) {
  auto *arg = static_cast<EventWaitArg *>(raw);
  arg->entered->fetch_add(1, std::memory_order_release);
  arg->event->wait();
  arg->returned->fetch_add(1, std::memory_order_release);
}
} // namespace

TEST(Sync, ManualEventBroadcastsAndStaysSignaledUntilReset) {
  Event event(EventResetMode::kManual);
  std::atomic<int> entered{0};
  std::atomic<int> returned{0};
  EventWaitArg arg{&event, &entered, &returned};
  Fiber fibers[] = {
      Fiber(kDefaultStackSize, &event_waiter, &arg),
      Fiber(kDefaultStackSize, &event_waiter, &arg),
      Fiber(kDefaultStackSize, &event_waiter, &arg),
  };
  Scheduler scheduler(2, system_allocator(), 64);
  for (auto &fiber : fibers)
    ASSERT_TRUE(scheduler.spawn(&fiber));
  while (entered.load(std::memory_order_acquire) != 3)
    std::this_thread::yield();
  event.signal();
  scheduler.drain();
  EXPECT_EQ(returned.load(std::memory_order_acquire), 3);
  EXPECT_TRUE(event.is_signaled());
  EXPECT_TRUE(event.wait_for(hyle::time::Duration::zero()));
  event.reset();
  EXPECT_FALSE(event.is_signaled());
  EXPECT_FALSE(event.wait_for(1ms));
}

TEST(Sync, AutoEventReleasesOneWaiterPerSignal) {
  Event event(EventResetMode::kAuto);
  std::atomic<int> entered{0};
  std::atomic<int> returned{0};
  EventWaitArg arg{&event, &entered, &returned};
  Fiber first(kDefaultStackSize, &event_waiter, &arg);
  Fiber second(kDefaultStackSize, &event_waiter, &arg);
  Scheduler scheduler(2, system_allocator(), 64);
  ASSERT_TRUE(scheduler.spawn(&first));
  ASSERT_TRUE(scheduler.spawn(&second));
  while (entered.load(std::memory_order_acquire) != 2)
    std::this_thread::yield();
  event.signal();
  while (returned.load(std::memory_order_acquire) != 1)
    std::this_thread::yield();
  EXPECT_FALSE(event.is_signaled());
  event.signal();
  scheduler.drain();
  EXPECT_EQ(returned.load(std::memory_order_acquire), 2);
  EXPECT_FALSE(event.is_signaled());
}

TEST(Sync, BarrierIsReusableAndElectsOnePhaseCompleter) {
  constexpr int kParticipants = 4;
  constexpr int kPhases = 50;
  Barrier barrier(kParticipants);
  std::atomic<int> leaders{0};
  std::vector<std::thread> threads;
  for (int participant = 0; participant < kParticipants; ++participant) {
    threads.emplace_back([&] {
      for (int phase = 0; phase < kPhases; ++phase) {
        if (barrier.arrive_and_wait())
          leaders.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  EXPECT_EQ(leaders.load(std::memory_order_relaxed), kPhases);
  EXPECT_EQ(barrier.participants(), kParticipants);
}

TEST(Sync, BarrierArriveAndDropReducesFuturePhase) {
  Barrier barrier(3);
  std::atomic<int> passed{0};
  std::thread dropping([&] {
    barrier.arrive_and_drop();
    passed.fetch_add(1, std::memory_order_relaxed);
  });
  std::thread participant([&] {
    barrier.arrive_and_wait();
    barrier.arrive_and_wait();
    passed.fetch_add(1, std::memory_order_relaxed);
  });
  barrier.arrive_and_wait();
  dropping.join();
  barrier.arrive_and_wait();
  participant.join();
  EXPECT_EQ(passed.load(std::memory_order_relaxed), 2);
  EXPECT_EQ(barrier.participants(), 2u);
}

// ---- Channel ----------------------------------------------------------------

struct ChanArg {
  Channel<int> *ch;
  int n;                     // values to produce
  std::atomic<long> sum{0};  // sum of received values
  std::atomic<int> count{0}; // number received
};

void chan_producer(void *p) {
  auto *a = static_cast<ChanArg *>(p);
  for (int i = 0; i < a->n; ++i)
    (void)a->ch->send(i);
  a->ch->close();
}
void chan_consumer(void *p) {
  auto *a = static_cast<ChanArg *>(p);
  int v;
  while (a->ch->recv(v)) {
    a->sum.fetch_add(v, std::memory_order_acq_rel);
    a->count.fetch_add(1, std::memory_order_acq_rel);
  }
}

// One producer fills a small bounded channel (forcing the producer to block
// when full), one consumer drains it (blocking when empty); close() ends the
// stream.
TEST(Sync, ChannelProducerConsumer) {
  constexpr int kN = 1000;
  Channel<int> ch(system_allocator(),
                  4); // tiny cap => exercises full/empty waits
  ChanArg a;
  a.ch = &ch;
  a.n = kN;
  std::unique_ptr<Fiber> prod =
      std::make_unique<Fiber>(kDefaultStackSize, &chan_producer, &a);
  std::unique_ptr<Fiber> cons =
      std::make_unique<Fiber>(kDefaultStackSize, &chan_consumer, &a);
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(prod.get());
  sched.spawn(cons.get());
  sched.drain();
  EXPECT_EQ(a.count.load(), kN);
  EXPECT_EQ(a.sum.load(), static_cast<long>(kN) * (kN - 1) / 2);
}

TEST(Sync, ChannelTryAndTimedPathsDistinguishFullEmptyAndClosed) {
  using Ch = Channel<int>;
  Ch ch(system_allocator(), 1);
  int out = -1;
  EXPECT_EQ(ch.try_recv(out), Ch::RecvStatus::kEmptyOrTimeout);
  EXPECT_EQ(ch.recv_for(out, 1ms), Ch::RecvStatus::kEmptyOrTimeout);
  EXPECT_TRUE(ch.try_send(7).has_value());
  auto full = ch.try_send(8);
  ASSERT_FALSE(full.has_value());
  EXPECT_EQ(full.error().reason(), ChannelSendFailure::kFull);
  EXPECT_EQ(full.error().value(), 8);
  auto timed_out = ch.send_for(9, 1ms);
  ASSERT_FALSE(timed_out.has_value());
  EXPECT_EQ(timed_out.error().reason(), ChannelSendFailure::kTimeout);
  EXPECT_EQ(timed_out.error().value(), 9);
  EXPECT_EQ(ch.try_recv(out), Ch::RecvStatus::kValue);
  EXPECT_EQ(out, 7);
  ch.close();
  auto closed_send = ch.try_send(10);
  ASSERT_FALSE(closed_send.has_value());
  EXPECT_EQ(closed_send.error().reason(), ChannelSendFailure::kClosed);
  EXPECT_EQ(closed_send.error().value(), 10);
  EXPECT_EQ(ch.try_recv(out), Ch::RecvStatus::kClosed);
}

TEST(Sync, ChannelNaturalReceiveReturnsTypedErrorsAndMoveOnlyValue) {
  struct MoveOnly {
    explicit MoveOnly(int value) noexcept : value(value) {}
    MoveOnly(MoveOnly &&) noexcept = default;
    MoveOnly &operator=(MoveOnly &&) = delete;
    int value;
  };

  Channel<MoveOnly> ch(system_allocator(), 1);
  auto empty = ch.try_recv();
  ASSERT_FALSE(empty.has_value());
  EXPECT_TRUE(empty.error().IsNotFound());
  EXPECT_TRUE(ch.try_send(MoveOnly(7)).has_value());
  auto value = ch.recv();
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(value->value, 7);
  auto timeout = ch.recv_for(1ms);
  ASSERT_FALSE(timeout.has_value());
  EXPECT_TRUE(timeout.error().IsDeadlineExceeded());
  ch.close();
  auto closed = ch.recv();
  ASSERT_FALSE(closed.has_value());
  EXPECT_TRUE(closed.error().IsCancelled());
}

namespace {
// Multi-producer/consumer channel, closed once all producers finish (via a
// WaitGroup) -- composes Channel + WaitGroup, the convergence thesis in action.
struct ChanMpmcArg {
  Channel<int> *ch;
  WaitGroup *producers_done;
  int per_producer;
  std::atomic<long> sum{0};
  std::atomic<int> count{0};
};
void mpmc_producer(void *p) {
  auto *a = static_cast<ChanMpmcArg *>(p);
  for (int i = 0; i < a->per_producer; ++i)
    (void)a->ch->send(1);
  a->producers_done->done();
}
void mpmc_closer(void *p) {
  auto *a = static_cast<ChanMpmcArg *>(p);
  a->producers_done->wait(); // all producers finished
  a->ch->close();
}
void mpmc_consumer(void *p) {
  auto *a = static_cast<ChanMpmcArg *>(p);
  int v;
  while (a->ch->recv(v)) {
    a->sum.fetch_add(v, std::memory_order_acq_rel);
    a->count.fetch_add(1, std::memory_order_acq_rel);
  }
}
} // namespace

TEST(Sync, ChannelMultiProducerMultiConsumer) {
  constexpr int kProducers = 4;
  constexpr int kConsumers = 4;
  constexpr int kPerProducer = 500;
  Channel<int> ch(system_allocator(), 8);
  WaitGroup producers_done;
  producers_done.add(kProducers);
  ChanMpmcArg a;
  a.ch = &ch;
  a.producers_done = &producers_done;
  a.per_producer = kPerProducer;

  std::vector<std::unique_ptr<Fiber>> fibers;
  for (int i = 0; i < kProducers; ++i)
    fibers.push_back(
        std::make_unique<Fiber>(kDefaultStackSize, &mpmc_producer, &a));
  for (int i = 0; i < kConsumers; ++i)
    fibers.push_back(
        std::make_unique<Fiber>(kDefaultStackSize, &mpmc_consumer, &a));
  fibers.push_back(
      std::make_unique<Fiber>(kDefaultStackSize, &mpmc_closer, &a));
  Scheduler sched(4, system_allocator(), 1024);
  for (auto &f : fibers)
    sched.spawn(f.get());
  sched.drain();
  EXPECT_EQ(a.count.load(), kProducers * kPerProducer);
  EXPECT_EQ(a.sum.load(), static_cast<long>(kProducers) * kPerProducer);
}

// ---- ConditionVariable: notify_all stress (all fibers) ----------------------

struct CvBroadcastArg {
  SpinLock mu;
  ConditionVariable cv;
  bool go = false;
  std::atomic<int> woke{0};
};

void cv_waiter(void *p) {
  auto *a = static_cast<CvBroadcastArg *>(p);
  a->mu.lock();
  while (!a->go)
    a->cv.wait(a->mu); // releases mu, yields, re-acquires
  a->mu.unlock();
  a->woke.fetch_add(1, std::memory_order_acq_rel);
}
void cv_broadcaster(void *p) {
  auto *a = static_cast<CvBroadcastArg *>(p);
  a->mu.lock();
  a->go = true;
  a->mu.unlock();
  a->cv.notify_all();
}

// N fibers block on one CV; a single notify_all (after setting the predicate)
// must release every one. The predicate-under-monitor-lock discipline closes
// the lost-wakeup window regardless of broadcaster-vs-waiter ordering.
TEST(Sync, CvNotifyAllWakesAllFibers) {
  constexpr int kWaiters = 200;
  CvBroadcastArg a;
  std::vector<std::unique_ptr<Fiber>> fibers;
  for (int i = 0; i < kWaiters; ++i)
    fibers.push_back(
        std::make_unique<Fiber>(kDefaultStackSize, &cv_waiter, &a));
  fibers.push_back(
      std::make_unique<Fiber>(kDefaultStackSize, &cv_broadcaster, &a));
  Scheduler sched(4, system_allocator(), 1024);
  for (auto &f : fibers)
    sched.spawn(f.get());
  sched.drain();
  EXPECT_EQ(a.woke.load(), kWaiters);
}

// ---- ConditionVariable: dual-mode (fiber <-> bare thread) -------------------

struct CrossArg {
  SpinLock mu;
  ConditionVariable cv;
  bool ready = false;
  std::atomic<bool> resumed{false};
};

void cross_notifier_fiber(void *p) {
  auto *a = static_cast<CrossArg *>(p);
  a->mu.lock();
  a->ready = true;
  a->mu.unlock();
  a->cv.notify_all();
}

// THREAD-mode waiter: a bare (non-worker) thread blocks on the CV and is woken
// by a fiber's notify. This is the cross-process / external-submit path -- the
// CV must fall back to Parker, not assume the waiter is a fiber.
TEST(Sync, CvThreadWaiterWokenByFiber) {
  CrossArg a;
  std::thread waiter([&a] {
    a.mu.lock();
    while (!a.ready)
      a.cv.wait(a.mu); // bare thread => thread-mode park
    a.mu.unlock();
    a.resumed.store(true, std::memory_order_release);
  });
  std::unique_ptr<Fiber> notifier =
      std::make_unique<Fiber>(kDefaultStackSize, &cross_notifier_fiber, &a);
  {
    Scheduler sched(2, system_allocator(), 256);
    sched.spawn(notifier.get());
    sched.drain();
  }
  waiter.join();
  EXPECT_TRUE(a.resumed.load());
}

namespace {
void cv_fiber_waiter(void *p) {
  auto *a = static_cast<CrossArg *>(p);
  a->mu.lock();
  while (!a->ready)
    a->cv.wait(a->mu); // worker fiber => fiber-mode suspend
  a->mu.unlock();
  a->resumed.store(true, std::memory_order_release);
}
} // namespace

// FIBER-mode waiter woken by a bare (main) thread's notify -- the inverse path.
TEST(Sync, CvFiberWaiterWokenByThread) {
  CrossArg a;
  std::unique_ptr<Fiber> waiter =
      std::make_unique<Fiber>(kDefaultStackSize, &cv_fiber_waiter, &a);
  {
    Scheduler sched(2, system_allocator(), 256);
    sched.spawn(waiter.get());
    // Notify from THIS (main, non-worker) thread, racing the fiber's suspend.
    a.mu.lock();
    a.ready = true;
    a.mu.unlock();
    a.cv.notify_all();
    sched.drain();
  }
  EXPECT_TRUE(a.resumed.load());
}

// ---- timed wait (thread + fiber modes) --------------------------------------

using hyle::sched::ConditionVariable;
using CvStatus = ConditionVariable::CvStatus;

// wait_for returns kTimeout when no notify arrives within the deadline, and it
// actually blocked for roughly the requested time (not a busy return).
TEST(SyncTimed, WaitForTimesOut) {
  ConditionVariable cv;
  SpinLock mu;
  auto t0 = std::chrono::steady_clock::now();
  CvStatus st;
  {
    std::lock_guard<SpinLock> lk(mu);
    st = cv.wait_for(mu, 30ms); // nobody notifies
  }
  auto elapsed = std::chrono::steady_clock::now() - t0;
  EXPECT_EQ(st, CvStatus::kTimeout);
  EXPECT_GE(elapsed, std::chrono::milliseconds(20)); // really slept
}

// wait_for returns kNoTimeout when a notify arrives before the deadline.
TEST(SyncTimed, WaitForNotifiedBeforeDeadline) {
  ConditionVariable cv;
  SpinLock mu;
  std::atomic<bool> parked{false};
  std::atomic<CvStatus> st{CvStatus::kTimeout};

  std::thread waiter([&] {
    std::lock_guard<SpinLock> lk(mu);
    parked.store(true, std::memory_order_release);
    st.store(cv.wait_for(mu, 5s), // we notify well before
             std::memory_order_release);
  });
  while (!parked.load(std::memory_order_acquire))
    std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  {
    std::lock_guard<SpinLock> lk(mu);
    cv.notify_one();
  }
  waiter.join();
  EXPECT_EQ(st.load(), CvStatus::kNoTimeout);
}

// Predicate wait_for: returns true when the predicate becomes true before the
// deadline (via a notify), false purely on timeout.
TEST(SyncTimed, WaitForPredicate) {
  ConditionVariable cv;
  SpinLock mu;
  bool ready = false;

  // Case A: predicate satisfied by a notify before the deadline.
  std::thread setter([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    std::lock_guard<SpinLock> lk(mu);
    ready = true;
    cv.notify_one();
  });
  bool got;
  {
    std::lock_guard<SpinLock> lk(mu);
    got = cv.wait_for(mu, 2s, [&] { return ready; });
  }
  setter.join();
  EXPECT_TRUE(got);

  // Case B: predicate never set -> times out -> returns false.
  bool never = false;
  bool got2;
  {
    std::lock_guard<SpinLock> lk(mu);
    got2 = cv.wait_for(mu, 20ms, [&] { return never; });
  }
  EXPECT_FALSE(got2);
}

// Stress the notify-vs-timeout arbitration: many waiters each with a short
// deadline, notified at the wire. Every waiter must return exactly once (either
// outcome) with no lost wake, no double-free, no hang -- TSan gate.
TEST(SyncTimed, NotifyVsTimeoutRace) {
  constexpr int kN = 64;
  ConditionVariable cv;
  SpinLock mu;
  std::atomic<int> returned{0};
  std::vector<std::thread> waiters;
  waiters.reserve(kN);
  for (int i = 0; i < kN; ++i) {
    waiters.emplace_back([&] {
      std::lock_guard<SpinLock> lk(mu);
      cv.wait_for(mu, 5ms); // half will race a notify, half time out
      returned.fetch_add(1, std::memory_order_acq_rel);
    });
  }
  // Fire notifies concurrently with the deadlines to force the race both ways.
  for (int i = 0; i < kN; ++i) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    cv.notify_one();
  }
  for (auto &t : waiters)
    t.join();
  EXPECT_EQ(returned.load(), kN); // all returned, none lost/hung
}

// wait_until with an already-passed deadline returns kTimeout immediately.
TEST(SyncTimed, WaitUntilPastDeadline) {
  ConditionVariable cv;
  SpinLock mu;
  std::lock_guard<SpinLock> lk(mu);
  CvStatus st = cv.wait_until(mu, hyle::time::MonotonicTimePoint{});
  EXPECT_EQ(st, CvStatus::kTimeout);
}

namespace {
struct FiberTimedArg {
  SpinLock mu;
  ConditionVariable cv;
  std::atomic<bool> entered{false};
  std::atomic<bool> returned{false};
  CvStatus status = CvStatus::kNoTimeout;
  hyle::time::Duration timeout{};
};
void fiber_timed_waiter(void *p) {
  auto *a = static_cast<FiberTimedArg *>(p);
  a->mu.lock();
  a->entered.store(true, std::memory_order_release);
  a->status = a->cv.wait_for(a->mu, a->timeout);
  a->mu.unlock();
  a->returned.store(true, std::memory_order_release);
}
} // namespace

TEST(SyncTimed, FiberWaitTimesOutWithoutBlockingWorker) {
  FiberTimedArg arg;
  arg.timeout = 15ms;
  Fiber f(kDefaultStackSize, &fiber_timed_waiter, &arg);
  Scheduler sched(1, system_allocator(), 64);
  EXPECT_TRUE(sched.spawn(&f));
  EXPECT_TRUE(sched.drain_for(2s));
  EXPECT_TRUE(arg.returned.load(std::memory_order_acquire));
  EXPECT_EQ(arg.status, CvStatus::kTimeout);
  sched.shutdown();
}

TEST(SyncTimed, FiberImmediateDeadlineIsSafe) {
  Scheduler sched(2, system_allocator(), 256);
  for (int round = 0; round < 200; ++round) {
    FiberTimedArg arg;
    arg.timeout = hyle::time::Duration::zero();
    Fiber f(kDefaultStackSize, &fiber_timed_waiter, &arg);
    EXPECT_TRUE(sched.spawn(&f));
    EXPECT_TRUE(sched.drain_for(1s));
    EXPECT_TRUE(arg.returned.load(std::memory_order_acquire));
    EXPECT_EQ(arg.status, CvStatus::kTimeout);
  }
  sched.shutdown();
}

TEST(SyncTimed, FiberNotifyWinsAndCancelsTimer) {
  FiberTimedArg arg;
  arg.timeout = 2s;
  Fiber f(kDefaultStackSize, &fiber_timed_waiter, &arg);
  Scheduler sched(2, system_allocator(), 64);
  EXPECT_TRUE(sched.spawn(&f));
  while (!arg.entered.load(std::memory_order_acquire))
    std::this_thread::yield();
  arg.mu.lock();
  arg.mu.unlock();
  arg.cv.notify_one();
  EXPECT_TRUE(sched.drain_for(2s));
  EXPECT_TRUE(arg.returned.load(std::memory_order_acquire));
  EXPECT_EQ(arg.status, CvStatus::kNoTimeout);
  sched.shutdown();
}

TEST(SyncTimed, FiberNotifyVsTimeoutRaceDoesNotDoubleWake) {
  Scheduler sched(4, system_allocator(), 256);
  for (int round = 0; round < 200; ++round) {
    FiberTimedArg arg;
    arg.timeout = 1ms;
    Fiber f(kDefaultStackSize, &fiber_timed_waiter, &arg);
    EXPECT_TRUE(sched.spawn(&f));
    while (!arg.entered.load(std::memory_order_acquire))
      std::this_thread::yield();
    if ((round & 1) == 0) {
      arg.mu.lock();
      arg.mu.unlock();
      arg.cv.notify_one();
    }
    EXPECT_TRUE(sched.drain_for(1s));
    EXPECT_TRUE(arg.returned.load(std::memory_order_acquire));
  }
  sched.shutdown();
}

// ---- Semaphore --------------------------------------------------------------

// try_acquire respects the count and never blocks.
TEST(Sync, SemaphoreTryAcquireCounts) {
  Semaphore s(3);
  EXPECT_TRUE(s.try_acquire(2));  // 3 -> 1
  EXPECT_FALSE(s.try_acquire(2)); // only 1 left
  EXPECT_TRUE(s.try_acquire(1));  // 1 -> 0
  EXPECT_FALSE(s.try_acquire(1)); // empty
  s.release(3);
  EXPECT_EQ(s.available(), 3u);
}

TEST(Sync, SemaphoreTimedAcquireSupportsImmediateSuccessAndTimeout) {
  Semaphore semaphore(1);
  EXPECT_TRUE(semaphore.try_acquire_for(hyle::time::Duration::zero()));
  EXPECT_FALSE(semaphore.try_acquire_for(2ms));
  semaphore.release();
  EXPECT_TRUE(semaphore.try_acquire_until(hyle::time::monotonic_now() + 1s));
}

TEST(Sync, SemaphoreTimedOutHeadMakesAccumulatedPermitsAvailable) {
  Semaphore semaphore(0);
  std::atomic<bool> head_entered{false};
  std::atomic<bool> follower_entered{false};
  std::atomic<bool> head_acquired{true};
  std::atomic<bool> follower_acquired{false};

  std::thread head([&] {
    head_entered.store(true, std::memory_order_release);
    head_acquired.store(semaphore.try_acquire_for(40ms, 2),
                        std::memory_order_release);
  });
  while (!head_entered.load(std::memory_order_acquire))
    std::this_thread::yield();
  std::this_thread::sleep_for(5ms); // establish the two-permit head waiter

  std::thread follower([&] {
    follower_entered.store(true, std::memory_order_release);
    follower_acquired.store(semaphore.acquire(1), std::memory_order_release);
  });
  while (!follower_entered.load(std::memory_order_acquire))
    std::this_thread::yield();
  std::this_thread::sleep_for(5ms); // follower queues behind the head
  semaphore.release(1); // insufficient for the head, enough after timeout

  head.join();
  follower.join();
  EXPECT_FALSE(head_acquired.load(std::memory_order_acquire));
  EXPECT_TRUE(follower_acquired.load(std::memory_order_acquire));
  EXPECT_EQ(semaphore.available(), 0u);
}

namespace {
struct TimedSemaphoreArg {
  Semaphore *semaphore;
  std::atomic<bool> *returned;
  bool acquired = true;
};

void timed_semaphore_waiter(void *raw) {
  auto *arg = static_cast<TimedSemaphoreArg *>(raw);
  arg->acquired = arg->semaphore->try_acquire_for(5ms);
  arg->returned->store(true, std::memory_order_release);
}
} // namespace

TEST(Sync, SemaphoreTimedAcquireSuspendsFiberWithoutBlockingWorker) {
  Semaphore semaphore(0);
  std::atomic<bool> returned{false};
  TimedSemaphoreArg arg{&semaphore, &returned};
  Fiber waiter(kDefaultStackSize, &timed_semaphore_waiter, &arg);
  Scheduler scheduler(1, system_allocator(), 64);
  ASSERT_TRUE(scheduler.spawn(&waiter));
  EXPECT_TRUE(scheduler.drain_for(1s));
  EXPECT_TRUE(returned.load(std::memory_order_acquire));
  EXPECT_FALSE(arg.acquired);
}

// A semaphore as a concurrency limiter: N fibers each
// acquire(1)/work/release(1) against a permit pool of `kLimit`; a shared gauge
// tracks concurrent holders and must never exceed kLimit. A broken semaphore
// both overshoots AND trips TSan.
struct SemLimitArg {
  Semaphore *sem;
  std::atomic<int> in_flight{0};
  std::atomic<int> peak{0};
  int iters;
};
void sem_limit_body(void *p) {
  auto *a = static_cast<SemLimitArg *>(p);
  for (int i = 0; i < a->iters; ++i) {
    a->sem->acquire(1);
    int now = a->in_flight.fetch_add(1, std::memory_order_acq_rel) + 1;
    int prev = a->peak.load(std::memory_order_relaxed);
    while (now > prev && !a->peak.compare_exchange_weak(
                             prev, now, std::memory_order_relaxed)) {
    }
    a->in_flight.fetch_sub(1, std::memory_order_acq_rel);
    a->sem->release(1);
  }
}
TEST(Sync, SemaphoreLimitsConcurrency) {
  constexpr int kLimit = 3;
  constexpr int kFibers = 16;
  Semaphore sem(kLimit);
  SemLimitArg a;
  a.sem = &sem;
  a.iters = 500;
  run_fibers(4, kFibers, &sem_limit_body, &a);
  EXPECT_LE(a.peak.load(), kLimit); // never more than kLimit concurrent holders
  EXPECT_EQ(sem.available(), static_cast<std::size_t>(kLimit)); // all returned
}

// close() wakes every blocked acquirer; their acquire() returns false.
struct SemCloseArg {
  Semaphore *sem;
  std::atomic<int> false_returns{0};
};
void sem_close_body(void *p) {
  auto *a = static_cast<SemCloseArg *>(p);
  if (!a->sem->acquire(1))
    a->false_returns.fetch_add(1, std::memory_order_acq_rel);
}
TEST(Sync, SemaphoreCloseWakesWaiters) {
  constexpr int kFibers = 8;
  Semaphore sem(0); // no permits: everyone blocks
  SemCloseArg a;
  a.sem = &sem;
  std::vector<std::unique_ptr<Fiber>> fibers;
  fibers.reserve(kFibers);
  for (int i = 0; i < kFibers; ++i)
    fibers.push_back(
        std::make_unique<Fiber>(kDefaultStackSize, &sem_close_body, &a));
  Scheduler sched(4, system_allocator(), 1024);
  for (auto &f : fibers)
    sched.spawn(f.get());
  // Close from a bare thread after giving the fibers time to block. drain()
  // then returns only once all acquirers have been woken and returned false.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  sem.close();
  sched.drain();
  EXPECT_EQ(a.false_returns.load(), kFibers);
}

// FIFO no-skip: a big head-of-line request must not be starved by later small
// ones. permits=0; waiter A (wants 4) queues first, then B (wants 1).
// release(1)x3 must NOT satisfy B (it's behind A and A isn't satisfiable yet);
// release the 4th then satisfies A first, then a 5th satisfies B. We verify A
// completes before B.
struct SemFifoArg {
  Semaphore *sem;
  std::size_t need;
  std::atomic<int> *order_ctr;
  int my_order = -1;
};
void sem_fifo_body(void *p) {
  auto *a = static_cast<SemFifoArg *>(p);
  a->sem->acquire(a->need);
  a->my_order = a->order_ctr->fetch_add(1, std::memory_order_acq_rel);
}
TEST(Sync, SemaphoreFifoNoSkip) {
  Semaphore sem(0);
  std::atomic<int> order_ctr{0};
  SemFifoArg big{&sem, 4, &order_ctr};
  SemFifoArg small{&sem, 1, &order_ctr};
  // Spawn A(need 4) first so it takes the FIFO head, then B(need 1). Single
  // worker
  // + a spawn gap makes the enqueue order deterministic.
  auto fa = std::make_unique<Fiber>(kDefaultStackSize, &sem_fifo_body, &big);
  auto fb = std::make_unique<Fiber>(kDefaultStackSize, &sem_fifo_body, &small);
  Scheduler sched(1, system_allocator(), 256);
  sched.spawn(fa.get());
  std::this_thread::sleep_for(
      std::chrono::milliseconds(10)); // A enqueues first
  sched.spawn(fb.get());
  std::this_thread::sleep_for(
      std::chrono::milliseconds(10)); // B enqueues behind
  // Feed permits one at a time from a bare thread. The first 3 must not release
  // B.
  for (int i = 0; i < 5; ++i) {
    sem.release(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  sched.drain();
  EXPECT_EQ(big.my_order, 0); // A (head, wants 4) completes first
  EXPECT_EQ(small.my_order,
            1); // B (wants 1) only after A, despite being smaller
}

// ---- RwLock -----------------------------------------------------------------

// Multiple readers hold the lock concurrently; a writer is exclusive. A shared
// non-atomic counter guarded by writes + read-verified stays consistent, and
// the concurrent-reader gauge proves reads truly overlap (not serialized).
struct RwArg {
  RwLock rw;
  long value = 0; // written under write lock, read under read lock
  std::atomic<int> readers_in{0};
  std::atomic<int> reader_peak{0};
  int iters;
};
void rw_writer_body(void *p) {
  auto *a = static_cast<RwArg *>(p);
  for (int i = 0; i < a->iters; ++i) {
    auto g = a->rw.write();
    ++a->value; // exclusive: no other reader/writer touches value now
  }
}
void rw_reader_body(void *p) {
  auto *a = static_cast<RwArg *>(p);
  for (int i = 0; i < a->iters; ++i) {
    auto g = a->rw.read();
    int now = a->readers_in.fetch_add(1, std::memory_order_acq_rel) + 1;
    int prev = a->reader_peak.load(std::memory_order_relaxed);
    while (now > prev && !a->reader_peak.compare_exchange_weak(
                             prev, now, std::memory_order_relaxed)) {
    }
    // Brief spin INSIDE the read lock so concurrent readers actually overlap in
    // time (a bare read is too fast to catch two holders at once) -- this is
    // what makes reader_peak>=2 observable, proving reads truly share (not
    // serialize).
    volatile long seen = 0;
    for (int k = 0; k < 200; ++k)
      seen += a->value;
    (void)seen;
    a->readers_in.fetch_sub(1, std::memory_order_acq_rel);
  }
}
// Writers are mutually exclusive AND exclusive vs readers: a shared non-atomic
// counter incremented under write locks totals exactly kWriters*iters (a lost
// or overlapping write both corrupts it and trips TSan). Readers verify-read it
// under read locks concurrently with each other.
TEST(Sync, RwLockReadersShareWriterExclusive) {
  constexpr int kWriters = 2;
  constexpr int kReaders = 8;
  RwArg a;
  a.iters = 300;
  std::vector<std::unique_ptr<Fiber>> fibers;
  fibers.reserve(kWriters + kReaders);
  for (int i = 0; i < kWriters; ++i)
    fibers.push_back(
        std::make_unique<Fiber>(kDefaultStackSize, &rw_writer_body, &a));
  for (int i = 0; i < kReaders; ++i)
    fibers.push_back(
        std::make_unique<Fiber>(kDefaultStackSize, &rw_reader_body, &a));
  Scheduler sched(4, system_allocator(), 1024);
  for (auto &f : fibers)
    sched.spawn(f.get());
  sched.drain();
  EXPECT_EQ(a.value, static_cast<long>(kWriters) * a.iters); // no lost writes
}

// Readers genuinely share (not serialized): with NO writers competing, many
// reader fibers holding the lock at once must push the concurrent-reader gauge
// above 1. (Separated from the writer test because dense writers fragment the
// readers into non-overlapping batches, hiding the sharing.)
// Readers genuinely share, verified DETERMINISTICALLY (not via a flaky timing
// gauge): every reader takes the read lock, then spins until ALL kReaders are
// simultaneously inside the lock. If RwLock serialized readers, they could
// never all be in at once -> the barrier never completes -> drain() hangs ->
// ctest timeout. So completing at all IS the proof of concurrent read holders.
// Needs
// >= kReaders workers so all reader fibers can run at once.
struct RwOverlapArg {
  RwLock rw;
  std::atomic<int> inside{0};
  int target;
};
void rw_overlap_body(void *p) {
  auto *a = static_cast<RwOverlapArg *>(p);
  auto g = a->rw.read(); // shared read lock
  a->inside.fetch_add(1, std::memory_order_acq_rel);
  // Spin until everyone is inside -- only possible if reads truly share.
  while (a->inside.load(std::memory_order_acquire) < a->target)
    hyle::sched::Fiber::yield();
}
TEST(Sync, RwLockReadersOverlap) {
  constexpr int kReaders = 8;
  auto a = std::make_unique<RwOverlapArg>();
  a->target = kReaders;
  std::vector<std::unique_ptr<Fiber>> fibers;
  fibers.reserve(kReaders);
  for (int i = 0; i < kReaders; ++i)
    fibers.push_back(
        std::make_unique<Fiber>(kDefaultStackSize, &rw_overlap_body, a.get()));
  Scheduler sched(kReaders, system_allocator(), 1024); // enough workers for all
  for (auto &f : fibers)
    sched.spawn(f.get());
  sched.drain(); // completes ONLY if all kReaders held the read lock at once
  EXPECT_EQ(a->inside.load(), kReaders);
}

// try_read / try_write reflect the current holder.
TEST(Sync, RwLockTryPaths) {
  RwLock rw;
  {
    auto w = rw.write();
    EXPECT_TRUE(w.owns_lock());
    EXPECT_FALSE(rw.try_read().owns_lock());  // write held -> no read
    EXPECT_FALSE(rw.try_write().owns_lock()); // write held -> no second write
  }
  {
    auto r1 = rw.read();
    EXPECT_TRUE(r1.owns_lock());
    EXPECT_TRUE(rw.try_read().owns_lock());   // reads share
    EXPECT_FALSE(rw.try_write().owns_lock()); // read held -> no write
  }
  EXPECT_TRUE(rw.try_write().owns_lock()); // all released -> write available
}

// ---- oneshot ----------------------------------------------------------------

// A sender fiber sends a value; a receiver fiber recv()s it. The value crosses
// the fiber boundary exactly once.
struct OneshotArg {
  hyle::sched::OneshotSender<int> tx;
  hyle::sched::OneshotReceiver<int> rx;
  std::atomic<int> got{-1};
};
void oneshot_tx_body(void *p) {
  auto *a = static_cast<OneshotArg *>(p);
  (void)std::move(a->tx).send(42);
}
void oneshot_rx_body(void *p) {
  auto *a = static_cast<OneshotArg *>(p);
  auto r = std::move(a->rx).recv();
  a->got.store(r.has_value() ? *r : -1, std::memory_order_release);
}
TEST(Sync, OneshotDeliversValue) {
  auto [tx, rx] = make_oneshot<int>(system_allocator());
  auto a = std::make_unique<OneshotArg>();
  a->tx = std::move(tx);
  a->rx = std::move(rx);
  auto ftx =
      std::make_unique<Fiber>(kDefaultStackSize, &oneshot_tx_body, a.get());
  auto frx =
      std::make_unique<Fiber>(kDefaultStackSize, &oneshot_rx_body, a.get());
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(frx.get()); // receiver blocks first
  sched.spawn(ftx.get());
  sched.drain();
  EXPECT_EQ(a->got.load(), 42);
}

// Sender dropped without sending -> recv() returns Cancelled (not a hang).
TEST(Sync, OneshotSenderDropCancels) {
  auto [tx, rx] = make_oneshot<int>(system_allocator());
  {
    auto dead = std::move(tx);
  } // drop the sender without sending
  auto r = std::move(rx).recv();
  EXPECT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsCancelled());
}

// Receiver dropped -> send() returns the value back as an error.
TEST(Sync, OneshotReceiverDropRejectsSend) {
  auto [tx, rx] = make_oneshot<int>(system_allocator());
  {
    auto dead = std::move(rx);
  } // drop the receiver
  auto r = std::move(tx).send(7);
  EXPECT_FALSE(r.has_value()); // nobody home
  EXPECT_EQ(r.error(), 7);     // the producer retains ownership of its value
}

TEST(Sync, OneshotTryRecvAfterValueReportsClosedInsteadOfHanging) {
  auto [tx, rx] = hyle::sched::make_oneshot<int>(system_allocator());
  ASSERT_TRUE(tx.send(42).has_value());
  auto first = rx.try_recv();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(*first, 42);
  auto second = rx.try_recv();
  ASSERT_FALSE(second.has_value());
  EXPECT_TRUE(second.error().IsCancelled());
}

// ---- watch ------------------------------------------------------------------

// borrow() returns the seeded value immediately; after a send, a receiver's
// changed() wakes and borrow() reflects the new value. Models a shutdown flag.
struct WatchArg {
  hyle::sched::WatchReceiver<int> rx;
  std::atomic<int> observed{-1};
  std::atomic<bool> changed_ok{false};
};
void watch_rx_body(void *p) {
  auto *a = static_cast<WatchArg *>(p);
  auto r = a->rx.changed(); // block until the first send
  a->changed_ok.store(r.has_value(), std::memory_order_release);
  a->observed.store(a->rx.borrow(), std::memory_order_release);
}
TEST(Sync, WatchSeesLatestValue) {
  auto [tx, rx] = make_watch<int>(system_allocator(), 0);
  EXPECT_EQ(rx.borrow(), 0); // late subscriber sees current immediately
  auto a = std::make_unique<WatchArg>();
  a->rx = rx; // copy (watch receivers are copyable)
  auto frx =
      std::make_unique<Fiber>(kDefaultStackSize, &watch_rx_body, a.get());
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(frx.get());
  std::this_thread::sleep_for(std::chrono::milliseconds(10)); // receiver blocks
  EXPECT_TRUE(tx.send(99).has_value()); // has receivers -> delivered
  sched.drain();
  EXPECT_TRUE(a->changed_ok.load());
  EXPECT_EQ(a->observed.load(), 99);
}

// All senders dropped -> changed() returns Cancelled.
TEST(Sync, WatchSenderDropCloses) {
  auto [tx, rx] = make_watch<int>(system_allocator(), 5);
  {
    auto dead = std::move(tx);
  } // drop the only sender
  auto r = rx.changed();
  EXPECT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsCancelled());
}

TEST(Sync, WatchStaysOpenUntilLastClonedSenderDrops) {
  auto [tx, rx] = make_watch<int>(system_allocator(), 5);
  auto tx2 = tx;
  tx = {};
  ASSERT_TRUE(tx2.send(9).has_value());
  ASSERT_TRUE(rx.changed().has_value());
  EXPECT_EQ(rx.borrow(), 9);
  tx2 = {};
  auto closed = rx.changed();
  ASSERT_FALSE(closed.has_value());
  EXPECT_TRUE(closed.error().IsCancelled());
}

// ---- broadcast --------------------------------------------------------------

// Two receivers each get EVERY value a sender broadcasts (fan-out).
struct BcastArg {
  hyle::sched::BroadcastReceiver<int> rx;
  int n;
  std::atomic<long> sum{0};
};
void bcast_rx_body(void *p) {
  auto *a = static_cast<BcastArg *>(p);
  int got = 0;
  while (got < a->n) {
    auto r = a->rx.recv();
    if (r.has_value()) {
      a->sum.fetch_add(*r, std::memory_order_acq_rel);
      ++got;
    } else if (r.error().status().IsCancelled()) {
      break; // closed
    }
    // Unavailable (lagged): just retry.
  }
}
TEST(Sync, BroadcastFanOutToAll) {
  constexpr int kN = 50;
  auto [tx, rx0] =
      make_broadcast<int>(system_allocator(), 64); // cap > N: no lag
  auto rx1 = tx.subscribe();
  auto a0 = std::make_unique<BcastArg>();
  auto a1 = std::make_unique<BcastArg>();
  a0->rx = rx0;
  a0->n = kN;
  a1->rx = rx1;
  a1->n = kN;
  auto f0 =
      std::make_unique<Fiber>(kDefaultStackSize, &bcast_rx_body, a0.get());
  auto f1 =
      std::make_unique<Fiber>(kDefaultStackSize, &bcast_rx_body, a1.get());
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(f0.get());
  sched.spawn(f1.get());
  std::this_thread::sleep_for(std::chrono::milliseconds(10)); // receivers block
  long expect = 0;
  for (int i = 1; i <= kN; ++i) {
    tx.send(i);
    expect += i;
  }
  sched.drain();
  EXPECT_EQ(a0->sum.load(), expect); // receiver 0 saw every value
  EXPECT_EQ(a1->sum.load(), expect); // receiver 1 saw every value too
}

// A slow receiver that falls behind the ring window gets a Lagged (Unavailable)
// signal rather than silently missing values or blocking the sender.
TEST(Sync, BroadcastLaggedOnOverflow) {
  auto [tx, rx] = make_broadcast<int>(system_allocator(), 4); // tiny ring
  for (int i = 0; i < 10; ++i)
    tx.send(i); // 10 sends into a 4-slot ring
  // The receiver never read; its cursor (0) is now far behind. First recv must
  // report the lag and fast-forward, not return stale/garbage.
  auto r = rx.recv();
  EXPECT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().lagged());
  EXPECT_EQ(r.error().missed(), 6u);
  EXPECT_TRUE(r.error().status().IsUnavailable());
  // After the lag signal, subsequent recv() yields retained values.
  auto r2 = rx.recv();
  EXPECT_TRUE(r2.has_value());
  EXPECT_EQ(*r2, 6); // oldest retained = write_pos(10) - cap(4)
}

TEST(Sync, BroadcastTryRecvReportsEmptyLagAndClose) {
  auto [tx, rx] = make_broadcast<int>(system_allocator(), 2);
  auto empty = rx.try_recv();
  ASSERT_FALSE(empty.has_value());
  EXPECT_TRUE(empty.error().status().IsNotFound());
  tx.send(1);
  tx.send(2);
  tx.send(3);
  auto lagged = rx.try_recv();
  ASSERT_FALSE(lagged.has_value());
  EXPECT_TRUE(lagged.error().lagged());
  EXPECT_EQ(lagged.error().missed(), 1u);
  ASSERT_EQ(*rx.try_recv(), 2);
  ASSERT_EQ(*rx.try_recv(), 3);
  tx = {};
  auto closed = rx.try_recv();
  ASSERT_FALSE(closed.has_value());
  EXPECT_TRUE(closed.error().status().IsCancelled());
}

TEST(Sync, MessagingCopyFailureDoesNotStrandMonitor) {
  auto [watch_tx, watch_rx] =
      make_watch<ThrowOnCopy>(system_allocator(), ThrowOnCopy(3));
  ThrowOnCopy::throw_copy = true;
  EXPECT_THROW((void)watch_rx.borrow(), std::runtime_error);
  ThrowOnCopy::throw_copy = false;
  EXPECT_EQ(watch_rx.borrow().value, 3);

  auto [broadcast_tx, broadcast_rx] =
      make_broadcast<ThrowOnCopy>(system_allocator(), 2);
  broadcast_tx.send(ThrowOnCopy(7));
  ThrowOnCopy::throw_copy = true;
  EXPECT_THROW((void)broadcast_rx.try_recv(), std::runtime_error);
  ThrowOnCopy::throw_copy = false;
  auto value = broadcast_rx.try_recv();
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(value->value, 7); // failed copy did not advance the receive cursor
}

// ---- completeness: newly-added responsibilities -----------------------------

// Semaphore RAII Permit releases automatically on scope exit; forget() removes
// it.
TEST(Sync, SemaphorePermitGuard) {
  Semaphore s(2);
  {
    auto p = s.acquire_guard(2);
    EXPECT_TRUE(p.owns());
    EXPECT_EQ(s.available(), 0u);
  }
  EXPECT_EQ(s.available(), 2u); // returned on scope exit
  {
    auto p = s.acquire_guard(1);
    p.forget(); // permanently remove this permit from the pool
  }
  EXPECT_EQ(s.available(), 1u); // forgot 1 -> capacity shrank
  auto none = s.try_acquire_guard(5);
  EXPECT_FALSE(none.owns()); // can't get 5 from 1
}

// oneshot Sender::closed() blocks until the receiver is dropped (async cancel).
struct OneshotClosedArg {
  hyle::sched::OneshotSender<int> tx;
  std::atomic<bool> saw_closed{false};
};
void oneshot_closed_body(void *p) {
  auto *a = static_cast<OneshotClosedArg *>(p);
  a->tx.closed(); // blocks until receiver gone
  a->saw_closed.store(true, std::memory_order_release);
}
TEST(Sync, OneshotClosedAwaitsReceiverDrop) {
  auto [tx, rx] = make_oneshot<int>(system_allocator());
  auto a = std::make_unique<OneshotClosedArg>();
  a->tx = std::move(tx);
  auto f =
      std::make_unique<Fiber>(kDefaultStackSize, &oneshot_closed_body, a.get());
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(f.get());
  std::this_thread::sleep_for(
      std::chrono::milliseconds(10)); // sender blocks in closed()
  EXPECT_FALSE(a->saw_closed.load()); // receiver still alive -> still blocked
  {
    auto dead = std::move(rx);
  } // drop the receiver
  sched.drain();
  EXPECT_TRUE(a->saw_closed.load()); // closed() returned
}

// RwLock write guard downgrades to a read guard without releasing the lock (no
// writer can slip in): the value written under the write lock is still
// readable.
TEST(Sync, RwLockDowngrade) {
  RwLock rw;
  auto rg = [&] {
    auto w = rw.write();
    EXPECT_TRUE(w.owns_lock());
    return std::move(w).downgrade(); // -> ReadGuard, still holding the lock
  }();
  EXPECT_TRUE(rg.owns_lock());
  EXPECT_TRUE(rw.try_read().owns_lock());   // other readers may now share
  EXPECT_FALSE(rw.try_write().owns_lock()); // but no writer (rg still holds 1)
}

// watch send_modify mutates in place + notifies; borrow_and_update marks seen;
// receiver_count reflects live receivers.
TEST(Sync, WatchModifyAndCounts) {
  auto [tx, rx] = make_watch<int>(system_allocator(), 10);
  EXPECT_EQ(tx.receiver_count(), 1u);
  {
    auto rx2 = rx; // copy -> another receiver
    EXPECT_EQ(tx.receiver_count(), 2u);
  }
  EXPECT_EQ(tx.receiver_count(), 1u); // rx2 dropped

  tx.send_modify([](int &v) noexcept { v += 5; }); // 10 -> 15, notify
  EXPECT_TRUE(rx.has_changed());
  EXPECT_EQ(rx.borrow_and_update(), 15); // read + mark seen
  EXPECT_FALSE(rx.has_changed());        // seen now aligned

  int old = tx.send_replace(99); // returns old, unconditional
  EXPECT_EQ(old, 15);
  EXPECT_EQ(rx.borrow(), 99);
}

// watch send() with no receivers returns Cancelled (deliver-to-consumer
// sematics), while send_replace still updates the shared state.
TEST(Sync, WatchSendNoReceivers) {
  auto [tx, rx] = make_watch<int>(system_allocator(), 0);
  {
    auto dead = std::move(rx);
  } // drop the only receiver
  EXPECT_EQ(tx.receiver_count(), 0u);
  auto r = tx.send(1);
  EXPECT_FALSE(r.has_value());      // no consumer -> error
  EXPECT_EQ(tx.send_replace(2), 0); // but state register still updates (old=0)
  EXPECT_EQ(tx.borrow(), 2);
}

// Stress the sender-drop-vs-recv race (the lost-wakeup window a red-team review
// found): a receiver fiber blocks in recv() while a bare thread drops the
// sender at an unpredictable moment. If the close signal weren't published
// under the CV monitor lock, the notify could race ahead of the flag and the
// receiver would hang -> drain() never returns -> ctest timeout. 200 rounds to
// hit the window.
TEST(Sync, OneshotSenderDropRaceNoHang) {
  for (int round = 0; round < 200; ++round) {
    auto [tx, rx] = make_oneshot<int>(system_allocator());
    std::atomic<bool> got_result{false};
    struct RaceArg {
      hyle::sched::OneshotReceiver<int> rx;
      std::atomic<bool> *done;
    };
    auto arg = std::make_unique<RaceArg>();
    arg->rx = std::move(rx);
    arg->done = &got_result;
    auto body = [](void *p) {
      auto *a = static_cast<RaceArg *>(p);
      auto r = std::move(a->rx).recv(); // must return (Cancelled), never hang
      a->done->store(true, std::memory_order_release);
      (void)r;
    };
    auto f = std::make_unique<Fiber>(kDefaultStackSize, body, arg.get());
    Scheduler sched(2, system_allocator(), 256);
    sched.spawn(f.get());
    // Alternate: half the rounds drop immediately (racing recv's entry), half
    // after a short delay (guaranteeing the receiver is already PARKED -- the
    // window where a lost close-notify hangs forever). Covers both
    // interleavings.
    if (round % 2 == 0)
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    {
      auto dead = std::move(tx);
    }              // drop sender vs a (maybe already parked) recv
    sched.drain(); // hangs here if the wakeup was lost
    EXPECT_TRUE(got_result.load());
  }
}

// Same race for watch: receiver blocked in changed() while the last sender
// drops.
TEST(Sync, WatchSenderDropRaceNoHang) {
  for (int round = 0; round < 200; ++round) {
    auto [tx, rx] = make_watch<int>(system_allocator(), 0);
    std::atomic<bool> returned{false};
    struct RaceArg {
      hyle::sched::WatchReceiver<int> rx;
      std::atomic<bool> *done;
    };
    auto arg = std::make_unique<RaceArg>();
    arg->rx = rx; // copy; original rx dropped at end of loop (keeps a receiver)
    arg->done = &returned;
    auto body = [](void *p) {
      auto *a = static_cast<RaceArg *>(p);
      (void)a->rx.changed(); // must return (Cancelled on close), never hang
      a->done->store(true, std::memory_order_release);
    };
    auto f = std::make_unique<Fiber>(kDefaultStackSize, body, arg.get());
    Scheduler sched(2, system_allocator(), 256);
    sched.spawn(f.get());
    {
      auto dead = std::move(tx);
    }              // last sender drops vs changed()
    sched.drain(); // hangs here if the wakeup was lost
    EXPECT_TRUE(returned.load());
  }
}

// broadcast receiver_count / sender_count track live handles.
TEST(Sync, BroadcastCounts) {
  auto [tx, rx] = make_broadcast<int>(system_allocator(), 8);
  EXPECT_EQ(tx.receiver_count(), 1u);
  EXPECT_EQ(tx.sender_count(), 1u);
  {
    auto rx2 = tx.subscribe();
    auto tx2 = tx;
    EXPECT_EQ(tx.receiver_count(), 2u);
    EXPECT_EQ(tx.sender_count(), 2u);
  }
  EXPECT_EQ(tx.receiver_count(), 1u);
  EXPECT_EQ(tx.sender_count(), 1u);
}

} // namespace
