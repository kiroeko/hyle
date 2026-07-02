#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"
#include "hyle/util/spin_lock.h"

using hyle::mem::system_allocator;
using hyle::sched::Channel;
using hyle::sched::ConditionVariable;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Latch;
using hyle::sched::Mutex;
using hyle::sched::Scheduler;
using hyle::sched::WaitGroup;
using hyle::util::SpinLock;

namespace {

// Spawn `bodies.size()` fibers on a `workers`-wide scheduler and block until all
// of them finish. Fibers outlive the scheduler (drain() returns only once every
// fiber is done), so a stack vector of Fibers is safe.
template <class Fn>
void run_fibers(unsigned workers, int count, Fn entry, void* arg) {
  std::vector<std::unique_ptr<Fiber>> fibers;
  fibers.reserve(count);
  for (int i = 0; i < count; ++i) {
    fibers.push_back(std::make_unique<Fiber>(system_allocator(),
                                             kDefaultStackSize, entry, arg));
  }
  Scheduler sched(workers, system_allocator(), 1024);
  for (auto& f : fibers) sched.spawn(f.get());
  sched.drain();
}

// ---- Mutex ------------------------------------------------------------------

struct MutexArg {
  Mutex m;
  long counter = 0;  // plain (non-atomic): a broken mutex => TSan race + wrong sum
  int iters;
};

void mutex_body(void* p) {
  auto* a = static_cast<MutexArg*>(p);
  for (int i = 0; i < a->iters; ++i) {
    a->m.lock();
    ++a->counter;  // critical section
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
  int n;                              // number of worker fibers
  std::atomic<int> seen_at_wait{-1};  // done_count observed when wait() returned
};

void wg_worker(void* p) {
  auto* a = static_cast<WgArg*>(p);
  a->done_count.fetch_add(1, std::memory_order_acq_rel);  // "work happened"
  a->wg.done();
}
void wg_coordinator(void* p) {
  auto* a = static_cast<WgArg*>(p);
  a->wg.wait();  // returns only after all n done()
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
  fibers.push_back(std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                           &wg_coordinator, &a));
  for (int i = 0; i < kN; ++i) {
    fibers.push_back(std::make_unique<Fiber>(system_allocator(),
                                             kDefaultStackSize, &wg_worker, &a));
  }
  Scheduler sched(4, system_allocator(), 1024);
  for (auto& f : fibers) sched.spawn(f.get());
  sched.drain();
  EXPECT_EQ(a.seen_at_wait.load(), kN);
}

// ---- Latch ------------------------------------------------------------------

struct LatchArg {
  Latch* latch;
  std::atomic<int> resumed{0};  // waiters that passed the latch
};

void latch_counter(void* p) {
  auto* a = static_cast<LatchArg*>(p);
  a->latch->count_down();
}
void latch_waiter(void* p) {
  auto* a = static_cast<LatchArg*>(p);
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
    fibers.push_back(std::make_unique<Fiber>(system_allocator(),
                                             kDefaultStackSize, &latch_waiter, &a));
  for (int i = 0; i < kCount; ++i)
    fibers.push_back(std::make_unique<Fiber>(system_allocator(),
                                             kDefaultStackSize, &latch_counter, &a));
  Scheduler sched(4, system_allocator(), 1024);
  for (auto& f : fibers) sched.spawn(f.get());
  sched.drain();
  EXPECT_EQ(a.resumed.load(), kWaiters);
}

// ---- Channel ----------------------------------------------------------------

struct ChanArg {
  Channel<int>* ch;
  int n;                       // values to produce
  std::atomic<long> sum{0};    // sum of received values
  std::atomic<int> count{0};   // number received
};

void chan_producer(void* p) {
  auto* a = static_cast<ChanArg*>(p);
  for (int i = 0; i < a->n; ++i) a->ch->send(i);
  a->ch->close();
}
void chan_consumer(void* p) {
  auto* a = static_cast<ChanArg*>(p);
  int v;
  while (a->ch->recv(v)) {
    a->sum.fetch_add(v, std::memory_order_acq_rel);
    a->count.fetch_add(1, std::memory_order_acq_rel);
  }
}

// One producer fills a small bounded channel (forcing the producer to block when
// full), one consumer drains it (blocking when empty); close() ends the stream.
TEST(Sync, ChannelProducerConsumer) {
  constexpr int kN = 1000;
  Channel<int> ch(system_allocator(), 4);  // tiny cap => exercises full/empty waits
  ChanArg a;
  a.ch = &ch;
  a.n = kN;
  std::unique_ptr<Fiber> prod = std::make_unique<Fiber>(
      system_allocator(), kDefaultStackSize, &chan_producer, &a);
  std::unique_ptr<Fiber> cons = std::make_unique<Fiber>(
      system_allocator(), kDefaultStackSize, &chan_consumer, &a);
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(prod.get());
  sched.spawn(cons.get());
  sched.drain();
  EXPECT_EQ(a.count.load(), kN);
  EXPECT_EQ(a.sum.load(), static_cast<long>(kN) * (kN - 1) / 2);
}

namespace {
// Multi-producer/consumer channel, closed once all producers finish (via a
// WaitGroup) -- composes Channel + WaitGroup, the convergence thesis in action.
struct ChanMpmcArg {
  Channel<int>* ch;
  WaitGroup* producers_done;
  int per_producer;
  std::atomic<long> sum{0};
  std::atomic<int> count{0};
};
void mpmc_producer(void* p) {
  auto* a = static_cast<ChanMpmcArg*>(p);
  for (int i = 0; i < a->per_producer; ++i) a->ch->send(1);
  a->producers_done->done();
}
void mpmc_closer(void* p) {
  auto* a = static_cast<ChanMpmcArg*>(p);
  a->producers_done->wait();  // all producers finished
  a->ch->close();
}
void mpmc_consumer(void* p) {
  auto* a = static_cast<ChanMpmcArg*>(p);
  int v;
  while (a->ch->recv(v)) {
    a->sum.fetch_add(v, std::memory_order_acq_rel);
    a->count.fetch_add(1, std::memory_order_acq_rel);
  }
}
}  // namespace

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
    fibers.push_back(std::make_unique<Fiber>(system_allocator(),
                                             kDefaultStackSize, &mpmc_producer, &a));
  for (int i = 0; i < kConsumers; ++i)
    fibers.push_back(std::make_unique<Fiber>(system_allocator(),
                                             kDefaultStackSize, &mpmc_consumer, &a));
  fibers.push_back(std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                           &mpmc_closer, &a));
  Scheduler sched(4, system_allocator(), 1024);
  for (auto& f : fibers) sched.spawn(f.get());
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

void cv_waiter(void* p) {
  auto* a = static_cast<CvBroadcastArg*>(p);
  a->mu.lock();
  while (!a->go) a->cv.wait(a->mu);  // releases mu, yields, re-acquires
  a->mu.unlock();
  a->woke.fetch_add(1, std::memory_order_acq_rel);
}
void cv_broadcaster(void* p) {
  auto* a = static_cast<CvBroadcastArg*>(p);
  a->mu.lock();
  a->go = true;
  a->mu.unlock();
  a->cv.notify_all();
}

// N fibers block on one CV; a single notify_all (after setting the predicate)
// must release every one. The predicate-under-monitor-lock discipline closes the
// lost-wakeup window regardless of broadcaster-vs-waiter ordering.
TEST(Sync, CvNotifyAllWakesAllFibers) {
  constexpr int kWaiters = 200;
  CvBroadcastArg a;
  std::vector<std::unique_ptr<Fiber>> fibers;
  for (int i = 0; i < kWaiters; ++i)
    fibers.push_back(std::make_unique<Fiber>(system_allocator(),
                                             kDefaultStackSize, &cv_waiter, &a));
  fibers.push_back(std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                           &cv_broadcaster, &a));
  Scheduler sched(4, system_allocator(), 1024);
  for (auto& f : fibers) sched.spawn(f.get());
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

void cross_notifier_fiber(void* p) {
  auto* a = static_cast<CrossArg*>(p);
  a->mu.lock();
  a->ready = true;
  a->mu.unlock();
  a->cv.notify_all();
}

// THREAD-mode waiter: a bare (non-worker) thread blocks on the CV and is woken by
// a fiber's notify. This is the cross-process / external-submit path -- the CV
// must fall back to Parker, not assume the waiter is a fiber.
TEST(Sync, CvThreadWaiterWokenByFiber) {
  CrossArg a;
  std::thread waiter([&a] {
    a.mu.lock();
    while (!a.ready) a.cv.wait(a.mu);  // bare thread => thread-mode park
    a.mu.unlock();
    a.resumed.store(true, std::memory_order_release);
  });
  std::unique_ptr<Fiber> notifier = std::make_unique<Fiber>(
      system_allocator(), kDefaultStackSize, &cross_notifier_fiber, &a);
  {
    Scheduler sched(2, system_allocator(), 256);
    sched.spawn(notifier.get());
    sched.drain();
  }
  waiter.join();
  EXPECT_TRUE(a.resumed.load());
}

namespace {
void cv_fiber_waiter(void* p) {
  auto* a = static_cast<CrossArg*>(p);
  a->mu.lock();
  while (!a->ready) a->cv.wait(a->mu);  // worker fiber => fiber-mode suspend
  a->mu.unlock();
  a->resumed.store(true, std::memory_order_release);
}
}  // namespace

// FIBER-mode waiter woken by a bare (main) thread's notify -- the inverse path.
TEST(Sync, CvFiberWaiterWokenByThread) {
  CrossArg a;
  std::unique_ptr<Fiber> waiter = std::make_unique<Fiber>(
      system_allocator(), kDefaultStackSize, &cv_fiber_waiter, &a);
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

// ---- timed wait (sched-04c, thread-mode) ------------------------------------
// These use BARE std::thread waiters (thread-mode: no scheduler/fiber), which is
// the shipped path for sched-04c. A fiber-mode timed wait is not yet supported.

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
    st = cv.wait_for(mu, 30'000'000ull);  // 30ms, nobody notifies
  }
  auto elapsed = std::chrono::steady_clock::now() - t0;
  EXPECT_EQ(st, CvStatus::kTimeout);
  EXPECT_GE(elapsed, std::chrono::milliseconds(20));  // really slept
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
    st.store(cv.wait_for(mu, 5'000'000'000ull),  // 5s -- we notify well before
             std::memory_order_release);
  });
  while (!parked.load(std::memory_order_acquire)) std::this_thread::yield();
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
    got = cv.wait_for(mu, 2'000'000'000ull, [&] { return ready; });
  }
  setter.join();
  EXPECT_TRUE(got);

  // Case B: predicate never set -> times out -> returns false.
  bool never = false;
  bool got2;
  {
    std::lock_guard<SpinLock> lk(mu);
    got2 = cv.wait_for(mu, 20'000'000ull, [&] { return never; });
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
      cv.wait_for(mu, 5'000'000ull);  // 5ms: half will race a notify, half time out
      returned.fetch_add(1, std::memory_order_acq_rel);
    });
  }
  // Fire notifies concurrently with the deadlines to force the race both ways.
  for (int i = 0; i < kN; ++i) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    cv.notify_one();
  }
  for (auto& t : waiters) t.join();
  EXPECT_EQ(returned.load(), kN);  // all returned, none lost/hung
}

// wait_until with an already-passed deadline returns kTimeout immediately.
TEST(SyncTimed, WaitUntilPastDeadline) {
  ConditionVariable cv;
  SpinLock mu;
  std::lock_guard<SpinLock> lk(mu);
  CvStatus st = cv.wait_until(mu, 1);  // deadline ns=1, long in the past
  EXPECT_EQ(st, CvStatus::kTimeout);
}

}  // namespace
