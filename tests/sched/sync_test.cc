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
using hyle::sched::make_broadcast;
using hyle::sched::make_oneshot;
using hyle::sched::make_watch;
using hyle::sched::Mutex;
using hyle::sched::RwLock;
using hyle::sched::Scheduler;
using hyle::sched::Semaphore;
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

// ---- Semaphore --------------------------------------------------------------

// try_acquire respects the count and never blocks.
TEST(Sync, SemaphoreTryAcquireCounts) {
  Semaphore s(3);
  EXPECT_TRUE(s.try_acquire(2));   // 3 -> 1
  EXPECT_FALSE(s.try_acquire(2));  // only 1 left
  EXPECT_TRUE(s.try_acquire(1));   // 1 -> 0
  EXPECT_FALSE(s.try_acquire(1));  // empty
  s.release(3);
  EXPECT_EQ(s.available(), 3u);
}

// A semaphore as a concurrency limiter: N fibers each acquire(1)/work/release(1)
// against a permit pool of `kLimit`; a shared gauge tracks concurrent holders and
// must never exceed kLimit. A broken semaphore both overshoots AND trips TSan.
struct SemLimitArg {
  Semaphore* sem;
  std::atomic<int> in_flight{0};
  std::atomic<int> peak{0};
  int iters;
};
void sem_limit_body(void* p) {
  auto* a = static_cast<SemLimitArg*>(p);
  for (int i = 0; i < a->iters; ++i) {
    a->sem->acquire(1);
    int now = a->in_flight.fetch_add(1, std::memory_order_acq_rel) + 1;
    int prev = a->peak.load(std::memory_order_relaxed);
    while (now > prev &&
           !a->peak.compare_exchange_weak(prev, now, std::memory_order_relaxed)) {
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
  EXPECT_LE(a.peak.load(), kLimit);  // never more than kLimit concurrent holders
  EXPECT_EQ(sem.available(), static_cast<std::size_t>(kLimit));  // all returned
}

// close() wakes every blocked acquirer; their acquire() returns false.
struct SemCloseArg {
  Semaphore* sem;
  std::atomic<int> false_returns{0};
};
void sem_close_body(void* p) {
  auto* a = static_cast<SemCloseArg*>(p);
  if (!a->sem->acquire(1)) a->false_returns.fetch_add(1, std::memory_order_acq_rel);
}
TEST(Sync, SemaphoreCloseWakesWaiters) {
  constexpr int kFibers = 8;
  Semaphore sem(0);  // no permits: everyone blocks
  SemCloseArg a;
  a.sem = &sem;
  std::vector<std::unique_ptr<Fiber>> fibers;
  fibers.reserve(kFibers);
  for (int i = 0; i < kFibers; ++i)
    fibers.push_back(std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                             &sem_close_body, &a));
  Scheduler sched(4, system_allocator(), 1024);
  for (auto& f : fibers) sched.spawn(f.get());
  // Close from a bare thread after giving the fibers time to block. drain() then
  // returns only once all acquirers have been woken and returned false.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  sem.close();
  sched.drain();
  EXPECT_EQ(a.false_returns.load(), kFibers);
}

// FIFO no-skip: a big head-of-line request must not be starved by later small
// ones. permits=0; waiter A (wants 4) queues first, then B (wants 1). release(1)x3
// must NOT satisfy B (it's behind A and A isn't satisfiable yet); release the 4th
// then satisfies A first, then a 5th satisfies B. We verify A completes before B.
struct SemFifoArg {
  Semaphore* sem;
  std::size_t need;
  std::atomic<int>* order_ctr;
  int my_order = -1;
};
void sem_fifo_body(void* p) {
  auto* a = static_cast<SemFifoArg*>(p);
  a->sem->acquire(a->need);
  a->my_order = a->order_ctr->fetch_add(1, std::memory_order_acq_rel);
}
TEST(Sync, SemaphoreFifoNoSkip) {
  Semaphore sem(0);
  std::atomic<int> order_ctr{0};
  SemFifoArg big{&sem, 4, &order_ctr};
  SemFifoArg small{&sem, 1, &order_ctr};
  // Spawn A(need 4) first so it takes the FIFO head, then B(need 1). Single worker
  // + a spawn gap makes the enqueue order deterministic.
  auto fa = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                    &sem_fifo_body, &big);
  auto fb = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                    &sem_fifo_body, &small);
  Scheduler sched(1, system_allocator(), 256);
  sched.spawn(fa.get());
  std::this_thread::sleep_for(std::chrono::milliseconds(10));  // A enqueues first
  sched.spawn(fb.get());
  std::this_thread::sleep_for(std::chrono::milliseconds(10));  // B enqueues behind
  // Feed permits one at a time from a bare thread. The first 3 must not release B.
  for (int i = 0; i < 5; ++i) {
    sem.release(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  sched.drain();
  EXPECT_EQ(big.my_order, 0);    // A (head, wants 4) completes first
  EXPECT_EQ(small.my_order, 1);  // B (wants 1) only after A, despite being smaller
}

// ---- RwLock -----------------------------------------------------------------

// Multiple readers hold the lock concurrently; a writer is exclusive. A shared
// non-atomic counter guarded by writes + read-verified stays consistent, and the
// concurrent-reader gauge proves reads truly overlap (not serialized).
struct RwArg {
  RwLock rw;
  long value = 0;  // written under write lock, read under read lock
  std::atomic<int> readers_in{0};
  std::atomic<int> reader_peak{0};
  int iters;
};
void rw_writer_body(void* p) {
  auto* a = static_cast<RwArg*>(p);
  for (int i = 0; i < a->iters; ++i) {
    auto g = a->rw.write();
    ++a->value;  // exclusive: no other reader/writer touches value now
  }
}
void rw_reader_body(void* p) {
  auto* a = static_cast<RwArg*>(p);
  for (int i = 0; i < a->iters; ++i) {
    auto g = a->rw.read();
    int now = a->readers_in.fetch_add(1, std::memory_order_acq_rel) + 1;
    int prev = a->reader_peak.load(std::memory_order_relaxed);
    while (now > prev && !a->reader_peak.compare_exchange_weak(
                             prev, now, std::memory_order_relaxed)) {
    }
    // Brief spin INSIDE the read lock so concurrent readers actually overlap in
    // time (a bare read is too fast to catch two holders at once) -- this is what
    // makes reader_peak>=2 observable, proving reads truly share (not serialize).
    volatile long seen = 0;
    for (int k = 0; k < 200; ++k) seen += a->value;
    (void)seen;
    a->readers_in.fetch_sub(1, std::memory_order_acq_rel);
  }
}
// Writers are mutually exclusive AND exclusive vs readers: a shared non-atomic
// counter incremented under write locks totals exactly kWriters*iters (a lost or
// overlapping write both corrupts it and trips TSan). Readers verify-read it under
// read locks concurrently with each other.
TEST(Sync, RwLockReadersShareWriterExclusive) {
  constexpr int kWriters = 2;
  constexpr int kReaders = 8;
  RwArg a;
  a.iters = 300;
  std::vector<std::unique_ptr<Fiber>> fibers;
  fibers.reserve(kWriters + kReaders);
  for (int i = 0; i < kWriters; ++i)
    fibers.push_back(std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                             &rw_writer_body, &a));
  for (int i = 0; i < kReaders; ++i)
    fibers.push_back(std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                             &rw_reader_body, &a));
  Scheduler sched(4, system_allocator(), 1024);
  for (auto& f : fibers) sched.spawn(f.get());
  sched.drain();
  EXPECT_EQ(a.value, static_cast<long>(kWriters) * a.iters);  // no lost writes
}

// Readers genuinely share (not serialized): with NO writers competing, many
// reader fibers holding the lock at once must push the concurrent-reader gauge
// above 1. (Separated from the writer test because dense writers fragment the
// readers into non-overlapping batches, hiding the sharing.)
TEST(Sync, RwLockReadersOverlap) {
  constexpr int kReaders = 8;
  RwArg a;
  a.iters = 200;
  std::vector<std::unique_ptr<Fiber>> fibers;
  fibers.reserve(kReaders);
  for (int i = 0; i < kReaders; ++i)
    fibers.push_back(std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                             &rw_reader_body, &a));
  Scheduler sched(4, system_allocator(), 1024);
  for (auto& f : fibers) sched.spawn(f.get());
  sched.drain();
  EXPECT_GE(a.reader_peak.load(), 2);  // reads overlapped -> truly shared
}

// try_read / try_write reflect the current holder.
TEST(Sync, RwLockTryPaths) {
  RwLock rw;
  {
    auto w = rw.write();
    EXPECT_TRUE(w.owns_lock());
    EXPECT_FALSE(rw.try_read().owns_lock());   // write held -> no read
    EXPECT_FALSE(rw.try_write().owns_lock());  // write held -> no second write
  }
  {
    auto r1 = rw.read();
    EXPECT_TRUE(r1.owns_lock());
    EXPECT_TRUE(rw.try_read().owns_lock());     // reads share
    EXPECT_FALSE(rw.try_write().owns_lock());   // read held -> no write
  }
  EXPECT_TRUE(rw.try_write().owns_lock());  // all released -> write available
}

// ---- oneshot ----------------------------------------------------------------

// A sender fiber sends a value; a receiver fiber recv()s it. The value crosses
// the fiber boundary exactly once.
struct OneshotArg {
  hyle::sched::OneshotSender<int> tx;
  hyle::sched::OneshotReceiver<int> rx;
  std::atomic<int> got{-1};
};
void oneshot_tx_body(void* p) {
  auto* a = static_cast<OneshotArg*>(p);
  (void)std::move(a->tx).send(42);
}
void oneshot_rx_body(void* p) {
  auto* a = static_cast<OneshotArg*>(p);
  auto r = std::move(a->rx).recv();
  a->got.store(r.has_value() ? *r : -1, std::memory_order_release);
}
TEST(Sync, OneshotDeliversValue) {
  auto [tx, rx] = make_oneshot<int>(system_allocator());
  auto a = std::make_unique<OneshotArg>();
  a->tx = std::move(tx);
  a->rx = std::move(rx);
  auto ftx = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                     &oneshot_tx_body, a.get());
  auto frx = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                     &oneshot_rx_body, a.get());
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(frx.get());  // receiver blocks first
  sched.spawn(ftx.get());
  sched.drain();
  EXPECT_EQ(a->got.load(), 42);
}

// Sender dropped without sending -> recv() returns Cancelled (not a hang).
TEST(Sync, OneshotSenderDropCancels) {
  auto [tx, rx] = make_oneshot<int>(system_allocator());
  { auto dead = std::move(tx); }  // drop the sender without sending
  auto r = std::move(rx).recv();
  EXPECT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsCancelled());
}

// Receiver dropped -> send() returns the value back as an error.
TEST(Sync, OneshotReceiverDropRejectsSend) {
  auto [tx, rx] = make_oneshot<int>(system_allocator());
  { auto dead = std::move(rx); }  // drop the receiver
  auto r = std::move(tx).send(7);
  EXPECT_FALSE(r.has_value());  // nobody home
}

// ---- watch ------------------------------------------------------------------

// borrow() returns the seeded value immediately; after a send, a receiver's
// changed() wakes and borrow() reflects the new value. Models a shutdown flag.
struct WatchArg {
  hyle::sched::WatchReceiver<int> rx;
  std::atomic<int> observed{-1};
  std::atomic<bool> changed_ok{false};
};
void watch_rx_body(void* p) {
  auto* a = static_cast<WatchArg*>(p);
  auto r = a->rx.changed();  // block until the first send
  a->changed_ok.store(r.has_value(), std::memory_order_release);
  a->observed.store(a->rx.borrow(), std::memory_order_release);
}
TEST(Sync, WatchSeesLatestValue) {
  auto [tx, rx] = make_watch<int>(system_allocator(), 0);
  EXPECT_EQ(rx.borrow(), 0);  // late subscriber sees current immediately
  auto a = std::make_unique<WatchArg>();
  a->rx = rx;  // copy (watch receivers are copyable)
  auto frx = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                     &watch_rx_body, a.get());
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(frx.get());
  std::this_thread::sleep_for(std::chrono::milliseconds(10));  // receiver blocks
  tx.send(99);
  sched.drain();
  EXPECT_TRUE(a->changed_ok.load());
  EXPECT_EQ(a->observed.load(), 99);
}

// All senders dropped -> changed() returns Cancelled.
TEST(Sync, WatchSenderDropCloses) {
  auto [tx, rx] = make_watch<int>(system_allocator(), 5);
  { auto dead = std::move(tx); }  // drop the only sender
  auto r = rx.changed();
  EXPECT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsCancelled());
}

// ---- broadcast --------------------------------------------------------------

// Two receivers each get EVERY value a sender broadcasts (fan-out).
struct BcastArg {
  hyle::sched::BroadcastReceiver<int> rx;
  int n;
  std::atomic<long> sum{0};
};
void bcast_rx_body(void* p) {
  auto* a = static_cast<BcastArg*>(p);
  int got = 0;
  while (got < a->n) {
    auto r = a->rx.recv();
    if (r.has_value()) {
      a->sum.fetch_add(*r, std::memory_order_acq_rel);
      ++got;
    } else if (r.error().IsCancelled()) {
      break;  // closed
    }
    // Unavailable (lagged): just retry.
  }
}
TEST(Sync, BroadcastFanOutToAll) {
  constexpr int kN = 50;
  auto [tx, rx0] = make_broadcast<int>(system_allocator(), 64);  // cap > N: no lag
  auto rx1 = tx.subscribe();
  auto a0 = std::make_unique<BcastArg>();
  auto a1 = std::make_unique<BcastArg>();
  a0->rx = rx0;
  a0->n = kN;
  a1->rx = rx1;
  a1->n = kN;
  auto f0 = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                    &bcast_rx_body, a0.get());
  auto f1 = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                    &bcast_rx_body, a1.get());
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(f0.get());
  sched.spawn(f1.get());
  std::this_thread::sleep_for(std::chrono::milliseconds(10));  // receivers block
  long expect = 0;
  for (int i = 1; i <= kN; ++i) {
    tx.send(i);
    expect += i;
  }
  sched.drain();
  EXPECT_EQ(a0->sum.load(), expect);  // receiver 0 saw every value
  EXPECT_EQ(a1->sum.load(), expect);  // receiver 1 saw every value too
}

// A slow receiver that falls behind the ring window gets a Lagged (Unavailable)
// signal rather than silently missing values or blocking the sender.
TEST(Sync, BroadcastLaggedOnOverflow) {
  auto [tx, rx] = make_broadcast<int>(system_allocator(), 4);  // tiny ring
  for (int i = 0; i < 10; ++i) tx.send(i);  // 10 sends into a 4-slot ring
  // The receiver never read; its cursor (0) is now far behind. First recv must
  // report the lag and fast-forward, not return stale/garbage.
  auto r = rx.recv();
  EXPECT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsUnavailable());
  // After the lag signal, subsequent recv() yields retained values.
  auto r2 = rx.recv();
  EXPECT_TRUE(r2.has_value());
  EXPECT_EQ(*r2, 6);  // oldest retained = write_pos(10) - cap(4)
}

}  // namespace
