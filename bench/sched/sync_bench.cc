// Fiber-aware sync microbenchmarks (sched-05): HYLE's own costs, measured
// honestly, with std baselines where a fair one exists.
//
//  1. UNCONTENDED Mutex lock/unlock vs std::mutex -- the raw primitive cost.
//     HYLE's Mutex is a futex (single CAS lock, single fetch_sub unlock), same
//     family as std::mutex. Any residual gap is largely glibc's single-threaded
//     atomic-elision (__libc_single_threaded) in this single-threaded microbench;
//     under real contention (see the marl bench) HYLE matches std/marl.
//  2. CONTENDED throughput: N fibers on W workers hammering one Mutex. The futex
//     parks the WORKER on contention (like std::mutex/marl), not the fiber -- a
//     mutex guards short sections where parking beats yielding.
//  3. Channel ping-pong: 1 producer + 1 consumer fiber, throughput per item.
//  4. WaitGroup fork-join: spawn N fibers that done(), one waiter -- the cost of
//     the fork-join sync itself.
//
// SOTA parity vs marl (the fiber-aware-sync reference this design converged from)
// needs marl's own scheduler wired in -- registered as bench-02d, not here.

#include <nanobench.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <latch>
#include <memory>
#include <mutex>
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

// ---- (2) contended mutex: each fiber locks/unlocks `iters` times ----
struct MutexHammer {
  Mutex m;
  std::uint64_t counter = 0;
  int iters;
};
void mutex_hammer_body(void* p) {
  auto* a = static_cast<MutexHammer*>(p);
  for (int i = 0; i < a->iters; ++i) {
    a->m.lock();
    ++a->counter;
    a->m.unlock();
  }
}

// ---- (3) channel ping-pong ----
struct ChanPair {
  Channel<int>* ch;
  int n;
};
void chan_prod(void* p) {
  auto* a = static_cast<ChanPair*>(p);
  for (int i = 0; i < a->n; ++i) (void)a->ch->send(i);
  a->ch->close();
}
void chan_cons(void* p) {
  auto* a = static_cast<ChanPair*>(p);
  int v;
  std::uint64_t sink = 0;
  while (a->ch->recv(v)) sink += static_cast<std::uint64_t>(v);
  ankerl::nanobench::doNotOptimizeAway(sink);
}

// ---- (4) waitgroup fork-join ----
struct WgFork {
  WaitGroup* wg;
};
void wg_done_body(void* p) { static_cast<WgFork*>(p)->wg->done(); }
void wg_wait_body(void* p) { static_cast<WgFork*>(p)->wg->wait(); }

// ---- (5) latch fork-join (one-shot countdown, std::latch shape) ----
struct LatchFork {
  Latch* l;
};
void latch_down_body(void* p) { static_cast<LatchFork*>(p)->l->count_down(); }
void latch_wait_body(void* p) { static_cast<LatchFork*>(p)->l->wait(); }

// ---- (6) CV wait+notify handoff: a consumer fiber BLOCKS in cv.wait until a
// producer fiber flips a counter + notifies. N rounds of real block->notify->
// resume, the CV's blocking path (the notify_one fast-path bench only covers the
// no-waiter case). Monitor lock is a HYLE Mutex (BasicLockable, what wait() takes).
// Single shared turn counter under the lock; classic monitor predicate loop.
struct CvPair {
  Mutex mu;
  ConditionVariable cv;
  long produced = 0;   // bumped by producer under mu
  long consumed = 0;   // bumped by consumer under mu
  int rounds;
};
void cv_producer(void* p) {
  auto* s = static_cast<CvPair*>(p);
  for (int i = 0; i < s->rounds; ++i) {
    s->mu.lock();
    ++s->produced;
    s->cv.notify_one();                       // wake the consumer
    // wait until the consumer has caught up before producing the next item
    s->cv.wait(s->mu, [&] { return s->consumed > s->produced - 1; });
    s->mu.unlock();
  }
}
void cv_consumer(void* p) {
  auto* s = static_cast<CvPair*>(p);
  for (int i = 0; i < s->rounds; ++i) {
    s->mu.lock();
    s->cv.wait(s->mu, [&] { return s->produced > s->consumed; });  // BLOCK here
    ++s->consumed;
    s->cv.notify_one();                       // let the producer proceed
    s->mu.unlock();
  }
}

}  // namespace

int main() {
  // ---- 1. uncontended Mutex lock/unlock: HYLE futex Mutex vs std::mutex ----
  // Uncontended lock() is a single CAS (no syscall, no scheduler) -- measured
  // from the calling thread, the raw primitive cost.
  {
    constexpr int kIters = 5000;
    ankerl::nanobench::Bench b;
    b.title("uncontended lock/unlock pair").relative(true).batch(kIters).minEpochIterations(200);
    b.run("std::mutex", [&] {
      std::mutex m;
      for (int i = 0; i < kIters; ++i) {
        m.lock();
        ankerl::nanobench::doNotOptimizeAway(&m);
        m.unlock();
      }
    });
    b.run("HYLE Mutex (platform fast lock)", [&] {
      Mutex m;
      for (int i = 0; i < kIters; ++i) {
        m.lock();
        ankerl::nanobench::doNotOptimizeAway(&m);
        m.unlock();
      }
    });
  }

  // ---- 1b. SAME, but with a live background thread (defeats glibc's
  // __libc_single_threaded atomic-elision). If std::mutex slows to HYLE's level
  // here, the "2.8x slower" above was a single-threaded-microbench artifact, not
  // a real gap -- std::mutex/marl pay the same atomic once the process is MT.
  {
    constexpr int kIters = 5000;
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> spin{0};
    std::thread keepalive([&] {  // exists only to make the process multi-threaded
      while (!stop.load(std::memory_order_relaxed))
        spin.fetch_add(1, std::memory_order_relaxed);
    });
    ankerl::nanobench::Bench b;
    b.title("uncontended lock/unlock pair (MT: bg thread alive)")
        .relative(true).batch(kIters).minEpochIterations(200);
    b.run("std::mutex (MT)", [&] {
      std::mutex m;
      for (int i = 0; i < kIters; ++i) {
        m.lock();
        ankerl::nanobench::doNotOptimizeAway(&m);
        m.unlock();
      }
    });
    b.run("HYLE Mutex (platform fast lock, MT)", [&] {
      Mutex m;
      for (int i = 0; i < kIters; ++i) {
        m.lock();
        ankerl::nanobench::doNotOptimizeAway(&m);
        m.unlock();
      }
    });
    stop.store(true, std::memory_order_relaxed);
    keepalive.join();
  }

  // ---- 2. contended Mutex throughput: 8 fibers / 4 workers ----
  {
    constexpr int kFibers = 8;
    constexpr int kIters = 2000;  // per fiber
    ankerl::nanobench::Bench b;
    b.title("contended Mutex (8 fibers/4 workers)").relative(true).batch(static_cast<std::uint64_t>(kFibers) * kIters).minEpochIterations(20);
    b.run("HYLE Mutex (platform fast lock) lock/inc/unlock", [&] {
      MutexHammer h;
      h.iters = kIters;
      std::vector<std::unique_ptr<Fiber>> fibers;
      fibers.reserve(kFibers);
      for (int i = 0; i < kFibers; ++i)
        fibers.push_back(std::make_unique<Fiber>(
            kDefaultStackSize, &mutex_hammer_body, &h));
      Scheduler sched(4, system_allocator(), 1024);
      for (auto& f : fibers) sched.spawn(f.get());
      sched.drain();
      ankerl::nanobench::doNotOptimizeAway(h.counter);
    });
  }

  // ---- 3. Channel ping-pong throughput (cap 16) ----
  {
    constexpr int kN = 5000;
    ankerl::nanobench::Bench b;
    b.title("Channel send+recv throughput").relative(true).batch(kN).minEpochIterations(20);
    b.run("HYLE Channel<int> 1p1c", [&] {
      Channel<int> ch(system_allocator(), 16);
      ChanPair a{&ch, kN};
      auto prod =
          std::make_unique<Fiber>(kDefaultStackSize, &chan_prod, &a);
      auto cons =
          std::make_unique<Fiber>(kDefaultStackSize, &chan_cons, &a);
      Scheduler sched(2, system_allocator(), 256);
      sched.spawn(prod.get());
      sched.spawn(cons.get());
      sched.drain();
    });
  }

  // ---- 4. WaitGroup fork-join: 64 workers + 1 waiter ----
  {
    constexpr int kWorkers = 64;
    ankerl::nanobench::Bench b;
    b.title("WaitGroup fork-join (64 workers)").relative(true).batch(kWorkers).minEpochIterations(50);
    b.run("HYLE WaitGroup add/done/wait", [&] {
      WaitGroup wg;
      wg.add(kWorkers);
      WgFork f{&wg};
      std::vector<std::unique_ptr<Fiber>> fibers;
      fibers.reserve(kWorkers + 1);
      fibers.push_back(std::make_unique<Fiber>(
          kDefaultStackSize, &wg_wait_body, &f));
      for (int i = 0; i < kWorkers; ++i)
        fibers.push_back(std::make_unique<Fiber>(
            kDefaultStackSize, &wg_done_body, &f));
      Scheduler sched(4, system_allocator(), 1024);
      for (auto& fb : fibers) sched.spawn(fb.get());
      sched.drain();
    });
  }

  // ---- 5. Latch fork-join: 64 fibers count_down + 1 waiter (vs std::latch) ----
  // HYLE Latch is the one-shot countdown (std::latch shape) on the fiber CV. The
  // std::latch rival runs the SAME shape on OS threads (the std primitive's only
  // mode) -- not apples-to-apples on the execution substrate, but it's the std
  // baseline for "wait until N done". The HYLE row is the fiber-native cost.
  {
    constexpr int kWorkers = 64;
    ankerl::nanobench::Bench b;
    b.title("Latch fork-join (64 down + 1 wait)").relative(true).batch(kWorkers).minEpochIterations(50);
    b.run("std::latch (64 threads)", [&] {
      std::latch latch(kWorkers);
      std::vector<std::thread> ts;
      ts.reserve(kWorkers);
      for (int i = 0; i < kWorkers; ++i)
        ts.emplace_back([&] { latch.count_down(); });
      latch.wait();
      for (auto& t : ts) t.join();
    });
    b.run("HYLE Latch (64 fibers, fiber-aware)", [&] {
      Latch latch(kWorkers);
      LatchFork f{&latch};
      std::vector<std::unique_ptr<Fiber>> fibers;
      fibers.reserve(kWorkers + 1);
      fibers.push_back(std::make_unique<Fiber>(
          kDefaultStackSize, &latch_wait_body, &f));
      for (int i = 0; i < kWorkers; ++i)
        fibers.push_back(std::make_unique<Fiber>(
            kDefaultStackSize, &latch_down_body, &f));
      Scheduler sched(4, system_allocator(), 1024);
      for (auto& fb : fibers) sched.spawn(fb.get());
      sched.drain();
    });
  }

  // ---- 6. ConditionVariable notify with NO waiter (the uncontended fast path) -
  // CV is the foundation under Mutex/WaitGroup/Channel/Latch; its notify fast path
  // (no parked waiter -> skip the internal lock entirely, per the header) is hit on
  // every uncontended unlock/done/send. This measures that bare notify cost vs
  // std::condition_variable's notify_one with no waiter. No fibers needed -- it's a
  // pure primitive call. std::cv needs the mutex held convention; HYLE notify is
  // standalone (caller holds the monitor lock by contract, none needed when empty).
  {
    constexpr int kIters = 10000;
    ankerl::nanobench::Bench b;
    b.title("CV notify_one, no waiter (fast path)").relative(true).batch(kIters).minEpochIterations(500);
    b.run("std::condition_variable::notify_one", [&] {
      std::condition_variable cv;
      for (int i = 0; i < kIters; ++i) {
        cv.notify_one();
        ankerl::nanobench::doNotOptimizeAway(&cv);
      }
    });
    b.run("HYLE ConditionVariable::notify_one", [&] {
      ConditionVariable cv;
      for (int i = 0; i < kIters; ++i) {
        cv.notify_one();
        ankerl::nanobench::doNotOptimizeAway(&cv);
      }
    });
  }

  // ---- 7. CV wait+notify HANDOFF: consumer fiber blocks in cv.wait, producer
  // notifies (the BLOCKING path, vs the no-waiter fast path above). Two fibers do
  // `kRounds` block->notify->resume handoffs under one Scheduler, retired by
  // drain() (the safe harness -- same as the suspend/wake spine). 1 worker so the
  // wait genuinely blocks and hands off (not parallel). No std rival on fibers;
  // the number is HYLE's fiber-aware CV blocking round-trip cost. batch=kRounds.
  {
    constexpr int kRounds = 5000;
    ankerl::nanobench::Bench b;
    b.title("CV wait+notify handoff (consumer blocks)").relative(true).batch(kRounds).minEpochIterations(20);
    b.run("HYLE ConditionVariable wait/notify ping-pong", [&] {
      CvPair s;
      s.rounds = kRounds;
      auto cons =
          std::make_unique<Fiber>(kDefaultStackSize, &cv_consumer, &s);
      auto prod =
          std::make_unique<Fiber>(kDefaultStackSize, &cv_producer, &s);
      Scheduler sched(1, system_allocator(), 256);  // 1 worker: real block/handoff
      sched.spawn(cons.get());
      sched.spawn(prod.get());
      sched.drain();
      ankerl::nanobench::doNotOptimizeAway(s.consumed);
    });
  }

  // ---- 8. TIMED CV wait, NOTIFY-BEFORE-DEADLINE latency (sched-04c thread-mode) -
  // The common timed-wait case: a waiter blocks with a generous deadline, a
  // notifier wakes it well before -- so we measure the wake ROUND-TRIP (block ->
  // notify -> return kNoTimeout), NOT the OS timer granularity. HYLE
  // CV::wait_for (thread-mode: park_timeout + linked-flag arbitration) vs
  // std::condition_variable::wait_for (the direct SOTA baseline; both park the OS
  // thread with a futex deadline). One waiter thread, main notifies each round.
  // batch=rounds. Bare threads (thread-mode), no scheduler.
  {
    constexpr int kRounds = 2000;
    ankerl::nanobench::Bench b;
    b.title("timed CV wait: notify-before-deadline round-trip")
        .relative(true)
        .batch(kRounds)
        .minEpochIterations(5);

    b.run("std::condition_variable::wait_for", [&] {
      std::mutex mu;
      std::condition_variable cv;
      bool ready = false;
      bool done = false;
      std::thread waiter([&] {
        for (int i = 0; i < kRounds; ++i) {
          std::unique_lock<std::mutex> lk(mu);
          cv.wait_for(lk, std::chrono::seconds(10), [&] { return ready; });
          ready = false;
          done = true;
          cv.notify_one();  // ack so the main loop can proceed to the next round
        }
      });
      for (int i = 0; i < kRounds; ++i) {
        std::unique_lock<std::mutex> lk(mu);
        ready = true;
        cv.notify_one();
        cv.wait(lk, [&] { return done; });
        done = false;
      }
      waiter.join();
    });

    b.run("HYLE ConditionVariable::wait_for (thread-mode)", [&] {
      SpinLock mu;
      ConditionVariable to_waiter;   // main -> waiter
      ConditionVariable to_main;     // waiter -> main (ack)
      bool ready = false;
      bool done = false;
      std::thread waiter([&] {
        for (int i = 0; i < kRounds; ++i) {
          std::lock_guard<SpinLock> lk(mu);
          to_waiter.wait_for(mu, std::chrono::seconds(10),
                             [&] { return ready; });
          ready = false;
          done = true;
          to_main.notify_one();
        }
      });
      for (int i = 0; i < kRounds; ++i) {
        std::lock_guard<SpinLock> lk(mu);
        ready = true;
        to_waiter.notify_one();
        to_main.wait(mu, [&] { return done; });
        done = false;
      }
      waiter.join();
    });
  }

  // ---- 9. TIMED CV wait, PURE TIMEOUT path cost (sched-04c thread-mode) --------
  // No notifier ever comes: measure the fixed overhead AROUND the OS park (arm the
  // node, park_timeout, retract on deadline) using a tiny 1ms deadline. This is
  // dominated by the OS timer floor (both HYLE and std park on the same futex/OS
  // deadline), so the two should be ~equal -- the point is to show HYLE adds no
  // extra overhead vs std on the timeout path, not to "win" the OS clock. batch=1.
  {
    ankerl::nanobench::Bench b;
    b.title("timed CV wait: pure timeout (1ms, no notify)")
        .relative(true)
        .batch(1)
        .minEpochIterations(50);
    b.run("std::condition_variable::wait_for", [&] {
      std::mutex mu;
      std::condition_variable cv;
      std::unique_lock<std::mutex> lk(mu);
      cv.wait_for(lk, std::chrono::milliseconds(1));
    });
    b.run("HYLE ConditionVariable::wait_for (thread-mode)", [&] {
      SpinLock mu;
      ConditionVariable cv;
      std::lock_guard<SpinLock> lk(mu);
      cv.wait_for(mu, std::chrono::milliseconds(1));
    });
  }

  return 0;
}
