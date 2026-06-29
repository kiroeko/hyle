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

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"

using hyle::mem::system_allocator;
using hyle::sched::Channel;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Mutex;
using hyle::sched::Scheduler;
using hyle::sched::WaitGroup;

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
  for (int i = 0; i < a->n; ++i) a->ch->send(i);
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
    b.run("HYLE Mutex (futex)", [&] {
      Mutex m;
      for (int i = 0; i < kIters; ++i) {
        m.lock();
        ankerl::nanobench::doNotOptimizeAway(&m);
        m.unlock();
      }
    });
  }

  // ---- 2. contended Mutex throughput: 8 fibers / 4 workers ----
  {
    constexpr int kFibers = 8;
    constexpr int kIters = 2000;  // per fiber
    ankerl::nanobench::Bench b;
    b.title("contended Mutex (8 fibers/4 workers)").batch(static_cast<std::uint64_t>(kFibers) * kIters).minEpochIterations(20);
    b.run("HYLE Mutex (futex) lock/inc/unlock", [&] {
      MutexHammer h;
      h.iters = kIters;
      std::vector<std::unique_ptr<Fiber>> fibers;
      fibers.reserve(kFibers);
      for (int i = 0; i < kFibers; ++i)
        fibers.push_back(std::make_unique<Fiber>(system_allocator(),
                                                 kDefaultStackSize,
                                                 &mutex_hammer_body, &h));
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
    b.title("Channel send+recv throughput").batch(kN).minEpochIterations(20);
    b.run("HYLE Channel<int> 1p1c", [&] {
      Channel<int> ch(system_allocator(), 16);
      ChanPair a{&ch, kN};
      auto prod = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                          &chan_prod, &a);
      auto cons = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                          &chan_cons, &a);
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
    b.title("WaitGroup fork-join (64 workers)").batch(kWorkers).minEpochIterations(50);
    b.run("HYLE WaitGroup add/done/wait", [&] {
      WaitGroup wg;
      wg.add(kWorkers);
      WgFork f{&wg};
      std::vector<std::unique_ptr<Fiber>> fibers;
      fibers.reserve(kWorkers + 1);
      fibers.push_back(std::make_unique<Fiber>(system_allocator(),
                                               kDefaultStackSize, &wg_wait_body,
                                               &f));
      for (int i = 0; i < kWorkers; ++i)
        fibers.push_back(std::make_unique<Fiber>(system_allocator(),
                                                 kDefaultStackSize,
                                                 &wg_done_body, &f));
      Scheduler sched(4, system_allocator(), 1024);
      for (auto& fb : fibers) sched.spawn(fb.get());
      sched.drain();
    });
  }

  return 0;
}
