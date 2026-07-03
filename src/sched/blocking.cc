#include "hyle/sched/blocking.h"

#include <condition_variable>
#include <mutex>
#include <vector>

#include "hyle/macros.h"
#include "hyle/platform/thread.h"

namespace hyle::sched {

// The pool's own thread management uses std::mutex/condition_variable ON PURPOSE:
// pool threads are meant to BLOCK (that is their whole job), so there is no fiber
// to yield -- the fiber-aware primitives would be the wrong tool here. This mirrors
// tokio's blocking::pool, which also uses a parking_lot Mutex/Condvar, not its
// async primitives.
struct BlockingPool::Impl {
  mem::Allocator alloc;
  const unsigned max_threads;
  const unsigned long long keep_alive_ns;

  std::mutex mu;
  std::condition_variable cv;

  // Intrusive FIFO job queue (head pop, tail push). BlockingJob carries `next`,
  // so the queue needs no per-job container allocation.
  detail::BlockingJob* head = nullptr;
  detail::BlockingJob* tail = nullptr;

  unsigned num_threads = 0;   // live pool threads
  unsigned num_idle = 0;      // threads currently waiting on cv (no job in hand)
  unsigned num_notify = 0;    // pending notifications, paired against spurious wakeups
  bool shutting_down = false;

  // Live thread handles, joined at shutdown. A retired-on-keep-alive thread leaves
  // its handle here for shutdown to join (it has already exited, so join returns
  // at once); this keeps handle ownership simple (one joiner: shutdown).
  std::vector<platform::Thread> threads;

  Impl(mem::Allocator a, unsigned max_t, unsigned long long keep_ns)
      : alloc(a), max_threads(max_t), keep_alive_ns(keep_ns) {
    HYLE_FATAL_ASSERT(max_t >= 1);
  }

  static void thread_main(void* p) noexcept {
    static_cast<Impl*>(p)->run_worker();
  }

  // Pop the next job under the lock (caller holds `lk`). Returns nullptr if empty.
  detail::BlockingJob* pop_locked() noexcept {
    detail::BlockingJob* j = head;
    if (j != nullptr) {
      head = j->next;
      if (head == nullptr) tail = nullptr;
      j->next = nullptr;
    }
    return j;
  }

  void run_worker() noexcept {
    std::unique_lock<std::mutex> lk(mu);
    for (;;) {
      // Drain available jobs first (before considering idle/exit).
      if (detail::BlockingJob* job = pop_locked(); job != nullptr) {
        lk.unlock();          // MUST release: the job blocks; holding mu deadlocks the pool
        job->run();
        job->destroy();
        lk.lock();
        continue;
      }
      if (shutting_down) break;  // no job + told to stop -> exit

      // No work: wait as an idle thread, bounded by keep_alive so a burst doesn't
      // pin threads forever. num_notify pairs a real wake against a spurious one.
      ++num_idle;
      const auto timeout = std::chrono::nanoseconds(keep_alive_ns);
      const std::cv_status st = cv.wait_for(lk, timeout);
      // Consume a pending notification if one was aimed at us.
      bool got_notify = false;
      if (num_notify > 0) {
        --num_notify;
        got_notify = true;
      }
      --num_idle;

      if (shutting_down) break;  // shutdown takes priority over a timeout exit
      if (!got_notify && st == std::cv_status::timeout) {
        // Idle past keep-alive with no work aimed at us: retire. Our handle stays
        // in `threads` for shutdown to join (we've exited, join is immediate).
        --num_threads;
        break;
      }
      // Woken with intent (or spurious but a job may be present): loop to re-check.
    }
  }

  // Submit under the lock: enqueue, then either wake an idle thread or spawn a new
  // one (up to the cap). Returns false if shutting down.
  bool submit(detail::BlockingJob* job) noexcept {
    std::unique_lock<std::mutex> lk(mu);
    if (shutting_down) return false;
    // Enqueue at the tail.
    job->next = nullptr;
    if (tail == nullptr) {
      head = job;
    } else {
      tail->next = job;
    }
    tail = job;

    if (num_idle > 0) {
      // Wake one idle thread to take it.
      ++num_notify;
      cv.notify_one();
      return true;
    }
    if (num_threads < max_threads) {
      // No idle thread and under the cap: spawn a new one. If the OS refuses but we
      // already have >=1 thread, the job simply waits for a busy thread to free up.
      platform::ThreadOpts opts;
      opts.name = "hyle-blocking";
      auto r = platform::Thread::spawn(&Impl::thread_main, this, opts);
      if (r.has_value()) {
        threads.push_back(std::move(*r));
        ++num_threads;
      } else {
        HYLE_FATAL_ASSERT(num_threads >= 1 &&
                          "blocking pool: OS refused first thread");
      }
      return true;
    }
    // At the cap with all threads busy: the job waits in the queue; a busy thread
    // will pop it when it finishes its current job.
    return true;
  }

  void shutdown() {
    std::vector<platform::Thread> to_join;
    detail::BlockingJob* leftover = nullptr;
    {
      std::unique_lock<std::mutex> lk(mu);
      if (shutting_down) {
        // Already shut down; nothing to join again (threads moved out last time).
        return;
      }
      shutting_down = true;
      // Take the not-yet-started jobs to cancel them outside the lock.
      leftover = head;
      head = tail = nullptr;
      // Wake every idle thread so they observe shutting_down and exit.
      num_notify += num_idle;
      cv.notify_all();
      to_join.swap(threads);
    }
    // Cancel queued-but-unstarted jobs: cancel() publishes Cancelled on the
    // JoinState WITHOUT running the closure, then destroy() frees the job (and
    // drops the pool-side ref). The joiner's JoinHandle then resolves to Cancelled.
    while (leftover != nullptr) {
      detail::BlockingJob* next = leftover->next;
      leftover->cancel();
      leftover->destroy();
      leftover = next;
    }
    // Join all threads (running jobs finish first; retired threads join instantly).
    for (auto& t : to_join) {
      if (t.joinable()) t.join();
    }
  }

  unsigned live_threads() noexcept {
    std::lock_guard<std::mutex> lk(mu);
    return num_threads;
  }
};

// ---- BlockingPool public surface --------------------------------------------

BlockingPool::BlockingPool(mem::Allocator alloc, unsigned max_threads,
                           unsigned long long keep_alive_ns)
    : impl_(new Impl(alloc, max_threads, keep_alive_ns)) {}

BlockingPool::~BlockingPool() {
  impl_->shutdown();
  delete impl_;
}

void BlockingPool::shutdown() { impl_->shutdown(); }

unsigned BlockingPool::live_threads() const noexcept {
  return impl_->live_threads();
}

bool BlockingPool::submit(detail::BlockingJob* job) noexcept {
  return impl_->submit(job);
}

}  // namespace hyle::sched
