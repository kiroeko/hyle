#include "hyle/sched/blocking.h"

#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#include "hyle/core/compiler.h"
#include "hyle/platform/thread.h"

namespace hyle::sched {

namespace {

// A blocking job may call arbitrary user code. If that code tries to shut down
// the pool currently executing it, joining the current OS thread is impossible
// and would otherwise surface as a platform-dependent deadlock/system_error.
// Keep this as a contract check instead of silently leaving a half-shut pool.
thread_local const void* tls_blocking_pool = nullptr;

}  // namespace

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
  bool shutting_down = false;
  bool shutdown_complete = false;

  struct ThreadRecord {
    Impl* owner = nullptr;
    platform::Thread thread;
    bool retired = false;  // guarded by mu
  };
  // Stable records let a worker publish its own retirement. The next submission
  // extracts and joins retired handles before growing the pool, preventing
  // unbounded handle retention across repeated burst/idle cycles.
  std::vector<std::unique_ptr<ThreadRecord>> threads;

  Impl(mem::Allocator a, unsigned max_t, unsigned long long keep_ns)
      : alloc(a), max_threads(max_t), keep_alive_ns(keep_ns) {
    HYLE_FATAL_ASSERT(max_t >= 1);
    // submit() is noexcept. Reserve the bounded handle set here, where a
    // construction failure may propagate to the owning factory/constructor.
    threads.reserve(max_threads);
  }

  static void thread_main(void* p) noexcept {
    auto* rec = static_cast<ThreadRecord*>(p);
    rec->owner->run_worker(rec);
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

  void run_worker(ThreadRecord* record) noexcept {
    HYLE_FATAL_ASSERT(tls_blocking_pool == nullptr);
    tls_blocking_pool = this;
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
      using Rep = std::chrono::nanoseconds::rep;
      const auto bounded_ns = static_cast<Rep>(
          keep_alive_ns > static_cast<unsigned long long>(
                              std::numeric_limits<Rep>::max())
              ? std::numeric_limits<Rep>::max()
              : keep_alive_ns);
      const bool ready = cv.wait_for(
          lk, std::chrono::nanoseconds(bounded_ns),
          [&] { return shutting_down || head != nullptr; });
      --num_idle;

      if (shutting_down) break;  // shutdown takes priority over a timeout exit
      if (!ready) {
        // Idle past keep-alive with no work aimed at us: retire. A later submit
        // reaps the exited handle before it creates a replacement.
        break;
      }
      // Predicate says a job is present: loop to pop it.
    }
    --num_threads;
    record->retired = true;
    lk.unlock();
    tls_blocking_pool = nullptr;
  }

  void reap_retired() noexcept {
    // Extract and join one at a time: no temporary vector allocation in this
    // noexcept path. A retired worker published the flag under `mu` immediately
    // before returning, so after we acquire/release `mu` its handle is joinable.
    for (;;) {
      std::unique_ptr<ThreadRecord> retired;
      {
        std::lock_guard<std::mutex> lk(mu);
        std::size_t found = threads.size();
        for (std::size_t i = 0; i < threads.size(); ++i) {
          if (!threads[i]->retired) continue;
          found = i;
          break;
        }
        if (found == threads.size()) return;
        retired = std::move(threads[found]);
        threads[found] = std::move(threads.back());
        threads.pop_back();
      }
      if (retired->thread.joinable()) retired->thread.join();
    }
  }

  // Submit under the lock: enqueue, then either wake an idle thread or spawn a new
  // one (up to the cap). Returns false if shutting down.
  bool submit(detail::BlockingJob* job) noexcept {
    reap_retired();
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
      cv.notify_one();
      return true;
    }
    if (num_threads < max_threads) {
      // No idle thread and under the cap: spawn a new one. If the OS refuses but we
      // already have >=1 thread, the job simply waits for a busy thread to free up.
      platform::ThreadOpts opts;
      opts.name = "hyle-blocking";
      std::unique_ptr<ThreadRecord> record(
          new (std::nothrow) ThreadRecord());
      if (!record) {
        if (num_threads == 0) {
          HYLE_FATAL_ASSERT(head == job && tail == job);
          head = tail = nullptr;
          return false;
        }
        return true;  // an existing busy worker will eventually take the job
      }
      record->owner = this;
      auto r = platform::Thread::spawn(&Impl::thread_main, record.get(), opts);
      if (r.has_value()) {
        record->thread = std::move(*r);
        threads.push_back(std::move(record));
        ++num_threads;
      } else if (num_threads == 0) {
        // No worker exists to drain the just-enqueued task. Retract it and let
        // submit's caller publish a normal Cancelled/resource failure.
        HYLE_FATAL_ASSERT(head == job && tail == job);
        head = tail = nullptr;
        return false;
      }
      return true;
    }
    // At the cap with all threads busy: the job waits in the queue; a busy thread
    // will pop it when it finishes its current job.
    return true;
  }

  void shutdown() {
    // A pool worker cannot join itself. Calling shutdown from a blocking task is
    // a programming error; request shutdown from its owner/scheduler instead.
    HYLE_FATAL_ASSERT(tls_blocking_pool != this);

    std::vector<std::unique_ptr<ThreadRecord>> to_join;
    detail::BlockingJob* leftover = nullptr;
    {
      std::unique_lock<std::mutex> lk(mu);
      if (shutting_down) {
        // Another caller may still be cancelling queued jobs and joining live
        // workers outside `mu`. Idempotent shutdown is a completion barrier, not
        // merely a request edge: do not let a concurrent destructor race ahead.
        cv.wait(lk, [&] { return shutdown_complete; });
        return;
      }
      shutting_down = true;
      // Take the not-yet-started jobs to cancel them outside the lock.
      leftover = head;
      head = tail = nullptr;
      // Wake every idle thread so they observe shutting_down and exit.
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
    for (auto& record : to_join) {
      if (record->thread.joinable()) record->thread.join();
    }
    {
      std::lock_guard<std::mutex> lk(mu);
      shutdown_complete = true;
    }
    cv.notify_all();
  }

  unsigned live_threads() noexcept {
    std::lock_guard<std::mutex> lk(mu);
    return num_threads;
  }

  std::size_t retained_thread_handles() noexcept {
    std::lock_guard<std::mutex> lk(mu);
    return threads.size();
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

std::size_t BlockingPool::retained_thread_handles() const noexcept {
  return impl_->retained_thread_handles();
}

bool BlockingPool::submit(detail::BlockingJob* job) noexcept {
  return impl_->submit(job);
}

}  // namespace hyle::sched
