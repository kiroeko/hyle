#include "hyle/sched/blocking.h"

#include <condition_variable>
#include <mutex>
#include <new>
#include <stdexcept>
#include <vector>

#include "hyle/core/compiler.h"
#include "hyle/mem/std_allocator.h"
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
  mem::AllocatorRef alloc;
  const unsigned max_threads;
  const time::Duration keep_alive;

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
  std::vector<ThreadRecord*, mem::StdAllocator<ThreadRecord*>> threads;

  Impl(mem::AllocatorRef a, unsigned max_t, time::Duration keep_alive_duration)
      : alloc(a),
        max_threads(max_t),
        keep_alive(keep_alive_duration > time::Duration::zero()
                       ? keep_alive_duration
                       : time::Duration::zero()),
        threads(mem::StdAllocator<ThreadRecord*>(a)) {
    HYLE_FATAL_ASSERT(max_t >= 1);
    // submit() is noexcept. Reserve the bounded handle set here, where a
    // construction failure may propagate to the owning factory/constructor.
    threads.reserve(max_threads);
  }

  ~Impl() {
    HYLE_FATAL_ASSERT(threads.empty());
    HYLE_FATAL_ASSERT(head == nullptr && tail == nullptr);
  }

  void destroy_record(ThreadRecord* record) noexcept {
    if (record == nullptr) return;
    record->~ThreadRecord();
    alloc.deallocate(record, sizeof(ThreadRecord), alignof(ThreadRecord));
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
      const bool ready = cv.wait_for(
          lk, keep_alive,
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
      ThreadRecord* retired = nullptr;
      {
        std::lock_guard<std::mutex> lk(mu);
        std::size_t found = threads.size();
        for (std::size_t i = 0; i < threads.size(); ++i) {
          if (!threads[i]->retired) continue;
          found = i;
          break;
        }
        if (found == threads.size()) return;
        retired = threads[found];
        threads[found] = threads.back();
        threads.pop_back();
      }
      if (retired->thread.joinable()) retired->thread.join();
      destroy_record(retired);
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
      void* storage = alloc.allocate(sizeof(ThreadRecord), alignof(ThreadRecord));
      if (storage == nullptr) {
        if (num_threads == 0) {
          HYLE_FATAL_ASSERT(head == job && tail == job);
          head = tail = nullptr;
          return false;
        }
        return true;  // an existing busy worker will eventually take the job
      }
      ThreadRecord* record = new (storage) ThreadRecord();
      record->owner = this;
      auto r = platform::Thread::spawn(&Impl::thread_main, record, opts);
      if (r.has_value()) {
        record->thread = std::move(*r);
        threads.push_back(record);
        ++num_threads;
      } else if (num_threads == 0) {
        // No worker exists to drain the just-enqueued task. Retract it and let
        // submit's caller publish a normal Cancelled/resource failure.
        HYLE_FATAL_ASSERT(head == job && tail == job);
        head = tail = nullptr;
        destroy_record(record);
        return false;
      } else {
        destroy_record(record);
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

    std::vector<ThreadRecord*, mem::StdAllocator<ThreadRecord*>> to_join(
        threads.get_allocator());
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
    for (ThreadRecord* record : to_join) {
      if (record->thread.joinable()) record->thread.join();
      destroy_record(record);
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

BlockingPool::BlockingPool(mem::AllocatorRef alloc, unsigned max_threads,
                           time::Duration keep_alive)
    : impl_(nullptr) {
  Status status;
  init_(alloc, max_threads, keep_alive, &status);
  HYLE_FATAL_ASSERT(status.ok() &&
                    "blocking pool construction failed; use the Status overload "
                    "when resource failure is recoverable");
}

BlockingPool::BlockingPool(mem::AllocatorRef alloc, unsigned max_threads,
                           time::Duration keep_alive,
                           Status* status) noexcept {
  HYLE_FATAL_ASSERT(status != nullptr);
  init_(alloc, max_threads, keep_alive, status);
}

void BlockingPool::init_(mem::AllocatorRef alloc, unsigned max_threads,
                         time::Duration keep_alive, Status* status) noexcept {
  *status = Status{};
  if (!alloc.valid()) {
    *status = Status::InvalidArgument(
        "blocking pool requires a complete allocator");
    return;
  }
  if (max_threads == 0) {
    *status = Status::InvalidArgument(
        "blocking pool max_threads must be at least one");
    return;
  }
  if (keep_alive < time::Duration::zero()) {
    *status = Status::InvalidArgument(
        "blocking pool keep_alive must be non-negative");
    return;
  }
  void* storage = alloc.allocate(sizeof(Impl), alignof(Impl));
  if (storage == nullptr) {
    *status = Status::ResourceExhausted(
        "blocking pool implementation allocation failed");
    return;
  }
  try {
    impl_ = new (storage) Impl(alloc, max_threads, keep_alive);
  } catch (const std::bad_alloc&) {
    alloc.deallocate(storage, sizeof(Impl), alignof(Impl));
    *status = Status::ResourceExhausted(
        "blocking pool bookkeeping allocation failed");
  } catch (const std::length_error&) {
    alloc.deallocate(storage, sizeof(Impl), alignof(Impl));
    *status = Status::InvalidArgument(
        "blocking pool max_threads exceeds platform limits");
  }
}

BlockingPool::~BlockingPool() {
  if (impl_ == nullptr) return;
  impl_->shutdown();
  const mem::AllocatorRef allocator = impl_->alloc;
  impl_->~Impl();
  allocator.deallocate(impl_, sizeof(Impl), alignof(Impl));
}

void BlockingPool::shutdown() {
  if (impl_ != nullptr) impl_->shutdown();
}

unsigned BlockingPool::live_threads() const noexcept {
  return impl_ != nullptr ? impl_->live_threads() : 0;
}

std::size_t BlockingPool::retained_thread_handles() const noexcept {
  return impl_ != nullptr ? impl_->retained_thread_handles() : 0;
}

bool BlockingPool::submit(detail::BlockingJob* job) noexcept {
  return impl_ != nullptr && impl_->submit(job);
}

}  // namespace hyle::sched
