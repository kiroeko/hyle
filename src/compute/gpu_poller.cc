#include "hyle/compute/gpu_await.h"

#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hyle/compute/stream.h"
#include "hyle/macros.h"
#include "hyle/obs/trace.h"
#include "hyle/platform/thread.h"
#include "hyle/platform/wait.h"
#include "hyle/sched/waker.h"

// compute-03: the GPU completion poller (plan A). A single dedicated thread holds the
// in-flight GpuOp slots and cudaEventQuery()s them; on ready it wakes the awaiting
// fiber via Waker::wake(). One thread reaps MANY events (io_uring-reap-shaped), not a
// blocking thread per op. NOT a second scheduler -- it only observes completions and
// re-injects fibers through sched's existing Waker spine (same role as io's reap).
//
// Poll cadence: when events are in flight, use Parker::park_timeout between sweeps
// (cudaEventQuery is a cheap non-blocking check; a tight spin would burn a core). A
// registration can interrupt that timeout, while a plain sleep could not. When none
// are in flight, park until register() adds one -- zero cost idle. Thread creation,
// parking and wakeup all use platform's portable failure/wait contracts rather than
// inventing a second OS abstraction inside compute.

namespace hyle::compute {
namespace detail {
namespace {

class GpuPoller {
 public:
  static GpuPoller& instance() noexcept {
    static GpuPoller p;  // Meyers singleton (thread-safe init, charter)
    return p;
  }

  Status register_op(GpuOp* op) noexcept {
    {
      std::lock_guard<std::mutex> lk(mu_);
      // shutdown() drains already-registered work but deliberately rejects new
      // work. Without this state check a registration could land after the
      // poller exited but before shutdown reset its thread handle.
      if (stop_) {
        return Status::FailedPrecondition(
            "GPU completion poller is shutting down");
      }

      try {
        inflight_.push_back(op);
      } catch (const std::bad_alloc&) {
        return Status::ResourceExhausted(
            "GPU completion registration allocation failed");
      } catch (const std::length_error&) {
        return Status::ResourceExhausted(
            "GPU completion registration capacity exhausted");
      }

      if (!running_) {
        platform::ThreadOpts opts;
        opts.name = "hyle.gpu-poller";
        auto spawned = platform::Thread::spawn([this] { run(); }, opts);
        if (!spawned) {
          inflight_.pop_back();
          return std::move(spawned).error().WithContext(
              "start GPU completion poller");
        }
        thread_ = std::move(*spawned);
        running_ = true;
      }
    }
    wake_.unpark();  // interrupt either the idle park or the polling timeout
    return Status();
  }

  void shutdown() noexcept {
    platform::Thread t;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!running_) return;
      stop_ = true;
      t = std::move(thread_);
    }
    wake_.unpark();
    if (t.joinable()) t.join();
    std::lock_guard<std::mutex> lk(mu_);
    running_ = false;
    stop_ = false;
  }

  ~GpuPoller() { shutdown(); }

 private:
  void run() noexcept {
    HYLE_TRACE_THREAD_NAME("hyle.gpu-poller");
    for (;;) {
      // Query + remove under the poller lock. Completed ops are threaded through
      // their frozen stack slots, so the reap path never allocates and cannot
      // terminate this noexcept thread during memory pressure. Waker::wake() is
      // called only after releasing our lock because it enters the scheduler.
      GpuOp* ready = nullptr;
      bool has_work = false;
      bool stopping = false;
      {
        HYLE_TRACE_ZONE_NAMED("hyle.compute.gpu_reap");
        std::unique_lock<std::mutex> lk(mu_);
        for (std::size_t i = 0; i < inflight_.size();) {
          GpuOp* op = inflight_[i];
          if (op->event.query() == EventStatus::kReady) {
            // Fill the slot BEFORE removing/waking (the fiber reads status on resume).
            op->status = Status();  // completion observed OK (a device fault would
                                    // surface via stream sync; query() maps non-NotReady
                                    // -> ready, matching compute-01's contract)
            op->done = true;
            op->ready_next = ready;
            ready = op;
            // swap-remove from the in-flight list (order doesn't matter).
            inflight_[i] = inflight_.back();
            inflight_.pop_back();
          } else {
            ++i;
          }
        }
        has_work = !inflight_.empty();
        stopping = stop_;
      }
#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
      static thread_local unsigned long long trace_completions = 0;
      for (GpuOp* op = ready; op != nullptr; op = op->ready_next) {
        ++trace_completions;
      }
      HYLE_TRACE_COUNTER("hyle.compute.gpu_completions", trace_completions);
#endif
      // Wake outside the lock.
      while (ready != nullptr) {
        GpuOp* op = ready;
        ready = ready->ready_next;
        op->ready_next = nullptr;
        op->waker.wake();
      }

      if (stopping && !has_work) return;

      // 50us is well below typical GPU op latency and avoids a hot spin. Unlike
      // sleep_for, park_timeout is interrupted immediately by new registration
      // or shutdown. A token arriving before park is remembered, so there is no
      // condition-variable lost-wakeup window.
      if (has_work) {
        (void)wake_.park_timeout(50'000);
      } else {
        wake_.park();
      }
    }
  }

  std::mutex mu_;
  std::vector<GpuOp*> inflight_;
  platform::Parker wake_;
  platform::Thread thread_;
  bool running_ = false;
  bool stop_ = false;
};

}  // namespace

Status gpu_poll_register(GpuOp* op) noexcept {
  return GpuPoller::instance().register_op(op);
}

void gpu_poll_shutdown() noexcept { GpuPoller::instance().shutdown(); }

}  // namespace detail

// ---- StreamComplete awaitable -----------------------------------------------

bool StreamComplete::await_ready() noexcept {
  // Create the completion event on the stream's device and record it at the current
  // point in the stream. If the event is ALREADY ready (host fallback runs inline, or
  // a GPU stream that happens to be drained), skip suspending entirely.
  auto ev = Event::create(stream->device());
  if (!ev.has_value()) {
    // Can't make an event -> fail fast on resume without suspending.
    op_.status = std::move(ev).error();
    op_.done = true;
    return true;  // don't suspend; await_resume returns the error
  }
  op_.event = std::move(*ev);
  stream->record(op_.event);
  if (op_.event.query() == EventStatus::kReady) {
    op_.status = Status();
    op_.done = true;
    return true;  // already complete -> no suspend
  }
  return false;  // not ready -> suspend in await_suspend
}

bool StreamComplete::await_suspend(std::coroutine_handle<>) noexcept {
  // Capture THIS fiber's waker, register the event with the poller, and fiber-yield.
  // The poller fills op_.status/done and wakes us when the event completes. Returning
  // false resumes the coroutine after the fiber comes back (same pattern as io RecvOp).
  op_.waker = sched::current_waker();
  armed_ = true;
  Status registration = detail::gpu_poll_register(&op_);
  if (!registration.ok()) {
    op_.status = std::move(registration);
    op_.done = true;
    armed_ = false;
    return false;
  }
  sched::suspend();  // fiber yields; poller's wake re-injects it
  return false;      // fiber is back -> continue the coroutine
}

}  // namespace hyle::compute
