#include "hyle/compute/gpu_await.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "hyle/compute/stream.h"
#include "hyle/macros.h"
#include "hyle/obs/trace.h"
#include "hyle/sched/waker.h"

// compute-03: the GPU completion poller (plan A). A single dedicated thread holds the
// in-flight GpuOp slots and cudaEventQuery()s them; on ready it wakes the awaiting
// fiber via Waker::wake(). One thread reaps MANY events (io_uring-reap-shaped), not a
// blocking thread per op. NOT a second scheduler -- it only observes completions and
// re-injects fibers through sched's existing Waker spine (same role as io's reap).
//
// Poll cadence: when events are in flight, poll with a short sleep between sweeps
// (cudaEventQuery is a cheap non-blocking check; a tight spin would burn a core). When
// none are in flight, block on a condvar until register() adds one -- zero cost idle
// (the io park-boundary idea: don't busy-poll when there's nothing to reap).

namespace hyle::compute {
namespace detail {
namespace {

class GpuPoller {
 public:
  static GpuPoller& instance() noexcept {
    static GpuPoller p;  // Meyers singleton (thread-safe init, charter)
    return p;
  }

  void register_op(GpuOp* op) noexcept {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!running_) {
        running_ = true;
        thread_ = std::thread([this] { run(); });
      }
      inflight_.push_back(op);
    }
    cv_.notify_one();  // wake the poller if it was idle-waiting
  }

  void shutdown() noexcept {
    std::thread t;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!running_) return;
      stop_ = true;
      t = std::move(thread_);
    }
    cv_.notify_one();
    if (t.joinable()) t.join();
    std::lock_guard<std::mutex> lk(mu_);
    running_ = false;
    stop_ = false;
  }

  ~GpuPoller() { shutdown(); }

 private:
  void run() noexcept {
    HYLE_TRACE_THREAD_NAME("hyle.gpu-poller");
    std::vector<GpuOp*> ready;  // reused scratch: ops observed done this sweep
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        // Idle-wait while nothing is in flight (and not stopping). Zero-cost idle.
        cv_.wait(lk, [this] { return stop_ || !inflight_.empty(); });
        if (stop_ && inflight_.empty()) return;
      }
      // Sweep the in-flight list OUTSIDE the lock as much as possible: take a snapshot
      // under lock, query each (query() is a backend vtable call -> cheap, no CUDA API
      // reentrancy risk since we're on our own thread), collect the ready ones, then
      // remove + wake them under lock. Waker::wake() is called WITHOUT our lock held
      // (it reaches into the scheduler; keep our critical section tight + deadlock-free).
      ready.clear();
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
            ready.push_back(op);
            // swap-remove from the in-flight list (order doesn't matter).
            inflight_[i] = inflight_.back();
            inflight_.pop_back();
          } else {
            ++i;
          }
        }
      }
#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
      static thread_local unsigned long long trace_completions = 0;
      trace_completions += ready.size();
      HYLE_TRACE_COUNTER("hyle.compute.gpu_completions", trace_completions);
#endif
      // Wake outside the lock.
      for (GpuOp* op : ready) op->waker.wake();

      // If work remains, sleep briefly before the next sweep (avoid a hot spin). The
      // interval trades latency vs CPU; 50us is well below typical GPU op latency and
      // negligible CPU. (A future tuning knob / the plan-B worker-veto model removes
      // the poller thread entirely.)
      bool has_work;
      {
        std::lock_guard<std::mutex> lk(mu_);
        has_work = !inflight_.empty();
        if (stop_ && inflight_.empty()) return;
      }
      if (has_work) std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<GpuOp*> inflight_;
  std::thread thread_;
  bool running_ = false;
  bool stop_ = false;
};

}  // namespace

void gpu_poll_register(GpuOp* op) noexcept { GpuPoller::instance().register_op(op); }

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
  detail::gpu_poll_register(&op_);
  sched::suspend();  // fiber yields; poller's wake re-injects it
  return false;      // fiber is back -> continue the coroutine
}

}  // namespace hyle::compute
