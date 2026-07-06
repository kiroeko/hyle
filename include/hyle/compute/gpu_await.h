#pragma once

#include <coroutine>
#include <cstddef>
#include <cstdint>

#include "hyle/compute/device.h"
#include "hyle/compute/stream.h"
#include "hyle/error/status.h"
#include "hyle/macros.h"
#include "hyle/sched/waker.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_COMPUTE_API
#elif defined(HYLE_COMPUTE_BUILD)
#  define HYLE_COMPUTE_API HYLE_EXPORT
#else
#  define HYLE_COMPUTE_API HYLE_IMPORT
#endif

// compute-03: GPU completion -> fiber. `co_await stream` / `co_await event` suspends
// the FIBER; the GPU finishing the work signals the event, a poller observes it and
// wakes the fiber. Mirrors io's `co_await async_read()` (net.h) -- one suspend/wake
// mechanism (sched Waker) for CPU, io, AND GPU completions (compute-design §4).
//
// MECHANISM (plan A, verified against ggml/CCCL/seastar/sched, zero sched changes):
//   - callbacks are OUT: ggml #if 0's its only cudaLaunchHostFunc (untested), CCCL
//     forbids CUDA calls in a host callback -> can't safely wake a fiber. So we POLL.
//   - a dedicated GPU POLLER thread holds the in-flight events; it cudaEventQuery()s
//     them (compute-01's Event::query) and, on ready, calls Waker::wake(). One poller
//     thread reaps MANY events (like io_uring reap), not one blocking thread per op.
//   - Waker::wake() from the poller (a foreign thread) reinjects the fiber via the
//     scheduler's global queue + notify(), unparking the owner worker (scheduler.cc
//     :449). No set_io_poll/set_io_wake needed -> sched is untouched.
//
// THE ggml ORCHESTRATION PRINCIPLE (user: "调度也参考 ggml"): GPU-to-GPU dependencies
// belong on the DEVICE side via cudaStreamWaitEvent (ggml fork/join, ggml-cuda.cu
// :4256) -- the fiber should co_await ONCE at the END of a dependency chain, not wake
// per intermediate step. co_await here is that single end-of-chain suspend; stream
// ordering + cudaStreamWaitEvent express the in-GPU deps without host round-trips.
// (The multi-stream fork/join API is a later milestone; compute-03 is the single
// co_await point it will build on.)
//
// HOST FALLBACK: work ran inline (event always ready) -> co_await completes without
// ever suspending (await_ready returns true). So the same code drives host + GPU.

namespace hyle::compute {

// Lazily-started process-wide GPU completion poller. Idempotent start; stopped at
// process teardown. Exposed mainly so tests can ensure a clean shutdown; normal code
// never touches it (the awaitables register with it implicitly).
struct GpuOp;  // forward (defined below); the poller stores GpuOp* slots
namespace detail {
// Register a GpuOp slot: the poller queries op->event and, when ready, fills
// op->status/op->done and fires op->waker. The slot lives on the awaiting fiber's
// frozen stack; the poller dereferences it until it fires, then drops it (one-shot).
// Starts the poller thread on first use. Thread-safe (called from a worker fiber;
// the poller runs on its own thread).
HYLE_COMPUTE_API void gpu_poll_register(GpuOp* op) noexcept;
// Stop the poller thread (process teardown / test cleanup). Waits until no in-flight
// events remain, then joins. Idempotent.
HYLE_COMPUTE_API void gpu_poll_shutdown() noexcept;
}  // namespace detail

// The per-await slot -- a GPU completion's landing pad, on the suspending fiber's
// stack (mirrors io::IoOperation, net.h:66). The poller fills `status`/`done` then
// fires `waker`. STACK-SLOT LIFETIME: the fiber stays suspended (its stack frozen)
// until the poller's wake, and the event WILL complete (a submitted GPU op always
// finishes or errors) -> the slot outlives the poll. A kernel that may hang forever
// would need a timed/heap slot (future work, like net_timeout); the base await
// assumes the op completes, same contract as an io base op.
struct GpuOp {
  sched::Waker waker;      // whom to wake when the event fires
  Event event;             // the completion signal (owns backend cudaEvent_t)
  Status status;           // filled before the wake (OK, or a device error)
  bool done = false;       // set true by the poller before waking (single-shot)
};

// co_await StreamComplete{&stream} -> Status : suspend the fiber until ALL work
// currently enqueued on `stream` has completed on the GPU. Records an event on the
// stream, registers it with the poller, and fiber-yields; resumes when the poller
// sees the event ready. On the host backend the event is immediately ready, so it
// completes without suspending.
//
// Usage: co_await StreamComplete{&stream};  // after enqueuing copies/kernels
struct StreamComplete {
  Stream* stream;
  GpuOp op_{};
  bool armed_ = false;

  HYLE_COMPUTE_API bool await_ready() noexcept;   // true if event already done (host)
  HYLE_COMPUTE_API bool await_suspend(std::coroutine_handle<>) noexcept;  // record+register+yield
  Status await_resume() noexcept { return op_.status; }
};

}  // namespace hyle::compute
