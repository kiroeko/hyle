#pragma once

#include <cstdint>

#include "hyle/compute/device.h"
#include "hyle/compute/types.h"
#include "hyle/error/result.h"
#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_COMPUTE_API
#elif defined(HYLE_COMPUTE_BUILD)
#  define HYLE_COMPUTE_API HYLE_EXPORT
#else
#  define HYLE_COMPUTE_API HYLE_IMPORT
#endif

// compute-01: Stream (async execution queue) + Event (completion signal). This is
// the compute pillar's completion PRIMITIVE -- the object that makes GPU work
// observable as "done yet?" without blocking (compute-design §2, §4).
//
// THE COMPLETION MODEL (compute-design §2, verified on the GPU backends): a Stream is
// an ordered async queue; you enqueue work, then record()
// an Event on it. The Event fires when all prior stream work completes. You learn
// that by POLLING event.query() -> Ready / NotReady -- the exact shape of CUDA's
// cudaEventQuery (cudaSuccess / cudaErrorNotReady), mirrored by CCCL's
// event_ref::is_done(). We DO NOT use cudaLaunchHostFunc/StreamAddCallback: both
// forbid calling any CUDA API from the callback (deadlock) and run the callback on
// a driver-internal thread. ggml-cuda's only cudaLaunchHostFunc is #if 0'd
// "untested" -- even the mature library doesn't use callbacks for host completion.
//
// WHY POLLING IS THE RIGHT SHAPE (the io_uring reap analogue): a sched worker will
// (compute-03) call query() at its existing poll boundary -- the same seam it
// reaps io_uring/IOCP completions at -- and wake the awaiting fiber when Ready. The
// wake then runs on the worker, not a CUDA thread; GPU becomes a THIRD completion
// source alongside CPU and io, one suspend/wake mechanism (sched-design §3). But
// that wiring is compute-03; compute-01 ONLY builds query() and proves it observes
// real GPU completion. There is no co_await here yet.
//
// TWO BACKENDS behind the same abstraction (device.cc / stream.cc, #if
// HYLE_COMPUTE_CUDA): the CUDA backend maps onto cudaStream_t/cudaEvent_t; the host
// fallback runs work inline (so an Event is always immediately Ready) -- it exists
// so the completion contract compiles and tests on ANY platform (no GPU / CUDA not
// built, e.g. mac or CI), and is the deterministic backend compute-03 uses to test
// fiber wakeup logic without a real GPU. Same role as io's epoll fallback.

namespace hyle::compute {

class Stream;

// Internal backend vtable (src/compute/backend.h). Forward-declared here so Stream/
// Event can hold a `const ComputeBackend*` WITHOUT the public header seeing any
// backend-native type (no cudaStream_t leak). The definition is internal-only.
struct ComputeBackend;

// EventStatus lives in types.h (shared with the internal backend vtable without
// dragging in std::expected -- see types.h for why the CUDA TU needs that split).

// A completion signal recorded on a Stream. Owns a backend event object
// (cudaEvent_t on GPU; a trivial flag on host). Move-only (single owner of the
// backend handle). Create with Event::create(device), record() it on a stream
// after enqueuing work, then query()/sync() for completion.
//
// NO PUBLIC PIMPL: the cross-backend wrapper is held inline as {vtable, native handle,
// ordinal}. query() is one vtable dispatch from public code. A backend may still need a
// small native control record (Vulkan stores fence ownership + record state so reset and
// destruction remain legal); that cost belongs to Event creation, not the polling loop.
// The opaque void* keeps backend types out of this public header.
//
// LIFETIME CONTRACT: an Event must outlive any worker that may poll it. Backends make
// destruction safe with respect to already-submitted device work, but destroying an Event
// while another thread still calls query() is a host data race. compute-03's awaitable parks
// it on the suspended fiber stack until the poller reports completion.
class Event {
 public:
  Event() noexcept = default;
  HYLE_DISABLE_COPY(Event);
  HYLE_COMPUTE_API Event(Event&& o) noexcept;
  HYLE_COMPUTE_API Event& operator=(Event&& o) noexcept;
  HYLE_COMPUTE_API ~Event();

  // Create an event on `device`. An invalid Device returns InvalidArgument.
  // Fails (Unavailable/ResourceExhausted) if the
  // backend can't allocate one (cudaEventCreate error) -- an external-boundary
  // failure, so it returns Result, not a fatal assert.
  HYLE_COMPUTE_API static Result<Event> create(const Device& device) noexcept;

  // Create a TIMING event (compute-06b): records a device timestamp when record()'d, so
  // GPU-elapsed time between two of them is readable via elapsed_ns(). This is the
  // cudaEventElapsedTime model -- a first-class profiling capability (adaptive kernel
  // selection, perf monitoring, compute benchmarks). SEPARATE from create() so the
  // completion path keeps the cheaper no-timing event. Returns Unimplemented on a backend
  // without timing support, and InvalidArgument for an invalid Device. Record it on a
  // stream with Stream::record() (auto-dispatched),
  // exactly like a plain event; read elapsed_ns(start, stop) after both complete.
  HYLE_COMPUTE_API static Result<Event> create_timing(const Device& device) noexcept;

  bool valid() const noexcept { return vt_ != nullptr; }

  // Poll: has the work recorded before this event finished? kReady means a sync()
  // would return immediately. Non-blocking (cudaEventQuery). CONTRACT: the event
  // must have been record()'d on a stream first -- querying an un-recorded event is
  // a caller bug (HYLE_FATAL_ASSERT), not a runtime error. On the host backend this
  // is always kReady (work ran inline at enqueue time).
  HYLE_COMPUTE_API EventStatus query() const noexcept;

  // Block the CALLING THREAD until this event's work completes (cudaEventSynchronize).
  // This is the blocking counterpart to query(); compute-01 tests use it to force
  // completion, and it is the honest fallback where no fiber scheduler drives the
  // poll. A fiber MUST NOT call this on a hot path (it blocks the whole worker) --
  // that is what compute-03's co_await replaces. No-op on the host backend.
  HYLE_COMPUTE_API void sync() const noexcept;

 private:
  friend class Stream;  // Stream::record reads event.native_ / timing_
  // GPU-elapsed nanoseconds between two completed timing events (both created via
  // create_timing, start recorded before stop on the same stream). Free function (not a
  // member) so it reads two events symmetrically. Unimplemented on a backend without timing.
  friend HYLE_COMPUTE_API Result<std::uint64_t> elapsed_ns(const Event& start,
                                                           const Event& stop) noexcept;
  const ComputeBackend* vt_ = nullptr;  // backend that owns native_ (null = invalid)
  void* native_ = nullptr;              // cudaEvent_t / host tag (opaque)
  int local_ = 0;                       // backend-local device ordinal
  bool timing_ = false;                 // created via create_timing -> Stream::record writes a timestamp
};

// An ordered asynchronous execution queue on a Device (compute-design "模块形状"
// stream.h): work enqueued on it runs in order, asynchronously to the host on a
// real GPU. It is the "schedulable resource that produces completions" of
// compute-design §4. Move-only (owns a cudaStream_t / host queue). compute-01
// gives it lifecycle + record + sync; the async work that makes an Event observably
// transition NotReady->Ready is a kernel launch (compute-04) or a Buffer copy
// (compute-02).
//
// CONCURRENCY CONTRACT (like a CUDA stream): a Stream belongs to ONE execution flow
// (one fiber / one thread) at a time. record()/launch()/launch_batch() on the
// SAME Stream must NOT be called concurrently from multiple fibers or threads -- the
// per-stream submission state (command pool + in-flight tracking on the Vulkan backend)
// is NOT internally locked; concurrent use is a data race + a Vulkan external-sync
// violation (VkCommandPool). For parallel submission, use ONE Stream PER fiber. Multiple
// distinct Streams may be used concurrently. The current Vulkan backend maps them to one
// shared VkQueue, so submission is serialized and record()/sync() may conservatively wait
// for earlier work from another Stream; this preserves completion correctness but is not a
// promise of CUDA-like inter-stream overlap. This matches io's "one connection per fiber"
// ownership model, not its parallelism model.
class Stream {
 public:
  Stream() noexcept = default;
  HYLE_DISABLE_COPY(Stream);
  HYLE_COMPUTE_API Stream(Stream&& o) noexcept;
  HYLE_COMPUTE_API Stream& operator=(Stream&& o) noexcept;
  HYLE_COMPUTE_API ~Stream();

  // Create a stream on `device`. An invalid Device returns InvalidArgument; backend
  // failures are also returned through Result.
  HYLE_COMPUTE_API static Result<Stream> create(const Device& device) noexcept;

  bool valid() const noexcept { return vt_ != nullptr; }
  const Device& device() const noexcept { return device_; }

  // Record `event` at the current point in this stream (cudaEventRecord). After
  // this, event.query() reports whether all work enqueued on the stream BEFORE the
  // record has completed. CONTRACT: `event` and `*this` are valid and on the same
  // device (caller bug otherwise -> fatal assert).
  HYLE_COMPUTE_API void record(Event& event) noexcept;

  // Block the calling thread until ALL work on this stream completes
  // (cudaStreamSynchronize). No-op on host (inline execution). External boundary
  // (a device error surfaces as a returned Status); the common success path is void.
  HYLE_COMPUTE_API Status sync() noexcept;

 private:
  // Inline state (no pimpl / no heap, same perf rationale as Event). A Stream is
  // long-lived (one per connection/fiber) so this isn't a hot-path alloc, but
  // inlining keeps record() to a single vtable indirect.
  // kernel.h's launch() needs native_ to submit to this stream (defined in the same
  // .cc dispatch layer); friend it rather than exposing the backend handle publicly.
  // This is the C-ABI core launch taking a KernelArgs (the variadic typed shell in
  // kernel.h builds the KernelArgs then calls this, so only this one needs friending).
  friend HYLE_COMPUTE_API Status launch(Stream&, const class Kernel&,
                                        const struct LaunchConfig&,
                                        const struct KernelArgs&) noexcept;
  friend HYLE_COMPUTE_API Status launch_batch(Stream&, const struct BatchItem*,
                                              unsigned) noexcept;
  friend class RecordedLaunch;  // record/replay read vt_ + native_ (same as launch)
  Device device_{};
  const ComputeBackend* vt_ = nullptr;  // backend that owns native_ (null = invalid)
  void* native_ = nullptr;              // cudaStream_t / host queue (opaque)
  int local_ = 0;                       // backend-local device ordinal
};

// GPU-elapsed time in NANOSECONDS between two timing events (compute-06b). Both must be
// created via Event::create_timing, recorded on the SAME stream (start before stop), and
// COMPLETE (co_await StreamComplete / stream.sync()) before calling this. Mirrors CUDA's
// cudaEventElapsedTime (there in ms; HYLE reports ns for a common integer unit across the
// CUDA/Vulkan/host backends). Returns Unimplemented on a backend without timing support,
// InvalidArgument unless both are timing events on the same backend device. On the host
// backend timing is inline so the result is 0.
HYLE_COMPUTE_API Result<std::uint64_t> elapsed_ns(const Event& start,
                                                  const Event& stop) noexcept;

}  // namespace hyle::compute
