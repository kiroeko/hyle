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
// THE COMPLETION MODEL (compute-design §2, decided before this header, verified on
// RTX 4070): a Stream is an ordered async queue; you enqueue work, then record()
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
// NO PIMPL / NO HEAP (perf): the whole state is {vtable, native handle, ordinal} =
// 3 words, held INLINE. An Event is a long-lived object (one per await episode /
// reused per fiber), not a hot-path temporary, but inlining still removes a heap
// alloc on create AND an extra indirection on the query() hot path (compute-03's
// worker polls query() every loop). query() is then ONE vtable indirect (the
// irreducible cost of backend polymorphism), no impl_ deref. The native handle is
// an opaque void* (a cudaEvent_t IS a pointer; the host backend stores a small
// tag) -- the public header stays free of backend types via the forward decl above.
//
// LIFETIME CONTRACT: an Event must outlive the stream work it was recorded on
// being polled -- i.e. don't destroy it while a worker may still query() it. In
// compute-01 the test owns it on the stack across the poll loop; compute-03's
// awaitable will park it on the fiber stack (like io's IoOperation slot).
class Event {
 public:
  Event() noexcept = default;
  HYLE_DISABLE_COPY(Event);
  HYLE_COMPUTE_API Event(Event&& o) noexcept;
  HYLE_COMPUTE_API Event& operator=(Event&& o) noexcept;
  HYLE_COMPUTE_API ~Event();

  // Create an event on `device`. Fails (Unavailable/ResourceExhausted) if the
  // backend can't allocate one (cudaEventCreate error) -- an external-boundary
  // failure, so it returns Result, not a fatal assert.
  HYLE_COMPUTE_API static Result<Event> create(const Device& device) noexcept;

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
  friend class Stream;  // Stream::record reads event.native_
  const ComputeBackend* vt_ = nullptr;  // backend that owns native_ (null = invalid)
  void* native_ = nullptr;              // cudaEvent_t / host tag (opaque)
  int local_ = 0;                       // backend-local device ordinal
};

// An ordered asynchronous execution queue on a Device (compute-design "模块形状"
// stream.h): work enqueued on it runs in order, asynchronously to the host on a
// real GPU. It is the "schedulable resource that produces completions" of
// compute-design §4. Move-only (owns a cudaStream_t / host queue). compute-01
// gives it lifecycle + record + a memcpy op sufficient to make an Event observably
// transition NotReady->Ready; kernel launch is compute-04.
//
// CONCURRENCY CONTRACT (like a CUDA stream): a Stream belongs to ONE execution flow
// (one fiber / one thread) at a time. record()/launch()/launch_batch()/memcpy on the
// SAME Stream must NOT be called concurrently from multiple fibers or threads -- the
// per-stream submission state (command pool + in-flight tracking on the Vulkan backend)
// is NOT internally locked; concurrent use is a data race + a Vulkan external-sync
// violation (VkCommandPool). For parallel submission, use ONE Stream PER fiber. Multiple
// distinct Streams may be used concurrently (their submits are serialized internally at
// the shared queue). This matches io's "one connection per fiber" model.
class Stream {
 public:
  Stream() noexcept = default;
  HYLE_DISABLE_COPY(Stream);
  HYLE_COMPUTE_API Stream(Stream&& o) noexcept;
  HYLE_COMPUTE_API Stream& operator=(Stream&& o) noexcept;
  HYLE_COMPUTE_API ~Stream();

  // Create a stream on `device`. Fails at the external boundary (returns Result).
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

  // ---- device memory + async copy (minimal, for compute-01 completion proof) ----
  // compute-01 needs *some* real async device work to make an Event visibly go
  // NotReady->Ready on a GPU. A device malloc + an async H2D/D2H copy is the
  // smallest such work that needs NO kernel (kernel launch is compute-04). This is
  // NOT the compute-02 buffer abstraction (owned, typed, buffer_type instances) --
  // it is the raw minimum to exercise the stream/event spine. compute-02 supersedes it.

  // Allocate `bytes` of DEVICE memory on this stream's device (cudaMalloc; host:
  // plain malloc). Returns the device pointer or an external-boundary failure.
  HYLE_COMPUTE_API Result<void*> device_alloc(std::size_t bytes) noexcept;
  // Free a pointer from device_alloc (cudaFree / free). Null is a no-op.
  HYLE_COMPUTE_API void device_free(void* ptr) noexcept;

  // Enqueue an ASYNC copy of `bytes` from `src` to `dst` on this stream
  // (cudaMemcpyAsync; host: memcpy inline). Direction is inferred by the backend
  // from the pointers it minted. `src`/`dst` must outlive the copy's completion
  // (caller contract -- same as io buffers). Returns a submit-time failure; a
  // completed-with-error surfaces via the recorded Event / sync().
  HYLE_COMPUTE_API Status memcpy_async(void* dst, const void* src,
                                       std::size_t bytes) noexcept;

 private:
  // Inline state (no pimpl / no heap, same perf rationale as Event). A Stream is
  // long-lived (one per connection/fiber) so this isn't a hot-path alloc, but
  // inlining keeps record()/memcpy_async() to a single vtable indirect.
  // kernel.h's launch() needs native_ to submit to this stream (defined in the same
  // .cc dispatch layer); friend it rather than exposing the backend handle publicly.
  // This is the C-ABI core launch taking a KernelArgs (the variadic typed shell in
  // kernel.h builds the KernelArgs then calls this, so only this one needs friending).
  friend HYLE_COMPUTE_API Status launch(Stream&, const class Kernel&,
                                        const struct LaunchConfig&,
                                        const struct KernelArgs&) noexcept;
  friend HYLE_COMPUTE_API Status launch_batch(Stream&, const struct BatchItem*,
                                              unsigned) noexcept;
  Device device_{};
  const ComputeBackend* vt_ = nullptr;  // backend that owns native_ (null = invalid)
  void* native_ = nullptr;              // cudaStream_t / host queue (opaque)
  int local_ = 0;                       // backend-local device ordinal
};

}  // namespace hyle::compute
