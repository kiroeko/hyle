#pragma once

// Compile-time-gated profiling hooks for zones, plots/counters, fibers, frames,
// messages, memory and thread names. This is independent from log.h's runtime
// structured-log path.
//
// Product code calls only HYLE_TRACE_*; a selected backend fills the macros in.
// The default OFF form does not evaluate arguments, call a function or include a
// backend. This makes it safe to leave hooks in fiber switching, IO completion
// and allocation paths. A Tracy build defines HYLE_TRACE_TRACY through the public
// hyle_trace CMake target and links one shared client for the whole HYLE process.

// ---- backend selection ------------------------------------------------------
// A backend define implies the master gate. HYLE_TRACE without a selected backend
// deliberately remains a no-op rather than exposing a half-wired profiler.
#if defined(HYLE_TRACE_TRACY) && !defined(HYLE_TRACE)
#  define HYLE_TRACE
#endif

#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)

// Fiber tracking needs TRACY_FIBERS; enable it consistently for every consumer.
#  ifndef TRACY_FIBERS
#    define TRACY_FIBERS
#  endif
#  include <cstdint>
#  include <cstring>
#  include <tracy/Tracy.hpp>

// Zone timing is RAII for the enclosing scope. A named zone uses a literal/static
// site name; dynamic detail belongs in HYLE_TRACE_MESSAGE or the C runtime zone-text
// seam.
#  define HYLE_TRACE_ZONE() ZoneScoped
#  define HYLE_TRACE_ZONE_NAMED(name) ZoneScopedN(name)

// A numeric time series. `name` MUST be a string literal / long-lived
// pointer (tracy stores the pointer by identity, not a copy).
#  define HYLE_TRACE_PLOT(name, value) TracyPlot(name, value)
// A counter is a plot of the caller-maintained running value.
#  define HYLE_TRACE_COUNTER(name, value) \
    TracyPlot(name, static_cast<std::int64_t>(value))

// Mark which fiber the current OS thread is running so zones/plots follow logical
// execution across worker migration. `name` identifies the fiber and MUST be a
// stable, long-lived pointer; the same fiber must always pass the same pointer.
// Leave when returning to non-fiber code.
#  define HYLE_TRACE_FIBER_ENTER(name) TracyFiberEnter(name)
#  define HYLE_TRACE_FIBER_LEAVE() TracyFiberLeave

// Frame boundary (a serving loop iteration / a graph tick).
#  define HYLE_TRACE_FRAME() FrameMark
#  define HYLE_TRACE_FRAME_NAMED(name) FrameMarkNamed(name)

// Dynamic messages are copied by Tracy before this call returns. The literal
// form avoids that copy and therefore requires static storage.
#  define HYLE_TRACE_MESSAGE(msg)                                           \
    do {                                                                    \
      const char* const hyle_trace_message_ = (msg);                        \
      TracyMessage(hyle_trace_message_, std::strlen(hyle_trace_message_));  \
    } while (false)
#  define HYLE_TRACE_MESSAGE_LITERAL(msg) TracyMessageL(msg)

// Memory attribution. A named pool string must have static/stable storage.
#  define HYLE_TRACE_ALLOC(ptr, size) TracyAlloc(ptr, size)
#  define HYLE_TRACE_FREE(ptr) TracyFree(ptr)
#  define HYLE_TRACE_ALLOC_NAMED(ptr, size, name) TracyAllocN(ptr, size, name)
#  define HYLE_TRACE_FREE_NAMED(ptr, name) TracyFreeN(ptr, name)

// Name the current OS thread in the trace backend.
#  define HYLE_TRACE_THREAD_NAME(name) ::tracy::SetThreadName(name)

#else  // ---- OFF (default) or a gate with no backend: every hook vanishes -------

// Arguments are not evaluated when tracing is off.
#  define HYLE_TRACE_ZONE() ((void)0)
#  define HYLE_TRACE_ZONE_NAMED(name) ((void)0)
#  define HYLE_TRACE_PLOT(name, value) ((void)0)
#  define HYLE_TRACE_COUNTER(name, value) ((void)0)
#  define HYLE_TRACE_FIBER_ENTER(name) ((void)0)
#  define HYLE_TRACE_FIBER_LEAVE() ((void)0)
#  define HYLE_TRACE_FRAME() ((void)0)
#  define HYLE_TRACE_FRAME_NAMED(name) ((void)0)
#  define HYLE_TRACE_MESSAGE(msg) ((void)0)
#  define HYLE_TRACE_MESSAGE_LITERAL(msg) ((void)0)
#  define HYLE_TRACE_ALLOC(ptr, size) ((void)0)
#  define HYLE_TRACE_FREE(ptr) ((void)0)
#  define HYLE_TRACE_ALLOC_NAMED(ptr, size, name) ((void)0)
#  define HYLE_TRACE_FREE_NAMED(ptr, name) ((void)0)
#  define HYLE_TRACE_THREAD_NAME(name) ((void)0)

#endif
