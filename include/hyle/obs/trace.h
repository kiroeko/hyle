#pragma once

// Compile-time-gated profiling hooks for zones, plots/counters, fibers, frames,
// messages, memory and thread names. This is independent from log.h's runtime
// structured-log path.
//
// Product code calls only HYLE_TRACE_*; a selected backend fills the macros in.
// The default OFF form does not evaluate arguments, call a function or include a
// backend. This makes it safe to leave hooks in fiber switching, IO completion
// and allocation paths. A Tracy build defines HYLE_TRACE_TRACY through the public
// hyle_trace CMake target. Shared HYLE provides one client DLL for the process.
// A static trace-on SDK is supported only when every HYLE archive is linked into
// one final executable/link image. If multiple DLLs/plugins each static-link it,
// each image gets a separate Tracy singleton; use shared hyle_obs for that shape.

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
#  include <cstddef>
#  include <cstdint>
#  include <cstring>
#  include <type_traits>
#  include <tracy/Tracy.hpp>

namespace hyle::obs::detail {

// Tracy stores copied zone text and messages with a 16-bit byte count. Its
// public API asserts the limit only in debug builds and truncates the count in
// release builds, so every adapter must bound the scan before entering Tracy.
inline constexpr std::size_t kTraceDynamicTextMax = 65534;

inline std::size_t bounded_trace_string_length(
    const char* text,
    std::size_t limit = kTraceDynamicTextMax) noexcept {
  std::size_t size = 0;
  while (size < limit && text[size] != '\0')
    ++size;
  return size;
}

// Tracy 0.14 has int64_t/float/double PlotData overloads, so an ordinary int
// argument is ambiguous on MSVC.  Normalize ordinary integral/enum counters to
// its signed transport, use double for a full-width unsigned value, and preserve
// float/double semantics.  This keeps
// HYLE_TRACE_PLOT("queue.depth", depth) a normal consumer shape instead of
// forcing every call site to know a profiler-backend overload set.
template <class T>
inline auto trace_plot_value(T value) noexcept {
  using U = std::remove_cv_t<std::remove_reference_t<T>>;
  static_assert(std::is_arithmetic_v<U> || std::is_enum_v<U>,
                "HYLE_TRACE_PLOT value must be numeric or an enum");
  if constexpr (std::is_floating_point_v<U>) {
    if constexpr (sizeof(U) <= sizeof(float)) {
      return static_cast<float>(value);
    } else {
      return static_cast<double>(value);
    }
  } else if constexpr (std::is_unsigned_v<U> && sizeof(U) >= sizeof(std::uint64_t)) {
    return static_cast<double>(value);
  } else {
    return static_cast<std::int64_t>(value);
  }
}

}  // namespace hyle::obs::detail

// Zone timing is RAII for the enclosing scope. A named zone uses a literal/static
// site name; dynamic detail belongs in HYLE_TRACE_MESSAGE or the C runtime zone-text
// seam.
#  define HYLE_TRACE_ZONE() ZoneScoped
#  define HYLE_TRACE_ZONE_NAMED(name) ZoneScopedN(name)

// A numeric time series. `name` MUST be a string literal / long-lived
// pointer (tracy stores the pointer by identity, not a copy).
#  define HYLE_TRACE_PLOT(name, value) \
    TracyPlot(name, ::hyle::obs::detail::trace_plot_value(value))
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
      TracyMessage(                                                        \
          hyle_trace_message_,                                             \
          ::hyle::obs::detail::bounded_trace_string_length(                \
              hyle_trace_message_));                                       \
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
