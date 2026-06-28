#pragma once

#include <cstddef>
#include <format>

#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_OBS_API
#elif defined(HYLE_OBS_BUILD)
#  define HYLE_OBS_API HYLE_EXPORT
#else
#  define HYLE_OBS_API HYLE_IMPORT
#endif

// Structured log production (clay sys/log.h). A level-gated message fans out to a
// registry of sinks; each sink is a {state, write-fn} vtable so a backend can be
// added across a DLL seam (console, file, ring buffer...). This is the "produce"
// half of obs/ -- aggregation/collection is an ops concern (hyle-scope), not here.
//
// Two deliberate departures from clay:
//   1. The sink fn is NON-VARIADIC: write(state, level, msg) takes a finished
//      string. clay's vtable had `printf(level, fmt, ...)` -- a variadic function
//      pointer, which is an ABI hazard across a C seam (the same trap error-02
//      removed from clay's error vtable). Formatting happens caller-side.
//   2. Formatting is std::format, not stb vsnprintf: the log_fmt<> template below
//      builds the string with compile-time-checked {} and hands only a const char*
//      to the seam (conventions §8). The default sink writes to stderr via
//      std::fprintf, never std streams (§1).

namespace hyle::obs {

enum class LogLevel : int {
  Trace = 0,
  Debug = 1,
  Info = 2,
  Warn = 3,
  Error = 4,
  Off = 5,  // set_level(Off) silences everything
};

// A log backend. `write` receives the level and an already-formatted, NUL-
// terminated message (no trailing newline required; sinks add their own). It is
// called from whatever thread logged, possibly concurrently -- a sink that
// touches shared state guards it itself.
//
// LIFETIME (hard contract): `state` must outlive every concurrent log call.
// log_message snapshots the sink list under a lock and then calls write()
// OUTSIDE the lock, so a sink removed on thread A can still be mid-call on
// thread B. Therefore: remove a sink only after all logging threads are
// quiescent, or never free its state before process exit. Removing + freeing a
// sink while another thread may be logging is a use-after-free.
struct LogSink {
  void* state;
  void (*write)(void* state, LogLevel level, const char* msg);
};

// ---- registry (compiled in log.cc) ------------------------------------------
// Register/unregister a sink. Identity is the {state, write} pair. Adding the
// same sink twice is a no-op; removing an absent one is a no-op. Sinks are a
// startup-time concern; these take an internal lock and are not hot-path.
HYLE_OBS_API void add_sink(LogSink sink) noexcept;
HYLE_OBS_API void remove_sink(LogSink sink) noexcept;

// The global threshold: messages with level < the set level are dropped before
// any sink runs. Default Info. The check is a relaxed atomic load (lock-free).
HYLE_OBS_API void set_level(LogLevel level) noexcept;
HYLE_OBS_API LogLevel level() noexcept;

// True if a message at `level` would be emitted (lets callers skip building a
// message that would be dropped).
HYLE_OBS_API bool should_log(LogLevel level) noexcept;

// Fan out an already-formatted message to every sink (if level passes the gate).
HYLE_OBS_API void log_message(LogLevel level, const char* msg) noexcept;

// ---- default console sink ---------------------------------------------------
// The stderr sink (ANSI color by level; also OutputDebugString on Windows). It
// is installed automatically on first use; call remove_sink(default_sink()) to
// silence it. Returned by value -- the same {state,write} compares equal.
HYLE_OBS_API LogSink default_sink() noexcept;

// ---- caller-side formatting (header template, no DLL hop for the format) -----
// Build the message with std::format into a stack buffer, then hand the finished
// C string to log_message. Formats into a fixed 1 KiB stack buffer (no allocation
// on the log path); a longer message is silently truncated at the buffer edge.
template <class... Args>
void log_fmt(LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
  if (!should_log(level)) return;  // skip formatting entirely when gated out
  char buf[1024];
  auto r = std::format_to_n(buf, sizeof(buf) - 1, fmt,
                            std::forward<Args>(args)...);
  *r.out = '\0';
  log_message(level, buf);
}

}  // namespace hyle::obs

// Convenience macros. Args are evaluated only if the level passes (the wrapper
// short-circuits via should_log). Use like: HYLE_LOG_INFO("x={}", x).
#define HYLE_LOG_TRACE(...) ::hyle::obs::log_fmt(::hyle::obs::LogLevel::Trace, __VA_ARGS__)
#define HYLE_LOG_DEBUG(...) ::hyle::obs::log_fmt(::hyle::obs::LogLevel::Debug, __VA_ARGS__)
#define HYLE_LOG_INFO(...)  ::hyle::obs::log_fmt(::hyle::obs::LogLevel::Info, __VA_ARGS__)
#define HYLE_LOG_WARN(...)  ::hyle::obs::log_fmt(::hyle::obs::LogLevel::Warn, __VA_ARGS__)
#define HYLE_LOG_ERROR(...) ::hyle::obs::log_fmt(::hyle::obs::LogLevel::Error, __VA_ARGS__)
