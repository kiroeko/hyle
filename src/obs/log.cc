#include "hyle/obs/log.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

namespace hyle::obs {
namespace {

constexpr std::size_t kMaxSinks = 64;

// Global registry. The level is a lock-free relaxed atomic so the hot gate
// (should_log) never locks. The sink array is guarded by a mutex used ONLY for
// add/remove and for taking a snapshot in log_message -- the actual sink I/O
// runs outside the lock (never hold a lock across a write() that may block).
std::atomic<LogLevel> g_level{LogLevel::Info};
std::mutex g_mutex;
LogSink g_sinks[kMaxSinks];
std::size_t g_count = 0;
bool g_initialized = false;  // default sink installed once (not re-added on empty)

void default_write(void*, LogLevel level, const char* msg) noexcept {
  // ANSI color by level; clay appended a newline when missing -- same here.
  const char* color = "";
  switch (level) {
    case LogLevel::Trace: color = "\x1b[37m"; break;  // grey
    case LogLevel::Debug: color = "\x1b[36m"; break;  // cyan
    case LogLevel::Info:  color = "\x1b[32m"; break;  // green
    case LogLevel::Warn:  color = "\x1b[33m"; break;  // yellow
    case LogLevel::Error: color = "\x1b[31m"; break;  // red
    default: break;
  }
  std::fprintf(stderr, "%s%s\x1b[0m\n", color, msg);
  std::fflush(stderr);
#if defined(_WIN32)
  OutputDebugStringA(msg);
  OutputDebugStringA("\n");
#endif
}

// Install the default sink the first time the registry is touched. Caller holds
// g_mutex. A user who calls remove_sink(default_sink()) keeps it gone (the flag
// stays set), matching clay's "auto-added, explicitly removable" contract.
void ensure_init_locked() noexcept {
  if (!g_initialized) {
    g_sinks[0] = LogSink{nullptr, &default_write};
    g_count = 1;
    g_initialized = true;
  }
}

bool same_sink(const LogSink& a, const LogSink& b) noexcept {
  return a.state == b.state && a.write == b.write;
}

}  // namespace

LogSink default_sink() noexcept { return LogSink{nullptr, &default_write}; }

void set_level(LogLevel level) noexcept {
  g_level.store(level, std::memory_order_relaxed);
}
LogLevel level() noexcept { return g_level.load(std::memory_order_relaxed); }

bool should_log(LogLevel lvl) noexcept {
  return static_cast<int>(lvl) >= static_cast<int>(level());
}

void add_sink(LogSink sink) noexcept {
  if (sink.write == nullptr) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_init_locked();
  for (std::size_t i = 0; i < g_count; ++i)
    if (same_sink(g_sinks[i], sink)) return;  // identity dedupe
  if (g_count >= kMaxSinks) return;           // silently full (clay parity)
  g_sinks[g_count++] = sink;
}

void remove_sink(LogSink sink) noexcept {
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_init_locked();  // so removing the default before any log() still works
  for (std::size_t i = 0; i < g_count; ++i) {
    if (same_sink(g_sinks[i], sink)) {
      g_sinks[i] = g_sinks[--g_count];  // swap-remove (order irrelevant)
      return;
    }
  }
}

void log_message(LogLevel lvl, const char* msg) noexcept {
  if (!should_log(lvl)) return;
  // Snapshot under the lock, then emit outside it: a slow/blocking sink must not
  // stall other threads' add/remove, and we must not call write() while holding
  // a lock the sink might re-enter.
  LogSink snap[kMaxSinks];
  std::size_t n;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_init_locked();
    n = g_count;
    std::memcpy(snap, g_sinks, n * sizeof(LogSink));
  }
  for (std::size_t i = 0; i < n; ++i) snap[i].write(snap[i].state, lvl, msg);
}

}  // namespace hyle::obs
