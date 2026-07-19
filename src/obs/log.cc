#include "hyle/obs/log.h"

#include <atomic>
#include <cstdio>
#include <mutex>

#include "hyle/macros.h"
#include "hyle/platform.h"

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  include <io.h>
#  include <windows.h>
#else
#  include <unistd.h>
#endif

namespace hyle::obs {
namespace {

constexpr std::size_t kMaxSinks = 64;

struct SinkSlot {
  LogSink sink{};
  std::atomic<std::size_t> active{0};
  std::atomic<bool> waiting{false};
  bool registered = false;  // guarded by g_mutex
  bool retiring = false;    // prevents reuse until remove's callback drain ends
};

struct ActiveFrame {
  SinkSlot* slot;
  ActiveFrame* previous;
};

std::atomic<LogLevel> g_level{LogLevel::Info};
std::mutex g_mutex;
std::mutex g_default_mutex;
SinkSlot g_slots[kMaxSinks];
SinkSlot* g_registered[kMaxSinks];
std::size_t g_count = 0;
bool g_initialized = false;
thread_local ActiveFrame* g_active_frame = nullptr;

bool valid_record_level(LogLevel value) noexcept {
  const int raw = static_cast<int>(value);
  return raw >= static_cast<int>(LogLevel::Trace) &&
         raw <= static_cast<int>(LogLevel::Error);
}

bool valid_threshold(LogLevel value) noexcept {
  const int raw = static_cast<int>(value);
  return raw >= static_cast<int>(LogLevel::Trace) &&
         raw <= static_cast<int>(LogLevel::Off);
}

const char* level_name(LogLevel level) noexcept {
  switch (level) {
    case LogLevel::Trace: return "trace";
    case LogLevel::Debug: return "debug";
    case LogLevel::Info: return "info";
    case LogLevel::Warn: return "warn";
    case LogLevel::Error: return "error";
    case LogLevel::Off: break;
  }
  return "invalid";
}

const char* level_color(LogLevel level) noexcept {
  switch (level) {
    case LogLevel::Trace: return "\x1b[37m";
    case LogLevel::Debug: return "\x1b[36m";
    case LogLevel::Info: return "\x1b[32m";
    case LogLevel::Warn: return "\x1b[33m";
    case LogLevel::Error: return "\x1b[31m";
    case LogLevel::Off: break;
  }
  return "";
}

void default_write(void*, const LogRecord& record) noexcept {
  std::lock_guard<std::mutex> lock(g_default_mutex);
#if defined(HYLE_OS_WINDOWS)
  const bool color_enabled = _isatty(_fileno(stderr)) != 0;
#else
  const bool color_enabled = isatty(fileno(stderr)) != 0;
#endif
  const char* color = color_enabled ? level_color(record.level) : "";
  const char* reset = color_enabled ? "\x1b[0m" : "";
  std::fprintf(stderr, "%s[%s]%s ", color, level_name(record.level), reset);
  if (record.file != nullptr && record.line != 0) {
    std::fprintf(stderr, "%s:%u ", record.file,
                 static_cast<unsigned>(record.line));
  }
  std::fputs(record.message, stderr);
  for (std::size_t i = 0; i < record.field_count; ++i) {
    std::fprintf(stderr, " %s=%s", record.fields[i].key,
                 record.fields[i].value);
  }
  std::fputc('\n', stderr);
  std::fflush(stderr);
#if defined(HYLE_OS_WINDOWS)
  char debug[1024];
  int n = std::snprintf(debug, sizeof(debug), "[%s] %s\n",
                        level_name(record.level), record.message);
  OutputDebugStringA(n > 0 ? debug : record.message);
#endif
}

bool same_sink(const LogSink& a, const LogSink& b) noexcept {
  return a.state == b.state && a.write == b.write;
}

SinkSlot* find_locked(LogSink sink, bool registered_only) noexcept {
  if (registered_only) {
    for (std::size_t i = 0; i < g_count; ++i) {
      if (same_sink(g_registered[i]->sink, sink)) return g_registered[i];
    }
    return nullptr;
  }
  for (SinkSlot& slot : g_slots) {
    if (slot.sink.write != nullptr && same_sink(slot.sink, sink)) {
      return &slot;
    }
  }
  return nullptr;
}

bool add_locked(LogSink sink) noexcept {
  // A retiring duplicate is not re-added until its last old callback leaves.
  if (SinkSlot* same = find_locked(sink, false); same != nullptr) {
    if (same->registered || same->retiring ||
        same->active.load(std::memory_order_acquire) != 0) {
      return false;
    }
    same->registered = true;
    g_registered[g_count++] = same;
    return true;
  }
  for (SinkSlot& slot : g_slots) {
    if (!slot.registered && !slot.retiring &&
        slot.active.load(std::memory_order_acquire) == 0) {
      slot.sink = sink;
      slot.registered = true;
      g_registered[g_count++] = &slot;
      return true;
    }
  }
  return false;
}

void ensure_init_locked() noexcept {
  if (!g_initialized) {
    HYLE_FATAL_ASSERT(add_locked(LogSink{nullptr, &default_write}));
    g_initialized = true;
  }
}

std::size_t current_active_count(SinkSlot* slot) noexcept {
  std::size_t count = 0;
  for (ActiveFrame* frame = g_active_frame; frame != nullptr;
       frame = frame->previous) {
    if (frame->slot == slot) ++count;
  }
  return count;
}

void wait_for_other_callbacks(SinkSlot* slot) noexcept {
  const std::size_t self = current_active_count(slot);
  std::size_t active = slot->active.load(std::memory_order_acquire);
  while (active > self) {
    slot->active.wait(active, std::memory_order_acquire);
    active = slot->active.load(std::memory_order_acquire);
  }
}

void finish_retire(SinkSlot* slot) noexcept {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (slot->retiring &&
      slot->active.load(std::memory_order_acquire) == 0) {
    slot->retiring = false;
    slot->waiting.store(false, std::memory_order_release);
    slot->sink = {};
  }
}

}  // namespace

LogSink default_sink() noexcept { return LogSink{nullptr, &default_write}; }

bool add_sink(LogSink sink) noexcept {
  if (sink.write == nullptr) return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_init_locked();
  return add_locked(sink);
}

bool remove_sink(LogSink sink) noexcept {
  SinkSlot* slot;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_init_locked();
    slot = find_locked(sink, true);
    if (slot == nullptr) return false;
    // A dispatch owns an active reference for every sink in its snapshot before
    // it invokes the first callback. Waiting for a different sink from inside a
    // callback could therefore wait on this very thread's pending callback.
    // Self-removal is safe because the trampoline no longer touches state after
    // returning from foreign code; reject all cross-removal here.
    if (g_active_frame != nullptr && g_active_frame->slot != slot) {
      return false;
    }
    slot->registered = false;
    slot->retiring = true;
    slot->waiting.store(true, std::memory_order_release);
    for (std::size_t i = 0; i < g_count; ++i) {
      if (g_registered[i] == slot) {
        g_registered[i] = g_registered[--g_count];
        break;
      }
    }
  }
  wait_for_other_callbacks(slot);
  finish_retire(slot);
  return true;
}

std::size_t sink_count() noexcept {
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_init_locked();
  return g_count;
}

std::size_t sink_capacity() noexcept { return kMaxSinks; }

void set_level(LogLevel value) noexcept {
  HYLE_FATAL_ASSERT(valid_threshold(value));
  g_level.store(value, std::memory_order_relaxed);
}

LogLevel level() noexcept {
  return g_level.load(std::memory_order_relaxed);
}

bool should_log(LogLevel value) noexcept {
  if (!valid_record_level(value)) return false;
  const LogLevel threshold = level();
  return threshold != LogLevel::Off &&
         static_cast<int>(value) >= static_cast<int>(threshold);
}

bool log_record(const LogRecord& record) noexcept {
  if (!should_log(record.level)) return false;
  HYLE_FATAL_ASSERT(record.message != nullptr);
  HYLE_FATAL_ASSERT(record.field_count == 0 || record.fields != nullptr);
  for (std::size_t i = 0; i < record.field_count; ++i) {
    HYLE_FATAL_ASSERT(record.fields[i].key != nullptr);
    HYLE_FATAL_ASSERT(record.fields[i].value != nullptr);
  }

  struct Dispatch {
    SinkSlot* slot;
    LogSink sink;
  };
  Dispatch snapshot[kMaxSinks];
  std::size_t count = 0;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_init_locked();
    for (std::size_t i = 0; i < g_count; ++i) {
      SinkSlot* slot = g_registered[i];
      slot->active.fetch_add(1, std::memory_order_relaxed);
      snapshot[count++] = Dispatch{slot, slot->sink};
    }
  }

  for (std::size_t i = 0; i < count; ++i) {
    ActiveFrame frame{snapshot[i].slot, g_active_frame};
    g_active_frame = &frame;
    snapshot[i].sink.write(snapshot[i].sink.state, record);
    g_active_frame = frame.previous;
    const std::size_t prior =
        snapshot[i].slot->active.fetch_sub(1, std::memory_order_release);
    if (snapshot[i].slot->waiting.load(std::memory_order_acquire)) {
      snapshot[i].slot->active.notify_all();
      if (prior == 1) finish_retire(snapshot[i].slot);
    }
  }
  return count != 0;
}

bool log_message(LogLevel log_level, const char* message) noexcept {
  return log_record(LogRecord{.level = log_level, .message = message});
}

bool default_sink_enabled() noexcept {
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_init_locked();
  return find_locked(default_sink(), true) != nullptr;
}

bool set_default_sink_enabled(bool enabled) noexcept {
  if (enabled) {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_init_locked();
    if (find_locked(default_sink(), true) != nullptr) return true;
    return add_locked(default_sink());
  }
  if (remove_sink(default_sink())) return true;
  // Concurrent same-direction disables are idempotent: one thread may have
  // removed it between our intent and remove_sink's lookup.
  return !default_sink_enabled();
}

}  // namespace hyle::obs
