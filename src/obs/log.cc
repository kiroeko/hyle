#include "hyle/obs/log.h"

#include <atomic>
#include <charconv>
#include <cstdio>
#include <mutex>
#include <new>
#include <string_view>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/core/environment.h"
#include "hyle/mem/reclamation.h"

#if HYLE_OS_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace hyle::obs {
namespace {

constexpr std::size_t kMaxSinks = 64;
constexpr std::size_t kActiveWaiterBit = std::size_t{1}
                                         << (sizeof(std::size_t) * 8 - 1);
constexpr std::size_t kActiveCountMask = kActiveWaiterBit - 1;

struct SinkSlot {
  LogSink sink{};
  // High bit = a remover is waiting; remaining bits = dispatch references.
  // Keeping both in one atomic makes the decrement-and-notify decision immune
  // to the missed-wake race of a separate waiter flag.
  std::atomic<std::size_t> active{0};
  bool registered = false; // guarded by RegistryState::mutex
  bool retiring = false;   // prevents reuse until remove's callback drain ends
};

struct ActiveFrame {
  SinkSlot *slot;
  ActiveFrame *previous;
  bool removed_self = false;
};

struct Dispatch {
  SinkSlot *slot;
  LogSink sink;
};

struct SinkSnapshot {
  Dispatch entries[kMaxSinks]{};
  std::size_t count = 0;
};

void default_write(void *, const LogRecord &record) noexcept;

struct RegistryState {
  RegistryState() noexcept {
    slots[0].sink = LogSink{nullptr, &default_write, nullptr};
    slots[0].registered = true;
    registered[0] = &slots[0];
    count = 1;
    snapshots[0].entries[0] = Dispatch{&slots[0], slots[0].sink};
    snapshots[0].count = 1;
    published.store(&snapshots[0], std::memory_order_relaxed);
  }

  // Mutations are rare and serialize here. Emitters never take this mutex:
  // they copy one immutable published snapshot during a short reader epoch.
  std::mutex mutex;
  std::mutex default_mutex;
  SinkSlot slots[kMaxSinks];
  SinkSlot *registered[kMaxSinks]{};
  std::size_t count = 0;
  SinkSnapshot snapshots[2];
  std::atomic<SinkSnapshot *> published{nullptr};
  // Protects only the short immutable-snapshot copy. Arbitrary callback
  // lifetime remains owned by each SinkSlot::active reference below.
  mem::ReclamationDomain snapshot_domain{0};
};

// Logging is valid from process-lifetime destructors. A normal function-local
// RegistryState would itself be destroyed during static teardown and make a
// later destructor log undefined. Match Abseil's NoDestructor shape: construct
// on first real registry use, never tear the process-global sink seam down.
template <class T> class NoDestructor {
public:
  NoDestructor() noexcept { ::new (storage_) T(); }
  T &get() noexcept { return *std::launder(reinterpret_cast<T *>(storage_)); }

private:
  alignas(T) unsigned char storage_[sizeof(T)];
};

RegistryState &registry() noexcept {
  static NoDestructor<RegistryState> state;
  return state.get();
}

constinit std::atomic<LogLevel> g_level{LogLevel::Info};
thread_local ActiveFrame *g_active_frame = nullptr;

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

const char *level_name(LogLevel level) noexcept {
  switch (level) {
  case LogLevel::Trace:
    return "trace";
  case LogLevel::Debug:
    return "debug";
  case LogLevel::Info:
    return "info";
  case LogLevel::Warn:
    return "warn";
  case LogLevel::Error:
    return "error";
  case LogLevel::Off:
    break;
  }
  return "invalid";
}

const char *level_color(LogLevel level) noexcept {
  switch (level) {
  case LogLevel::Trace:
    return "\x1b[37m";
  case LogLevel::Debug:
    return "\x1b[36m";
  case LogLevel::Info:
    return "\x1b[32m";
  case LogLevel::Warn:
    return "\x1b[33m";
  case LogLevel::Error:
    return "\x1b[31m";
  case LogLevel::Off:
    break;
  }
  return "";
}

// Keep one physical record per line and prevent arbitrary message/field data
// from injecting terminal control sequences. This is intentionally a streaming
// transform: the default sink is an I/O path and does not need a second buffer
// or an allocation merely to be safe.
void write_escaped(FILE *stream, std::string_view text) noexcept {
  for (unsigned char value : text) {
    switch (value) {
    case '\n':
      std::fputs("\\n", stream);
      break;
    case '\r':
      std::fputs("\\r", stream);
      break;
    case '\t':
      std::fputs("\\t", stream);
      break;
    default:
      if (value < 0x20 || value == 0x7f) {
        std::fprintf(stream, "\\x%02x", static_cast<unsigned>(value));
      } else {
        std::fputc(static_cast<int>(value), stream);
      }
      break;
    }
  }
}

template <class Number> void write_number(FILE *stream, Number value) noexcept {
  char bytes[128];
  const auto result = std::to_chars(bytes, bytes + sizeof(bytes), value);
  HYLE_FATAL_ASSERT(result.ec == std::errc{});
  static_cast<void>(std::fwrite(
      bytes, 1, static_cast<std::size_t>(result.ptr - bytes), stream));
}

void write_value(FILE *stream, const LogValue &value) noexcept {
  switch (value.kind()) {
  case LogValueKind::String:
    write_escaped(stream, value.as_string());
    return;
  case LogValueKind::SignedInteger:
    write_number(stream, value.as_signed_integer());
    return;
  case LogValueKind::UnsignedInteger:
    write_number(stream, value.as_unsigned_integer());
    return;
  case LogValueKind::FloatingPoint:
    write_number(stream, value.as_floating_point());
    return;
  case LogValueKind::Boolean:
    std::fputs(value.as_boolean() ? "true" : "false", stream);
    return;
  }
  HYLE_UNREACHABLE();
}

void write_attributes(FILE *stream, std::span<const LogAttribute> attributes,
                      std::string_view prefix = {}) noexcept {
  for (const LogAttribute &attribute : attributes) {
    std::fputc(' ', stream);
    if (!prefix.empty()) {
      static_cast<void>(std::fwrite(prefix.data(), 1, prefix.size(), stream));
    }
    write_escaped(stream, attribute.key);
    std::fputc('=', stream);
    write_value(stream, attribute.value);
  }
}

#if HYLE_OS_WINDOWS
// OutputDebugStringA has no length-taking form. Mirror the same escaped logical
// record as stderr into one bounded buffer so hostile control bytes cannot
// forge debugger lines and the mirror does not allocate in a diagnostic path.
class DebugLine final {
public:
  void append(const char *text) noexcept {
    while (*text != 0)
      append_byte(static_cast<unsigned char>(*text++));
  }

  void append_escaped(std::string_view text) noexcept {
    constexpr char kHex[] = "0123456789abcdef";
    for (unsigned char value : text) {
      switch (value) {
      case '\n':
        append("\\n");
        break;
      case '\r':
        append("\\r");
        break;
      case '\t':
        append("\\t");
        break;
      default:
        if (value < 0x20 || value == 0x7f) {
          append("\\x");
          append_byte(static_cast<unsigned char>(kHex[value >> 4]));
          append_byte(static_cast<unsigned char>(kHex[value & 0x0f]));
        } else {
          append_byte(value);
        }
        break;
      }
    }
  }

  void append_raw(std::string_view text) noexcept {
    for (unsigned char value : text)
      append_byte(value);
  }

  template <class Number> void append_number(Number value) noexcept {
    char bytes[128];
    const auto result = std::to_chars(bytes, bytes + sizeof(bytes), value);
    HYLE_FATAL_ASSERT(result.ec == std::errc{});
    append_raw(
        std::string_view(bytes, static_cast<std::size_t>(result.ptr - bytes)));
  }

  void append_value(const LogValue &value) noexcept {
    switch (value.kind()) {
    case LogValueKind::String:
      append_escaped(value.as_string());
      return;
    case LogValueKind::SignedInteger:
      append_number(value.as_signed_integer());
      return;
    case LogValueKind::UnsignedInteger:
      append_number(value.as_unsigned_integer());
      return;
    case LogValueKind::FloatingPoint:
      append_number(value.as_floating_point());
      return;
    case LogValueKind::Boolean:
      append(value.as_boolean() ? "true" : "false");
      return;
    }
    HYLE_UNREACHABLE();
  }

  void append_attributes(std::span<const LogAttribute> attributes,
                         std::string_view prefix = {}) noexcept {
    for (const LogAttribute &attribute : attributes) {
      append(" ");
      append_raw(prefix);
      append_escaped(attribute.key);
      append("=");
      append_value(attribute.value);
    }
  }

  void emit() noexcept {
    constexpr char kMarker[] = "...[truncated]\n";
    if (!truncated_)
      append_byte('\n');
    if (truncated_) {
      constexpr std::size_t kMarkerSize = sizeof(kMarker) - 1;
      static_assert(kMarkerSize < sizeof(bytes_));
      size_ = sizeof(bytes_) - 1 - kMarkerSize;
      std::memcpy(bytes_ + size_, kMarker, kMarkerSize);
      size_ += kMarkerSize;
    }
    bytes_[size_] = '\0';
    OutputDebugStringA(bytes_);
  }

private:
  void append_byte(unsigned char value) noexcept {
    if (size_ + 1 < sizeof(bytes_)) {
      bytes_[size_++] = static_cast<char>(value);
    } else {
      truncated_ = true;
    }
  }

  char bytes_[4096]{};
  std::size_t size_ = 0;
  bool truncated_ = false;
};
#endif

void default_write(void *, const LogRecord &record) noexcept {
  std::lock_guard<std::mutex> lock(registry().default_mutex);
#if HYLE_OS_WINDOWS
  const bool color_enabled = _isatty(_fileno(stderr)) != 0;
#else
  const bool color_enabled = isatty(fileno(stderr)) != 0;
#endif
  const char *color = color_enabled ? level_color(record.level) : "";
  const char *reset = color_enabled ? "\x1b[0m" : "";
  std::fprintf(stderr, "%s[%s]%s ", color, level_name(record.level), reset);
  if (!record.category.empty()) {
    std::fputs("category=", stderr);
    write_escaped(stderr, record.category);
    std::fputc(' ', stderr);
  }
  if (!record.event.empty()) {
    std::fputs("event=", stderr);
    write_escaped(stderr, record.event);
    std::fputc(' ', stderr);
  }
  if (!record.source.file.empty()) {
    write_escaped(stderr, record.source.file);
    if (record.source.line != 0) {
      std::fputc(':', stderr);
      write_number(stderr, record.source.line);
      if (record.source.column != 0) {
        std::fputc(':', stderr);
        write_number(stderr, record.source.column);
      }
    }
    std::fputc(' ', stderr);
  }
  write_escaped(stderr, record.message);
  write_attributes(stderr, record.attributes);
  write_attributes(stderr, record.correlation_attributes, "correlation.");
  if (!record.source.function.empty()) {
    std::fputs(" source.function=", stderr);
    write_escaped(stderr, record.source.function);
  }
  if (record.timestamp.has_value()) {
    std::fputs(" timestamp.seconds=", stderr);
    write_number(stderr, record.timestamp->seconds);
    std::fputs(" timestamp.nanoseconds=", stderr);
    write_number(stderr, record.timestamp->nanoseconds);
  }
  std::fputc('\n', stderr);
  std::fflush(stderr);
#if HYLE_OS_WINDOWS
  DebugLine debug;
  debug.append("[");
  debug.append(level_name(record.level));
  debug.append("] ");
  if (!record.category.empty()) {
    debug.append("category=");
    debug.append_escaped(record.category);
    debug.append(" ");
  }
  if (!record.event.empty()) {
    debug.append("event=");
    debug.append_escaped(record.event);
    debug.append(" ");
  }
  if (!record.source.file.empty()) {
    debug.append_escaped(record.source.file);
    if (record.source.line != 0) {
      debug.append(":");
      debug.append_number(record.source.line);
      if (record.source.column != 0) {
        debug.append(":");
        debug.append_number(record.source.column);
      }
    }
    debug.append(" ");
  }
  debug.append_escaped(record.message);
  debug.append_attributes(record.attributes);
  debug.append_attributes(record.correlation_attributes, "correlation.");
  if (!record.source.function.empty()) {
    debug.append(" source.function=");
    debug.append_escaped(record.source.function);
  }
  if (record.timestamp.has_value()) {
    debug.append(" timestamp.seconds=");
    debug.append_number(record.timestamp->seconds);
    debug.append(" timestamp.nanoseconds=");
    debug.append_number(record.timestamp->nanoseconds);
  }
  debug.emit();
#endif
}

bool same_sink(const LogSink &a, const LogSink &b) noexcept {
  return a.state == b.state && a.write == b.write;
}

SinkSlot *find_locked(RegistryState &state, LogSink sink,
                      bool registered_only) noexcept {
  if (registered_only) {
    for (std::size_t i = 0; i < state.count; ++i) {
      if (same_sink(state.registered[i]->sink, sink)) {
        return state.registered[i];
      }
    }
    return nullptr;
  }
  for (SinkSlot &slot : state.slots) {
    if (slot.sink.write != nullptr && same_sink(slot.sink, sink)) {
      return &slot;
    }
  }
  return nullptr;
}

// Caller holds state.mutex. Two in-place immutable snapshots are enough: wait
// until no emitter is copying the inactive generation, fill it, then publish.
// A remover performs one more reader drain after publication before waiting on
// the target slot, closing the old-snapshot -> active-reference handoff race.
void publish_locked(RegistryState &state) noexcept {
  static_cast<void>(state.snapshot_domain.synchronize());
  SinkSnapshot *current = state.published.load(std::memory_order_relaxed);
  SinkSnapshot *next = current == &state.snapshots[0] ? &state.snapshots[1]
                                                      : &state.snapshots[0];
  next->count = state.count;
  for (std::size_t i = 0; i < state.count; ++i) {
    next->entries[i] = Dispatch{state.registered[i], state.registered[i]->sink};
  }
  state.published.store(next, std::memory_order_release);
}

Status add_locked(RegistryState &state, LogSink sink) noexcept {
  // A retiring duplicate is not re-added until its last old callback leaves.
  if (SinkSlot *same = find_locked(state, sink, false); same != nullptr) {
    if (same->registered) {
      return Status::AlreadyExists("log sink is already registered");
    }
    if (same->retiring || (same->active.load(std::memory_order_acquire) &
                           kActiveCountMask) != 0) {
      return Status::FailedPrecondition("log sink is still retiring");
    }
    same->registered = true;
    state.registered[state.count++] = same;
    publish_locked(state);
    return {};
  }
  for (SinkSlot &slot : state.slots) {
    if (!slot.registered && !slot.retiring &&
        (slot.active.load(std::memory_order_acquire) & kActiveCountMask) == 0) {
      slot.sink = sink;
      slot.registered = true;
      state.registered[state.count++] = &slot;
      publish_locked(state);
      return {};
    }
  }
  return Status::ResourceExhausted("log sink registry is full");
}

std::size_t current_active_count(SinkSlot *slot) noexcept {
  std::size_t count = 0;
  for (ActiveFrame *frame = g_active_frame; frame != nullptr;
       frame = frame->previous) {
    if (frame->slot == slot)
      ++count;
  }
  return count;
}

void wait_for_other_callbacks(SinkSlot *slot) noexcept {
  const std::size_t self = current_active_count(slot);
  std::size_t active =
      slot->active.fetch_or(kActiveWaiterBit, std::memory_order_acq_rel);
  while ((active & kActiveCountMask) > self) {
    slot->active.wait(active, std::memory_order_acquire);
    active = slot->active.load(std::memory_order_acquire);
  }
}

void finish_retire(RegistryState &state, SinkSlot *slot) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  if (slot->retiring &&
      (slot->active.load(std::memory_order_acquire) & kActiveCountMask) == 0) {
    slot->retiring = false;
    slot->sink = {};
    slot->active.store(0, std::memory_order_release);
  }
}

std::size_t snapshot_sinks(RegistryState &state,
                           Dispatch (&snapshot)[kMaxSinks]) noexcept {
  auto guard = state.snapshot_domain.read();
  const SinkSnapshot *published = guard.protect(state.published);
  const std::size_t count = published->count;
  for (std::size_t i = 0; i < count; ++i) {
    SinkSlot *slot = published->entries[i].slot;
    slot->active.fetch_add(1, std::memory_order_relaxed);
    snapshot[i] = published->entries[i];
  }
  guard.reset();
  return count;
}

void release_dispatch(RegistryState &state, Dispatch &dispatch,
                      bool finish_self_retire = false) noexcept {
  const std::size_t prior =
      dispatch.slot->active.fetch_sub(1, std::memory_order_release);
  if ((prior & kActiveWaiterBit) != 0) {
    dispatch.slot->active.notify_all();
    // An external remover performs the final cleanup itself before returning.
    // Only self-removal must defer cleanup until this callback releases its own
    // reference. Letting every last releaser race to finish a retirement can
    // leave a stale helper that observes a later reuse of the same fixed slot.
    if (finish_self_retire && (prior & kActiveCountMask) == 1) {
      finish_retire(state, dispatch.slot);
    }
  }
}

} // namespace

LogSink default_sink() noexcept {
  return LogSink{nullptr, &default_write, nullptr};
}

Status add_sink(LogSink sink) noexcept {
  if (sink.write == nullptr) {
    return Status::InvalidArgument("log sink write callback must be non-null");
  }
  RegistryState &state = registry();
  std::lock_guard<std::mutex> lock(state.mutex);
  return add_locked(state, sink);
}

Status remove_sink(LogSink sink) noexcept {
  if (sink.write == nullptr) {
    return Status::InvalidArgument("log sink write callback must be non-null");
  }
  RegistryState &state = registry();
  SinkSlot *slot;
  bool self_removal = false;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    slot = find_locked(state, sink, true);
    if (slot == nullptr) {
      return Status::NotFound("log sink is not registered");
    }
    // A dispatch owns an active reference for every sink in its snapshot before
    // it invokes the first callback. Waiting for a different sink from inside a
    // callback could therefore wait on this very thread's pending callback.
    // Self-removal is safe because the trampoline no longer touches state after
    // returning from foreign code; reject all cross-removal here.
    if (g_active_frame != nullptr && g_active_frame->slot != slot) {
      return Status::FailedPrecondition(
          "a sink callback cannot remove a different sink");
    }
    self_removal = g_active_frame != nullptr;
    if (self_removal)
      g_active_frame->removed_self = true;
    slot->registered = false;
    slot->retiring = true;
    for (std::size_t i = 0; i < state.count; ++i) {
      if (state.registered[i] == slot) {
        state.registered[i] = state.registered[--state.count];
        break;
      }
    }
    publish_locked(state);
  }
  // Close the old-snapshot -> active-reference handoff before inspecting the
  // target's callback references. This grace period does not cover callbacks.
  static_cast<void>(state.snapshot_domain.synchronize());
  wait_for_other_callbacks(slot);
  if (!self_removal)
    finish_retire(state, slot);
  return {};
}

Result<SinkRegistration> SinkRegistration::create(LogSink sink) noexcept {
  Status status = add_sink(sink);
  if (!status.ok())
    return make_unexpected(std::move(status));
  SinkRegistration registration(sink);
  return registration;
}

SinkRegistration::~SinkRegistration() noexcept { reset_or_abort(); }

SinkRegistration::SinkRegistration(SinkRegistration &&other) noexcept
    : sink_(std::exchange(other.sink_, {})),
      registered_(std::exchange(other.registered_, false)) {}

SinkRegistration &
SinkRegistration::operator=(SinkRegistration &&other) noexcept {
  if (this == &other)
    return *this;
  reset_or_abort();
  sink_ = std::exchange(other.sink_, {});
  registered_ = std::exchange(other.registered_, false);
  return *this;
}

Status SinkRegistration::reset() noexcept {
  if (!registered_)
    return {};
  Status status = remove_sink(sink_);
  if (status.ok()) {
    sink_ = {};
    registered_ = false;
  }
  return status;
}

void SinkRegistration::reset_or_abort() noexcept {
  Status status = reset();
  HYLE_FATAL_ASSERT(status.ok());
}

std::size_t sink_count() noexcept {
  RegistryState &state = registry();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.count;
}

std::size_t sink_capacity() noexcept { return kMaxSinks; }

std::size_t flush_sinks() noexcept {
  // A flush callback may itself log. Nested registered-sink dispatch would
  // otherwise recurse through the same arbitrary callback graph.
  if (g_active_frame != nullptr)
    return 0;

  RegistryState &state = registry();
  Dispatch snapshot[kMaxSinks];
  const std::size_t count = snapshot_sinks(state, snapshot);
  std::size_t flushed = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (snapshot[i].sink.flush != nullptr) {
      ActiveFrame frame{snapshot[i].slot, g_active_frame};
      g_active_frame = &frame;
      snapshot[i].sink.flush(snapshot[i].sink.state);
      g_active_frame = frame.previous;
      ++flushed;
      release_dispatch(state, snapshot[i], frame.removed_self);
    } else {
      release_dispatch(state, snapshot[i]);
    }
  }
  return flushed;
}

void set_level(LogLevel value) noexcept {
  HYLE_FATAL_ASSERT(valid_threshold(value));
  g_level.store(value, std::memory_order_relaxed);
}

LogLevel level() noexcept { return g_level.load(std::memory_order_relaxed); }

bool should_log(LogLevel value) noexcept {
  if (!valid_record_level(value))
    return false;
  const LogLevel threshold = level();
  return threshold != LogLevel::Off &&
         static_cast<int>(value) >= static_cast<int>(threshold);
}

bool detail::log_record_unchecked(const LogRecord &record) noexcept {
  // Match mature sink registries: a sink is allowed to log for diagnostics,
  // but that nested record must not re-enter the registered sink graph. If the
  // caller kept the default sink enabled, preserve the record on stderr;
  // otherwise report that no sink received it.
  if (g_active_frame != nullptr) {
    if (!default_sink_enabled())
      return false;
    default_write(nullptr, record);
    return true;
  }

  RegistryState &state = registry();
  Dispatch snapshot[kMaxSinks];
  const std::size_t count = snapshot_sinks(state, snapshot);

  for (std::size_t i = 0; i < count; ++i) {
    ActiveFrame frame{snapshot[i].slot, g_active_frame};
    g_active_frame = &frame;
    snapshot[i].sink.write(snapshot[i].sink.state, record);
    g_active_frame = frame.previous;
    release_dispatch(state, snapshot[i], frame.removed_self);
  }
  return count != 0;
}

bool log_record(const LogRecord &record) noexcept {
  HYLE_FATAL_ASSERT(!record.timestamp.has_value() ||
                    time::unix_timestamp_valid(*record.timestamp));
  if (!should_log(record.level))
    return false;
  return detail::log_record_unchecked(record);
}

bool log_message(LogLevel log_level, std::string_view message) noexcept {
  if (!should_log(log_level))
    return false;
  const LogRecord record{.level = log_level, .message = message};
  return detail::log_record_unchecked(record);
}

bool default_sink_enabled() noexcept {
  RegistryState &state = registry();
  std::lock_guard<std::mutex> lock(state.mutex);
  return find_locked(state, default_sink(), true) != nullptr;
}

Status set_default_sink_enabled(bool enabled) noexcept {
  RegistryState &state = registry();
  if (enabled) {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (find_locked(state, default_sink(), true) != nullptr)
      return {};
    return add_locked(state, default_sink());
  }
  Status status = remove_sink(default_sink());
  if (status.ok())
    return {};
  // Concurrent same-direction disables are idempotent: one thread may have
  // removed it between our intent and remove_sink's lookup.
  if (status.code() == StatusCode::NotFound && !default_sink_enabled()) {
    return {};
  }
  return status;
}

} // namespace hyle::obs
