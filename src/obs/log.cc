#include "hyle/obs/log.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <new>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/core/environment.h"

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
constexpr std::size_t kSnapshotWaiterOne = std::size_t{1}
                                           << (sizeof(std::size_t) * 4);
constexpr std::size_t kSnapshotReaderMask = kSnapshotWaiterOne - 1;
static_assert(sizeof(std::size_t) >= 4);

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
  // Low half = readers copying a published snapshot; high half = mutations
  // waiting for a zero-reader epoch. A single word supports multiple removers
  // without the missed wake possible with one shared boolean waiter flag.
  std::atomic<std::size_t> snapshot_state{0};
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

  void append_line(std::uint32_t line) noexcept {
    char digits[16];
    const int count = std::snprintf(digits, sizeof(digits), "%u",
                                    static_cast<unsigned>(line));
    if (count > 0)
      append(digits);
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
  if (record.file != nullptr && record.line != 0) {
    write_escaped(stderr, record.file);
    std::fprintf(stderr, ":%u ", static_cast<unsigned>(record.line));
  }
  write_escaped(stderr, record.message);
  for (const LogField &field : record.fields) {
    std::fputc(' ', stderr);
    write_escaped(stderr, field.key);
    std::fputc('=', stderr);
    write_escaped(stderr, field.value);
  }
  std::fputc('\n', stderr);
  std::fflush(stderr);
#if HYLE_OS_WINDOWS
  DebugLine debug;
  debug.append("[");
  debug.append(level_name(record.level));
  debug.append("] ");
  if (record.file != nullptr && record.line != 0) {
    debug.append_escaped(record.file);
    debug.append(":");
    debug.append_line(record.line);
    debug.append(" ");
  }
  debug.append_escaped(record.message);
  for (const LogField &field : record.fields) {
    debug.append(" ");
    debug.append_escaped(field.key);
    debug.append("=");
    debug.append_escaped(field.value);
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

void wait_for_snapshot_readers(RegistryState &state) noexcept {
  std::size_t snapshot = state.snapshot_state.fetch_add(
      kSnapshotWaiterOne, std::memory_order_seq_cst);
  snapshot += kSnapshotWaiterOne;
  while ((snapshot & kSnapshotReaderMask) != 0) {
    state.snapshot_state.wait(snapshot, std::memory_order_seq_cst);
    snapshot = state.snapshot_state.load(std::memory_order_seq_cst);
  }
  state.snapshot_state.fetch_sub(kSnapshotWaiterOne, std::memory_order_seq_cst);
}

// Caller holds state.mutex. Two in-place immutable snapshots are enough: wait
// until no emitter is copying the inactive generation, fill it, then publish.
// A remover performs one more reader drain after publication before waiting on
// the target slot, closing the old-snapshot -> active-reference handoff race.
void publish_locked(RegistryState &state) noexcept {
  wait_for_snapshot_readers(state);
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
  state.snapshot_state.fetch_add(1, std::memory_order_seq_cst);
  const SinkSnapshot *published =
      state.published.load(std::memory_order_acquire);
  const std::size_t count = published->count;
  for (std::size_t i = 0; i < count; ++i) {
    SinkSlot *slot = published->entries[i].slot;
    slot->active.fetch_add(1, std::memory_order_relaxed);
    snapshot[i] = published->entries[i];
  }
  const std::size_t prior =
      state.snapshot_state.fetch_sub(1, std::memory_order_seq_cst);
  if ((prior & ~kSnapshotReaderMask) != 0) {
    state.snapshot_state.notify_all();
  }
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
  wait_for_snapshot_readers(state);
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
