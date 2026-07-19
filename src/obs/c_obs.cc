#include "hyle/c/obs.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>

#include "hyle/obs/log.h"
#include "hyle/obs/trace.h"

#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
#  include <tracy/TracyC.h>
#endif

struct hyle_obs_sink_o {
  hyle_obs_sink_write_fn write;
  void* state;
  hyle::obs::LogSink native;

  hyle_obs_sink_o(hyle_obs_sink_write_fn callback, void* callback_state) noexcept
      : write(callback),
        state(callback_state),
        native{this, &hyle_obs_sink_o::dispatch} {}

  static void dispatch(void* opaque,
                       const hyle::obs::LogRecord& record) noexcept {
    // Copy both values before entering foreign code. This makes destroying the
    // handle from inside its own callback safe: remove_sink drains every OTHER
    // callback and this trampoline touches no wrapper state after the call.
    auto* self = static_cast<hyle_obs_sink_o*>(opaque);
    hyle_obs_sink_write_fn callback = self->write;
    void* callback_state = self->state;
    hyle_obs_log_record c_record{
        sizeof(hyle_obs_log_record),
        static_cast<hyle_obs_log_level>(record.level),
        record.message,
        record.fields,
        record.field_count,
        record.file,
        record.function,
        record.line,
        record.timestamp_ns,
    };
    callback(callback_state, &c_record);
  }
};

namespace {

thread_local const char* g_last_error = "";

void clear_error() noexcept { g_last_error = ""; }
void set_error(const char* message) noexcept { g_last_error = message; }

bool valid_level(hyle_obs_log_level level, bool threshold) noexcept {
  const int raw = static_cast<int>(level);
  const int top = threshold ? HYLE_OBS_LOG_OFF : HYLE_OBS_LOG_ERROR;
  return raw >= HYLE_OBS_LOG_TRACE && raw <= top;
}

bool has_member(std::uint32_t struct_size, std::size_t offset,
                std::size_t member_size) noexcept {
  return struct_size >= offset && struct_size - offset >= member_size;
}

template <class Member>
bool has_member(std::uint32_t struct_size, std::size_t offset) noexcept {
  return has_member(struct_size, offset, sizeof(Member));
}

#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
hyle_obs_trace_zone from_tracy_zone(TracyCZoneCtx context) noexcept {
  hyle_obs_trace_zone result{};
  static_assert(sizeof(context) <= sizeof(result));
  std::memcpy(&result, &context, sizeof(context));
  return result;
}

TracyCZoneCtx to_tracy_zone(hyle_obs_trace_zone zone) noexcept {
  TracyCZoneCtx result{};
  static_assert(sizeof(result) <= sizeof(zone));
  std::memcpy(&result, &zone, sizeof(result));
  return result;
}
#endif

static_assert(std::is_standard_layout_v<hyle_obs_log_field>);
static_assert(std::is_trivially_copyable_v<hyle_obs_log_field>);
static_assert(std::is_standard_layout_v<hyle_obs_log_record>);
static_assert(std::is_trivially_copyable_v<hyle_obs_log_record>);
static_assert(offsetof(hyle_obs_log_record, struct_size) == 0);
static_assert(static_cast<int>(hyle::obs::LogLevel::Trace) ==
              HYLE_OBS_LOG_TRACE);
static_assert(static_cast<int>(hyle::obs::LogLevel::Debug) ==
              HYLE_OBS_LOG_DEBUG);
static_assert(static_cast<int>(hyle::obs::LogLevel::Info) ==
              HYLE_OBS_LOG_INFO);
static_assert(static_cast<int>(hyle::obs::LogLevel::Warn) ==
              HYLE_OBS_LOG_WARN);
static_assert(static_cast<int>(hyle::obs::LogLevel::Error) ==
              HYLE_OBS_LOG_ERROR);
static_assert(static_cast<int>(hyle::obs::LogLevel::Off) == HYLE_OBS_LOG_OFF);

}  // namespace

extern "C" {

const char* hyle_obs_last_error(void) { return g_last_error; }

int hyle_obs_set_level(hyle_obs_log_level level) {
  if (!valid_level(level, true)) {
    set_error("invalid log threshold");
    return 0;
  }
  hyle::obs::set_level(static_cast<hyle::obs::LogLevel>(level));
  clear_error();
  return 1;
}

hyle_obs_log_level hyle_obs_get_level(void) {
  clear_error();
  return static_cast<hyle_obs_log_level>(hyle::obs::level());
}

int hyle_obs_should_log(hyle_obs_log_level level) {
  if (!valid_level(level, false)) {
    set_error("invalid log record level");
    return 0;
  }
  clear_error();
  return hyle::obs::should_log(
             static_cast<hyle::obs::LogLevel>(level))
             ? 1
             : 0;
}

int hyle_obs_emit_record(const hyle_obs_log_record* record) {
  constexpr std::size_t kRequired =
      offsetof(hyle_obs_log_record, message) +
      sizeof(const char*);
  if (record == nullptr || record->struct_size < kRequired) {
    set_error("log record is null or has an invalid struct_size");
    return 0;
  }
  if (!valid_level(record->level, false) || record->message == nullptr) {
    set_error("log record has an invalid level or null message");
    return 0;
  }

  hyle::obs::LogRecord native{
      .level = static_cast<hyle::obs::LogLevel>(record->level),
      .message = record->message,
  };
  if (has_member<const hyle_obs_log_field*>(
          record->struct_size, offsetof(hyle_obs_log_record, fields))) {
    native.fields = record->fields;
  }
  if (has_member<std::size_t>(
          record->struct_size, offsetof(hyle_obs_log_record, field_count))) {
    native.field_count = record->field_count;
  }
  if (native.field_count != 0 && native.fields == nullptr) {
    set_error("log record field_count requires a non-null fields array");
    return 0;
  }
  for (std::size_t i = 0; i < native.field_count; ++i) {
    if (native.fields[i].key == nullptr ||
        native.fields[i].value == nullptr) {
      set_error("log field key and value must be non-null");
      return 0;
    }
  }
  if (has_member<const char*>(
          record->struct_size, offsetof(hyle_obs_log_record, file))) {
    native.file = record->file;
  }
  if (has_member<const char*>(
          record->struct_size, offsetof(hyle_obs_log_record, function))) {
    native.function = record->function;
  }
  if (has_member<std::uint32_t>(
          record->struct_size, offsetof(hyle_obs_log_record, line))) {
    native.line = record->line;
  }
  if (has_member<std::uint64_t>(
          record->struct_size, offsetof(hyle_obs_log_record, timestamp_ns))) {
    native.timestamp_ns = record->timestamp_ns;
  }

  clear_error();
  return hyle::obs::log_record(native) ? 1 : 0;
}

int hyle_obs_log(hyle_obs_log_level level, const char* message) {
  if (!valid_level(level, false) || message == nullptr) {
    set_error("log level is invalid or message is null");
    return 0;
  }
  clear_error();
  return hyle::obs::log_message(
             static_cast<hyle::obs::LogLevel>(level), message)
             ? 1
             : 0;
}

hyle_obs_sink* hyle_obs_sink_create(hyle_obs_sink_write_fn write,
                                    void* state) {
  if (write == nullptr) {
    set_error("sink callback must be non-null");
    return nullptr;
  }
  auto* sink = new (std::nothrow) hyle_obs_sink(write, state);
  if (sink == nullptr) {
    set_error("sink allocation failed");
    return nullptr;
  }
  if (!hyle::obs::add_sink(sink->native)) {
    delete sink;
    set_error("sink registry is full");
    return nullptr;
  }
  clear_error();
  return sink;
}

int hyle_obs_sink_destroy(hyle_obs_sink* sink) {
  if (sink == nullptr) {
    clear_error();
    return 1;
  }
  if (!hyle::obs::remove_sink(sink->native)) {
    set_error("sink is absent or cross-removal was attempted in a callback");
    return 0;
  }
  delete sink;
  clear_error();
  return 1;
}

size_t hyle_obs_sink_count(void) {
  clear_error();
  return hyle::obs::sink_count();
}

size_t hyle_obs_sink_capacity(void) {
  clear_error();
  return hyle::obs::sink_capacity();
}

int hyle_obs_default_sink_enabled(void) {
  clear_error();
  return hyle::obs::default_sink_enabled() ? 1 : 0;
}

int hyle_obs_set_default_sink_enabled(int enabled) {
  if (!hyle::obs::set_default_sink_enabled(enabled != 0)) {
    set_error("default sink state could not be changed");
    return 0;
  }
  clear_error();
  return 1;
}

int hyle_obs_trace_enabled(void) {
#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
  return 1;
#else
  return 0;
#endif
}

hyle_obs_trace_zone hyle_obs_trace_zone_begin(
    const char* name, const char* file, const char* function, uint32_t line) {
  hyle_obs_trace_zone result{};
#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
  const char* safe_name = name != nullptr ? name : "";
  const char* safe_file = file != nullptr ? file : "";
  const char* safe_function = function != nullptr ? function : "";
  const uint64_t source = ___tracy_alloc_srcloc_name(
      line, safe_file, std::strlen(safe_file), safe_function,
      std::strlen(safe_function), safe_name, std::strlen(safe_name), 0);
  result = from_tracy_zone(___tracy_emit_zone_begin_alloc(source, 1));
#else
  (void)name;
  (void)file;
  (void)function;
  (void)line;
#endif
  return result;
}

void hyle_obs_trace_zone_end(hyle_obs_trace_zone zone) {
#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
  ___tracy_emit_zone_end(to_tracy_zone(zone));
#else
  (void)zone;
#endif
}

void hyle_obs_trace_zone_text(hyle_obs_trace_zone zone, const char* text) {
#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
  if (text != nullptr) {
    ___tracy_emit_zone_text(to_tracy_zone(zone), text, std::strlen(text));
  }
#else
  (void)zone;
  (void)text;
#endif
}

void hyle_obs_trace_plot(const char* name, double value) {
  if (name == nullptr) return;
  (void)value;
  HYLE_TRACE_PLOT(name, value);
}

void hyle_obs_trace_counter(const char* name, int64_t value) {
  if (name == nullptr) return;
  (void)value;
  HYLE_TRACE_COUNTER(name, value);
}

void hyle_obs_trace_message(const char* message) {
  if (message == nullptr) return;
  HYLE_TRACE_MESSAGE(message);
}

void hyle_obs_trace_frame(const char* stable_name) {
#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
  if (stable_name != nullptr) {
    FrameMarkNamed(stable_name);
  } else {
    FrameMark;
  }
#else
  (void)stable_name;
#endif
}

void hyle_obs_trace_alloc(const void* pointer, size_t size,
                          const char* stable_pool_name) {
  if (pointer == nullptr) return;
#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
  if (stable_pool_name != nullptr) {
    TracyAllocN(pointer, size, stable_pool_name);
  } else {
    TracyAlloc(pointer, size);
  }
#else
  (void)size;
  (void)stable_pool_name;
#endif
}

void hyle_obs_trace_free(const void* pointer,
                         const char* stable_pool_name) {
  if (pointer == nullptr) return;
#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
  if (stable_pool_name != nullptr) {
    TracyFreeN(pointer, stable_pool_name);
  } else {
    TracyFree(pointer);
  }
#else
  (void)stable_pool_name;
#endif
}

void hyle_obs_trace_thread_name(const char* name) {
  if (name == nullptr) return;
  HYLE_TRACE_THREAD_NAME(name);
}

}  // extern "C"
