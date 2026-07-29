#include "hyle/c/obs.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string_view>
#include <type_traits>
#include <utility>

#include "hyle/obs/log.h"
#include "hyle/obs/trace.h"
#include "../error/status_access.h"

#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
#  include <tracy/TracyC.h>
#endif

struct hyle_obs_sink_o {
  hyle_obs_sink_write_fn write;
  hyle_obs_sink_flush_fn flush;
  void* state;
  hyle::obs::LogSink native;

  hyle_obs_sink_o(hyle_obs_sink_write_fn callback,
                  hyle_obs_sink_flush_fn flush_callback,
                  void* callback_state) noexcept
      : write(callback),
        flush(flush_callback),
        state(callback_state),
        native{this, &hyle_obs_sink_o::dispatch,
               flush_callback != nullptr ? &hyle_obs_sink_o::flush_dispatch
                                         : nullptr} {}

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
        record.message.data(),
        record.message.size(),
        record.fields,
        record.field_count,
        record.file,
        record.function,
        record.line,
        record.timestamp_ns,
    };
    callback(callback_state, &c_record);
  }

  static void flush_dispatch(void* opaque) noexcept {
    // The native sink only installs this trampoline when flush is non-null.
    auto* self = static_cast<hyle_obs_sink_o*>(opaque);
    hyle_obs_sink_flush_fn callback = self->flush;
    void* callback_state = self->state;
    callback(callback_state);
  }
};

namespace {

hyle_status steal_status(hyle::Status&& status) noexcept {
  return hyle_status{hyle::detail::StatusAccess::release(status)};
}

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

hyle_status hyle_obs_set_level(hyle_obs_log_level level) {
  if (!valid_level(level, true)) {
    return steal_status(
        hyle::Status::InvalidArgument("invalid log threshold"));
  }
  hyle::obs::set_level(static_cast<hyle::obs::LogLevel>(level));
  return hyle_status{};
}

hyle_obs_log_level hyle_obs_get_level(void) {
  return static_cast<hyle_obs_log_level>(hyle::obs::level());
}

int hyle_obs_should_log(hyle_obs_log_level level) {
  if (!valid_level(level, false)) return 0;
  return hyle::obs::should_log(
             static_cast<hyle::obs::LogLevel>(level))
             ? 1
             : 0;
}

hyle_status hyle_obs_emit_record(int* out_emitted,
                                 const hyle_obs_log_record* record) {
  if (out_emitted != nullptr) *out_emitted = 0;
  constexpr std::size_t kRequired =
      offsetof(hyle_obs_log_record, message_size) + sizeof(std::size_t);
  if (record == nullptr || record->struct_size < kRequired) {
    return steal_status(hyle::Status::InvalidArgument(
        "log record is null or has an invalid struct_size"));
  }
  if (!valid_level(record->level, false) ||
      (record->message == nullptr && record->message_size != 0)) {
    return steal_status(hyle::Status::InvalidArgument(
        "log record has an invalid level or message byte view"));
  }

  const char* message = record->message != nullptr ? record->message : "";
  const std::size_t message_size =
      record->message_size == HYLE_OBS_NUL_TERMINATED
          ? std::strlen(message)
          : record->message_size;

  hyle::obs::LogRecord native{
      .level = static_cast<hyle::obs::LogLevel>(record->level),
      .message = std::string_view(message, message_size),
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
    return steal_status(hyle::Status::InvalidArgument(
        "log record field_count requires a non-null fields array"));
  }
  for (std::size_t i = 0; i < native.field_count; ++i) {
    if (native.fields[i].key == nullptr ||
        native.fields[i].value == nullptr) {
      return steal_status(hyle::Status::InvalidArgument(
          "log field key and value must be non-null"));
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

  const bool emitted = hyle::obs::log_record(native);
  if (out_emitted != nullptr) *out_emitted = emitted ? 1 : 0;
  return hyle_status{};
}

hyle_status hyle_obs_log(int* out_emitted, hyle_obs_log_level level,
                         const char* message) {
  if (message == nullptr) {
    if (out_emitted != nullptr) *out_emitted = 0;
    return steal_status(hyle::Status::InvalidArgument(
        "log message is null"));
  }
  return hyle_obs_log_n(out_emitted, level, message, std::strlen(message));
}

hyle_status hyle_obs_log_n(int* out_emitted, hyle_obs_log_level level,
                           const char* message, size_t message_size) {
  if (out_emitted != nullptr) *out_emitted = 0;
  if (!valid_level(level, false) ||
      (message == nullptr && message_size != 0)) {
    return steal_status(hyle::Status::InvalidArgument(
        "log level is invalid or message byte view is invalid"));
  }
  if (message == nullptr) message = "";
  const bool emitted = hyle::obs::log_message(
      static_cast<hyle::obs::LogLevel>(level),
      std::string_view(message, message_size));
  if (out_emitted != nullptr) *out_emitted = emitted ? 1 : 0;
  return hyle_status{};
}

hyle_status hyle_obs_sink_create(hyle_obs_sink** out,
                                 hyle_obs_sink_write_fn write,
                                 hyle_obs_sink_flush_fn flush, void* state) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "sink output must be non-null"));
  }
  *out = nullptr;
  if (write == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "sink write callback must be non-null"));
  }
  auto* sink = new (std::nothrow) hyle_obs_sink(write, flush, state);
  if (sink == nullptr) {
    return steal_status(
        hyle::Status::ResourceExhausted("sink allocation failed"));
  }
  hyle::Status status = hyle::obs::add_sink(sink->native);
  if (!status.ok()) {
    delete sink;
    return steal_status(std::move(status));
  }
  *out = sink;
  return hyle_status{};
}

hyle_status hyle_obs_sink_destroy(hyle_obs_sink* sink) {
  if (sink == nullptr) return hyle_status{};
  hyle::Status status = hyle::obs::remove_sink(sink->native);
  if (!status.ok()) {
    return steal_status(std::move(status));
  }
  delete sink;
  return hyle_status{};
}

size_t hyle_obs_sink_count(void) {
  return hyle::obs::sink_count();
}

size_t hyle_obs_sink_capacity(void) {
  return hyle::obs::sink_capacity();
}

size_t hyle_obs_flush_sinks(void) { return hyle::obs::flush_sinks(); }

int hyle_obs_default_sink_enabled(void) {
  return hyle::obs::default_sink_enabled() ? 1 : 0;
}

hyle_status hyle_obs_set_default_sink_enabled(int enabled) {
  return steal_status(hyle::obs::set_default_sink_enabled(enabled != 0));
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
