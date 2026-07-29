/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 *
 * Pure C ABI for HYLE structured logging and explicit trace emission.
 * Native zero-cost trace macros remain in hyle/obs/trace.h.
 * Docs (1:1 with this header): docs/guide/c-obs.md
 */
#ifndef HYLE_C_OBS_H_
#define HYLE_C_OBS_H_

#include <stddef.h>
#include <stdint.h>

#include "hyle/c/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#  define HYLE_OBS_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_OBS_BUILD)
#    define HYLE_OBS_C_API __declspec(dllexport)
#  else
#    define HYLE_OBS_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_OBS_C_API __attribute__((visibility("default")))
#else
#  define HYLE_OBS_C_API
#endif

typedef enum hyle_obs_log_level {
  HYLE_OBS_LOG_TRACE = 0,
  HYLE_OBS_LOG_DEBUG = 1,
  HYLE_OBS_LOG_INFO = 2,
  HYLE_OBS_LOG_WARN = 3,
  HYLE_OBS_LOG_ERROR = 4,
  HYLE_OBS_LOG_OFF = 5
} hyle_obs_log_level;

/* Consumer-side compile-time floor for logging convenience macros
 * (0=TRACE ... 4=ERROR, 5=OFF). This is call-site policy, not library ABI
 * state; direct hyle_obs_log/emit_record calls retain runtime filtering. */
#ifndef HYLE_OBS_COMPILED_MIN_LEVEL
#  define HYLE_OBS_COMPILED_MIN_LEVEL 0
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL < 0 || HYLE_OBS_COMPILED_MIN_LEVEL > 5
#  error "HYLE_OBS_COMPILED_MIN_LEVEL must be in [0, 5]"
#endif

typedef struct hyle_obs_log_field {
  const char* key;
  const char* value;
} hyle_obs_log_field;

/* Borrowed for one synchronous emit/callback. message_size is an exact byte
 * count and may preserve embedded NUL; HYLE_OBS_NUL_TERMINATED asks the checked
 * emit adapter to call strlen. timestamp_ns is optional Unix time; 0 means
 * absent. field/file/function strings remain NUL-terminated UTF-8. */
#define HYLE_OBS_NUL_TERMINATED SIZE_MAX
typedef struct hyle_obs_log_record {
  uint32_t struct_size;
  hyle_obs_log_level level;
  const char* message;
  size_t message_size;
  const hyle_obs_log_field* fields;
  size_t field_count;
  const char* file;
  const char* function;
  uint32_t line;
  uint64_t timestamp_ns;
} hyle_obs_log_record;

#define HYLE_OBS_LOG_RECORD_INIT                                             \
  { sizeof(hyle_obs_log_record), HYLE_OBS_LOG_INFO, "",                    \
    HYLE_OBS_NUL_TERMINATED, NULL, 0, NULL, NULL, 0, 0 }

typedef struct hyle_obs_sink_o hyle_obs_sink;
typedef void (*hyle_obs_sink_write_fn)(void* state,
                                       const hyle_obs_log_record* record);
typedef void (*hyle_obs_sink_flush_fn)(void* state);

/* Recoverable failures use the shared owning hyle_status contract. Output
 * parameters are cleared before work; gated/no-sink delivery is OK with
 * *out_emitted == 0, not an error. */
HYLE_OBS_C_API hyle_status hyle_obs_set_level(hyle_obs_log_level level);
HYLE_OBS_C_API hyle_obs_log_level hyle_obs_get_level(void);
HYLE_OBS_C_API int hyle_obs_should_log(hyle_obs_log_level level);
HYLE_OBS_C_API hyle_status hyle_obs_emit_record(
    int* out_emitted, const hyle_obs_log_record* record);
HYLE_OBS_C_API hyle_status hyle_obs_log(
    int* out_emitted, hyle_obs_log_level level, const char* message);
HYLE_OBS_C_API hyle_status hyle_obs_log_n(
    int* out_emitted, hyle_obs_log_level level, const char* message,
    size_t message_size);

/* create registers immediately. destroy unregisters quiescently and is NULL
 * safe. A successful return guarantees no callback on another thread still
 * uses state. A callback may destroy its OWN handle; destroying another sink
 * from a callback fails (it may be pending in the same dispatch snapshot). */
HYLE_OBS_C_API hyle_status hyle_obs_sink_create(
    hyle_obs_sink** out, hyle_obs_sink_write_fn write,
    hyle_obs_sink_flush_fn flush, void* state);
HYLE_OBS_C_API hyle_status hyle_obs_sink_destroy(hyle_obs_sink* sink);
HYLE_OBS_C_API size_t hyle_obs_sink_count(void);
HYLE_OBS_C_API size_t hyle_obs_sink_capacity(void);
HYLE_OBS_C_API size_t hyle_obs_flush_sinks(void);
HYLE_OBS_C_API int hyle_obs_default_sink_enabled(void);
HYLE_OBS_C_API hyle_status hyle_obs_set_default_sink_enabled(int enabled);

static inline hyle_status hyle_obs_log_trace(const char* message) {
  return hyle_obs_log(NULL, HYLE_OBS_LOG_TRACE, message);
}
static inline hyle_status hyle_obs_log_debug(const char* message) {
  return hyle_obs_log(NULL, HYLE_OBS_LOG_DEBUG, message);
}
static inline hyle_status hyle_obs_log_info(const char* message) {
  return hyle_obs_log(NULL, HYLE_OBS_LOG_INFO, message);
}
static inline hyle_status hyle_obs_log_warn(const char* message) {
  return hyle_obs_log(NULL, HYLE_OBS_LOG_WARN, message);
}
static inline hyle_status hyle_obs_log_error(const char* message) {
  return hyle_obs_log(NULL, HYLE_OBS_LOG_ERROR, message);
}
static inline hyle_status hyle_obs_log_trace_n(const char* message,
                                               size_t message_size) {
  return hyle_obs_log_n(NULL, HYLE_OBS_LOG_TRACE, message, message_size);
}
static inline hyle_status hyle_obs_log_debug_n(const char* message,
                                               size_t message_size) {
  return hyle_obs_log_n(NULL, HYLE_OBS_LOG_DEBUG, message, message_size);
}
static inline hyle_status hyle_obs_log_info_n(const char* message,
                                              size_t message_size) {
  return hyle_obs_log_n(NULL, HYLE_OBS_LOG_INFO, message, message_size);
}
static inline hyle_status hyle_obs_log_warn_n(const char* message,
                                              size_t message_size) {
  return hyle_obs_log_n(NULL, HYLE_OBS_LOG_WARN, message, message_size);
}
static inline hyle_status hyle_obs_log_error_n(const char* message,
                                               size_t message_size) {
  return hyle_obs_log_n(NULL, HYLE_OBS_LOG_ERROR, message, message_size);
}

/* These expression macros preserve the Status-returning C shape while
 * removing disabled arguments at preprocessing time. An ignored OK literal
 * optimizes away; direct functions remain available for dynamic levels. */
#ifdef __cplusplus
#  define HYLE_OBS_OK_LITERAL_ hyle_status{0}
#else
#  define HYLE_OBS_OK_LITERAL_ ((hyle_status){0})
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 0
#  define HYLE_OBS_EMIT_TRACE(message) hyle_obs_log_trace((message))
#else
#  define HYLE_OBS_EMIT_TRACE(message) HYLE_OBS_OK_LITERAL_
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 1
#  define HYLE_OBS_EMIT_DEBUG(message) hyle_obs_log_debug((message))
#else
#  define HYLE_OBS_EMIT_DEBUG(message) HYLE_OBS_OK_LITERAL_
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 2
#  define HYLE_OBS_EMIT_INFO(message) hyle_obs_log_info((message))
#else
#  define HYLE_OBS_EMIT_INFO(message) HYLE_OBS_OK_LITERAL_
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 3
#  define HYLE_OBS_EMIT_WARN(message) hyle_obs_log_warn((message))
#else
#  define HYLE_OBS_EMIT_WARN(message) HYLE_OBS_OK_LITERAL_
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 4
#  define HYLE_OBS_EMIT_ERROR(message) hyle_obs_log_error((message))
#else
#  define HYLE_OBS_EMIT_ERROR(message) HYLE_OBS_OK_LITERAL_
#endif

/* Runtime trace seam for C/FFI callers. These are no-ops when HYLE was built
 * without HYLE_TRACE. Native C++ hot paths should use hyle/obs/trace.h macros:
 * their disabled form has no function call or argument evaluation. */
typedef struct hyle_obs_trace_zone {
  uintptr_t rep[2];
} hyle_obs_trace_zone;

HYLE_OBS_C_API int hyle_obs_trace_enabled(void);
HYLE_OBS_C_API hyle_obs_trace_zone hyle_obs_trace_zone_begin(
    const char* name, const char* file, const char* function, uint32_t line);
HYLE_OBS_C_API void hyle_obs_trace_zone_end(hyle_obs_trace_zone zone);
HYLE_OBS_C_API void hyle_obs_trace_zone_text(
    hyle_obs_trace_zone zone, const char* text);
HYLE_OBS_C_API void hyle_obs_trace_plot(const char* name, double value);
HYLE_OBS_C_API void hyle_obs_trace_counter(const char* name, int64_t value);
HYLE_OBS_C_API void hyle_obs_trace_message(const char* message);
HYLE_OBS_C_API void hyle_obs_trace_frame(const char* stable_name);
HYLE_OBS_C_API void hyle_obs_trace_alloc(
    const void* pointer, size_t size, const char* stable_pool_name);
HYLE_OBS_C_API void hyle_obs_trace_free(
    const void* pointer, const char* stable_pool_name);
HYLE_OBS_C_API void hyle_obs_trace_thread_name(const char* name);

#define HYLE_OBS_TRACE_ZONE_BEGIN(name) \
  hyle_obs_trace_zone_begin((name), __FILE__, __func__, (uint32_t)__LINE__)

#ifdef __cplusplus
}
#endif

#endif /* HYLE_C_OBS_H_ */
