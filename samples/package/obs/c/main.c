#define HYLE_OBS_COMPILED_MIN_LEVEL 2
#include <hyle/c/obs.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
typedef char installed_log_level_must_be_u32[
    sizeof(hyle_obs_log_level) == sizeof(uint32_t) ? 1 : -1];
#else
_Static_assert(sizeof(hyle_obs_log_level) == sizeof(uint32_t),
               "installed log level ABI must stay fixed-width");
#endif

typedef struct capture {
  int writes;
  int flushes;
  int matched;
} capture;

static void capture_write(void* state, const hyle_obs_log_record* record) {
  capture* out = (capture*)state;
  ++out->writes;
  if (record->level == HYLE_OBS_LOG_WARN &&
      record->message_size == strlen("queue delayed") &&
      memcmp(record->message, "queue delayed", record->message_size) == 0 &&
      record->field_count == 1 &&
      strcmp(record->fields[0].key, "device") == 0 &&
      strcmp(record->fields[0].value, "gpu0") == 0) {
    out->matched = 1;
  }
}

static void capture_flush(void* state) {
  ++((capture*)state)->flushes;
}

static int status_ok_and_free(hyle_status status) {
  const int ok = hyle_status_is_ok(status);
  hyle_status_free(&status);
  return ok;
}

int main(void) {
  int evaluated = 0;
  hyle_status compiled_out =
      HYLE_OBS_EMIT_DEBUG((++evaluated, "compiled out"));
  if (evaluated != 0 || !status_ok_and_free(compiled_out)) return 1;

  if (!status_ok_and_free(hyle_obs_set_default_sink_enabled(0))) return 1;
  if (!status_ok_and_free(hyle_obs_set_level(HYLE_OBS_LOG_TRACE))) return 2;
  hyle_obs_trace_fiber_enter("package-c-fiber");
  hyle_obs_trace_fiber_leave();

  capture out = {0, 0, 0};
  hyle_obs_sink* sink = NULL;
  const size_t control_size = hyle_obs_sink_control_size();
  const size_t control_alignment = hyle_obs_sink_control_alignment();
  if (control_alignment == 0 ||
      (control_alignment & (control_alignment - 1)) != 0 ||
      control_size > SIZE_MAX - (control_alignment - 1)) {
    return 3;
  }
  void* control_storage =
      malloc(control_size + control_alignment - 1);
  if (control_storage == NULL) return 3;
  const uintptr_t aligned_address =
      ((uintptr_t)control_storage + control_alignment - 1) &
      ~(uintptr_t)(control_alignment - 1);
  if (!status_ok_and_free(hyle_obs_sink_init(
          &sink, (void*)aligned_address, control_size, capture_write,
          capture_flush, &out))) {
    free(control_storage);
    return 3;
  }

  hyle_obs_log_field field = {"device", "gpu0"};
  hyle_obs_log_record record = HYLE_OBS_LOG_RECORD_INIT;
  record.level = HYLE_OBS_LOG_WARN;
  record.message = "queue delayed";
  record.fields = &field;
  record.field_count = 1;
  int emitted = 0;
  if (!status_ok_and_free(hyle_obs_emit_record(&emitted, &record)) ||
      !emitted || hyle_obs_flush_sinks() != 1) {
    if (status_ok_and_free(hyle_obs_sink_destroy(sink))) {
      free(control_storage);
    }
    return 4;
  }

  if (!status_ok_and_free(hyle_obs_sink_destroy(sink))) {
    free(control_storage);
    return 5;
  }
  free(control_storage);
  if (!status_ok_and_free(hyle_obs_set_default_sink_enabled(1))) return 6;
  return out.writes == 1 && out.flushes == 1 && out.matched ? 0 : 7;
}
