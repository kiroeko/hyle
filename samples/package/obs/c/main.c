#define HYLE_OBS_COMPILED_MIN_LEVEL 2
#include <hyle/c/obs.h>

#include <string.h>

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

  capture out = {0, 0, 0};
  hyle_obs_sink* sink = NULL;
  if (!status_ok_and_free(
          hyle_obs_sink_create(&sink, capture_write, capture_flush, &out))) {
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
    (void)hyle_obs_sink_destroy(sink);
    return 4;
  }

  if (!status_ok_and_free(hyle_obs_sink_destroy(sink))) return 5;
  if (!status_ok_and_free(hyle_obs_set_default_sink_enabled(1))) return 6;
  return out.writes == 1 && out.flushes == 1 && out.matched ? 0 : 7;
}
