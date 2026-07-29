/* Pure C consumer of include/hyle/c/obs.h. */
#include "hyle/c/obs.h"

#include <stdio.h>
#include <string.h>

static int g_fails = 0;
#define CHECK(c)                                                              \
  do {                                                                        \
    if (!(c)) {                                                               \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #c, __FILE__, __LINE__); \
      ++g_fails;                                                              \
    }                                                                         \
  } while (0)

static int status_ok(hyle_status status) {
  int ok = hyle_status_is_ok(status);
  if (!ok) {
    fprintf(stderr, "unexpected status: %s\n",
            hyle_status_message_cstr(status));
  }
  hyle_status_free(&status);
  return ok;
}

static hyle_status_code take_status_code(hyle_status status) {
  hyle_status_code code = hyle_status_get_code(status);
  hyle_status_free(&status);
  return code;
}

#define CHECK_OK(expr) CHECK(status_ok((expr)))

struct capture {
  size_t calls;
  size_t flushes;
  hyle_obs_log_level level;
  size_t fields;
  uint32_t line;
  uint64_t timestamp_ns;
  size_t message_size;
  char message[64];
};

static void capture_flush(void* state) {
  ++((struct capture*)state)->flushes;
}

static void capture_write(void* state, const hyle_obs_log_record* record) {
  struct capture* capture = (struct capture*)state;
  ++capture->calls;
  capture->level = record->level;
  capture->fields = record->field_count;
  capture->line = record->line;
  capture->timestamp_ns = record->timestamp_ns;
  capture->message_size = record->message_size;
  {
    const size_t copy = record->message_size < sizeof(capture->message) - 1
                            ? record->message_size
                            : sizeof(capture->message) - 1;
    if (copy != 0) memcpy(capture->message, record->message, copy);
    capture->message[copy] = '\0';
  }
}

struct self_remove {
  hyle_obs_sink* sink;
  size_t calls;
};

static void self_remove_write(void* state,
                              const hyle_obs_log_record* record) {
  struct self_remove* self = (struct self_remove*)state;
  (void)record;
  ++self->calls;
  CHECK_OK(hyle_obs_sink_destroy(self->sink));
  self->sink = NULL;
}

struct cross_remove {
  hyle_obs_sink* target;
  hyle_status_code result;
};

static void cross_remove_write(void* state,
                               const hyle_obs_log_record* record) {
  struct cross_remove* cross = (struct cross_remove*)state;
  (void)record;
  cross->result = take_status_code(hyle_obs_sink_destroy(cross->target));
}

int main(void) {
  (void)"HYLE_C_OBS_COVERAGE_BODY_MARKER";
  CHECK_OK(hyle_obs_set_default_sink_enabled(0));
  CHECK(!hyle_obs_default_sink_enabled());
  CHECK(hyle_obs_sink_capacity() >= 1);
  CHECK(hyle_obs_sink_count() == 0);

  CHECK(take_status_code(hyle_obs_set_level((hyle_obs_log_level)99)) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK_OK(hyle_obs_set_level(HYLE_OBS_LOG_TRACE));
  CHECK(hyle_obs_get_level() == HYLE_OBS_LOG_TRACE);
  CHECK(hyle_obs_should_log(HYLE_OBS_LOG_DEBUG));
  CHECK(!hyle_obs_should_log(HYLE_OBS_LOG_OFF));
  CHECK(hyle_obs_get_level() == HYLE_OBS_LOG_TRACE);

  {
    hyle_obs_sink* invalid = (hyle_obs_sink*)(uintptr_t)1;
    CHECK(take_status_code(
              hyle_obs_sink_create(&invalid, NULL, NULL, NULL)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(invalid == NULL);
  }

  {
    struct capture capture = {0};
    hyle_obs_sink* sink = NULL;
    CHECK_OK(hyle_obs_sink_create(&sink, capture_write, capture_flush,
                                  &capture));
    CHECK(sink != NULL);
    CHECK(hyle_obs_sink_count() == 1);
    CHECK_OK(hyle_obs_log_info("hello"));
    {
      int emitted = 0;
      CHECK_OK(hyle_obs_log(&emitted, HYLE_OBS_LOG_INFO, "direct"));
      CHECK(emitted);
    }
    CHECK_OK(hyle_obs_log_trace("trace"));
    CHECK_OK(hyle_obs_log_debug("debug"));
    CHECK_OK(hyle_obs_log_warn("warn"));
    CHECK_OK(hyle_obs_log_error("error"));
    CHECK(capture.calls == 6);
    CHECK(strcmp(capture.message, "error") == 0);
    {
      const char bytes[] = {'a', '\0', 'b'};
      CHECK_OK(hyle_obs_log_n(NULL, HYLE_OBS_LOG_INFO, bytes, sizeof(bytes)));
      CHECK(capture.message_size == sizeof(bytes));
      CHECK(memcmp(capture.message, bytes, sizeof(bytes)) == 0);
    }
    CHECK(capture.calls == 7);
    CHECK(hyle_obs_flush_sinks() == 1);
    CHECK(capture.flushes == 1);

    {
      hyle_obs_log_field fields[] = {
          {"request_id", "abc"},
          {"peer", "127.0.0.1"},
      };
      hyle_obs_log_record record = HYLE_OBS_LOG_RECORD_INIT;
      record.level = HYLE_OBS_LOG_WARN;
      record.message = "slow";
      record.fields = fields;
      record.field_count = 2;
      record.file = "server.c";
      record.function = "serve";
      record.line = 42;
      record.timestamp_ns = UINT64_C(123);
      int emitted = 0;
      CHECK_OK(hyle_obs_emit_record(&emitted, &record));
      CHECK(emitted);
      CHECK(capture.level == HYLE_OBS_LOG_WARN);
      CHECK(capture.fields == 2);
      CHECK(capture.line == 42);
      CHECK(capture.timestamp_ns == UINT64_C(123));
    }

    {
      hyle_obs_log_record bad = HYLE_OBS_LOG_RECORD_INIT;
      bad.struct_size = 1;
      CHECK(take_status_code(hyle_obs_emit_record(NULL, &bad)) ==
            HYLE_STATUS_INVALID_ARGUMENT);
      bad = (hyle_obs_log_record)HYLE_OBS_LOG_RECORD_INIT;
      bad.message = NULL;
      CHECK(take_status_code(hyle_obs_emit_record(NULL, &bad)) ==
            HYLE_STATUS_INVALID_ARGUMENT);
      bad.message_size = 0;
      CHECK_OK(hyle_obs_emit_record(NULL, &bad));
    }

    CHECK_OK(hyle_obs_set_level(HYLE_OBS_LOG_OFF));
    {
      int emitted = 1;
      CHECK_OK(hyle_obs_log(&emitted, HYLE_OBS_LOG_ERROR, "gated"));
      CHECK(!emitted);
    }
    CHECK(capture.calls == 9);
    CHECK_OK(hyle_obs_sink_destroy(sink));
    CHECK_OK(hyle_obs_sink_destroy(NULL));
    CHECK(hyle_obs_sink_count() == 0);
  }

  /* A sink may unregister/destroy its own opaque handle in its callback. */
  {
    struct self_remove self = {0};
    CHECK_OK(hyle_obs_set_level(HYLE_OBS_LOG_INFO));
    CHECK_OK(hyle_obs_sink_create(&self.sink, self_remove_write, NULL, &self));
    CHECK(self.sink != NULL);
    CHECK_OK(hyle_obs_log_info("self-remove"));
    CHECK(self.calls == 1);
    CHECK(self.sink == NULL);
    CHECK(hyle_obs_sink_count() == 0);
  }

  /* Cross-destroy is rejected without freeing the registered target. */
  {
    struct cross_remove cross = {0};
    struct capture target = {0};
    hyle_obs_sink* remover = NULL;
    CHECK_OK(hyle_obs_sink_create(&remover, cross_remove_write, NULL, &cross));
    CHECK(remover != NULL);
    CHECK_OK(hyle_obs_sink_create(&cross.target, capture_write, NULL, &target));
    CHECK(cross.target != NULL);
    CHECK_OK(hyle_obs_log_info("cross-remove"));
    CHECK(cross.result == HYLE_STATUS_FAILED_PRECONDITION);
    CHECK(target.calls == 1);
    CHECK_OK(hyle_obs_sink_destroy(remover));
    CHECK_OK(hyle_obs_sink_destroy(cross.target));
    CHECK(hyle_obs_sink_count() == 0);
  }

  /* Runtime trace calls are stable no-ops in the default non-tracing build. */
  {
    int value = 7;
    hyle_obs_trace_zone zone = HYLE_OBS_TRACE_ZONE_BEGIN("c-zone");
    hyle_obs_trace_zone direct_zone =
        hyle_obs_trace_zone_begin("direct", __FILE__, __func__, __LINE__);
    hyle_obs_trace_zone_text(zone, "work");
    hyle_obs_trace_plot("value", 1.5);
    hyle_obs_trace_counter("count", 2);
    hyle_obs_trace_message("message");
    hyle_obs_trace_frame(NULL);
    hyle_obs_trace_alloc(&value, sizeof(value), "test");
    hyle_obs_trace_free(&value, "test");
    hyle_obs_trace_thread_name("hyle-obs-test");
    hyle_obs_trace_zone_end(zone);
    hyle_obs_trace_zone_end(direct_zone);
    CHECK(hyle_obs_trace_enabled() == 0 ||
          hyle_obs_trace_enabled() == 1);
  }

  CHECK_OK(hyle_obs_set_default_sink_enabled(1));
  CHECK(hyle_obs_default_sink_enabled());
  CHECK_OK(hyle_obs_set_default_sink_enabled(0));

  if (g_fails != 0) {
    fprintf(stderr, "obs_c_test: %d failure(s)\n", g_fails);
    return 1;
  }
  puts("obs_c_test: all checks passed");
  return 0;
}
