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

struct capture {
  size_t calls;
  hyle_obs_log_level level;
  size_t fields;
  uint32_t line;
  uint64_t timestamp_ns;
  char message[64];
};

static void capture_write(void* state, const hyle_obs_log_record* record) {
  struct capture* capture = (struct capture*)state;
  ++capture->calls;
  capture->level = record->level;
  capture->fields = record->field_count;
  capture->line = record->line;
  capture->timestamp_ns = record->timestamp_ns;
  (void)snprintf(capture->message, sizeof(capture->message), "%s",
                 record->message);
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
  CHECK(hyle_obs_sink_destroy(self->sink));
  self->sink = NULL;
}

struct cross_remove {
  hyle_obs_sink* target;
  int result;
};

static void cross_remove_write(void* state,
                               const hyle_obs_log_record* record) {
  struct cross_remove* cross = (struct cross_remove*)state;
  (void)record;
  cross->result = hyle_obs_sink_destroy(cross->target);
}

int main(void) {
  CHECK(hyle_obs_set_default_sink_enabled(0));
  CHECK(!hyle_obs_default_sink_enabled());
  CHECK(hyle_obs_sink_capacity() >= 1);
  CHECK(hyle_obs_sink_count() == 0);

  CHECK(!hyle_obs_set_level((hyle_obs_log_level)99));
  CHECK(strstr(hyle_obs_last_error(), "threshold") != NULL);
  CHECK(hyle_obs_set_level(HYLE_OBS_LOG_TRACE));
  CHECK(strcmp(hyle_obs_last_error(), "") == 0);
  CHECK(hyle_obs_get_level() == HYLE_OBS_LOG_TRACE);
  CHECK(hyle_obs_should_log(HYLE_OBS_LOG_DEBUG));
  CHECK(!hyle_obs_should_log(HYLE_OBS_LOG_OFF));
  CHECK(strstr(hyle_obs_last_error(), "record level") != NULL);
  CHECK(hyle_obs_get_level() == HYLE_OBS_LOG_TRACE);
  CHECK(strcmp(hyle_obs_last_error(), "") == 0);

  CHECK(hyle_obs_sink_create(NULL, NULL) == NULL);
  CHECK(strstr(hyle_obs_last_error(), "callback") != NULL);

  {
    struct capture capture = {0};
    hyle_obs_sink* sink = hyle_obs_sink_create(capture_write, &capture);
    CHECK(sink != NULL);
    CHECK(hyle_obs_sink_count() == 1);
    CHECK(hyle_obs_log_info("hello"));
    CHECK(hyle_obs_log(HYLE_OBS_LOG_INFO, "direct"));
    CHECK(hyle_obs_log_trace("trace"));
    CHECK(hyle_obs_log_debug("debug"));
    CHECK(hyle_obs_log_warn("warn"));
    CHECK(hyle_obs_log_error("error"));
    CHECK(capture.calls == 6);
    CHECK(strcmp(capture.message, "error") == 0);

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
      CHECK(hyle_obs_emit_record(&record));
      CHECK(capture.level == HYLE_OBS_LOG_WARN);
      CHECK(capture.fields == 2);
      CHECK(capture.line == 42);
      CHECK(capture.timestamp_ns == UINT64_C(123));
    }

    {
      hyle_obs_log_record bad = HYLE_OBS_LOG_RECORD_INIT;
      bad.struct_size = 1;
      CHECK(!hyle_obs_emit_record(&bad));
      CHECK(strstr(hyle_obs_last_error(), "struct_size") != NULL);
      bad = (hyle_obs_log_record)HYLE_OBS_LOG_RECORD_INIT;
      bad.message = NULL;
      CHECK(!hyle_obs_emit_record(&bad));
      CHECK(strstr(hyle_obs_last_error(), "null message") != NULL);
    }

    CHECK(hyle_obs_set_level(HYLE_OBS_LOG_OFF));
    CHECK(!hyle_obs_log_error("gated"));
    CHECK(capture.calls == 7);
    CHECK(hyle_obs_sink_destroy(sink));
    CHECK(hyle_obs_sink_destroy(NULL));
    CHECK(hyle_obs_sink_count() == 0);
  }

  /* A sink may unregister/destroy its own opaque handle in its callback. */
  {
    struct self_remove self = {0};
    CHECK(hyle_obs_set_level(HYLE_OBS_LOG_INFO));
    self.sink = hyle_obs_sink_create(self_remove_write, &self);
    CHECK(self.sink != NULL);
    CHECK(hyle_obs_log_info("self-remove"));
    CHECK(self.calls == 1);
    CHECK(self.sink == NULL);
    CHECK(hyle_obs_sink_count() == 0);
  }

  /* Cross-destroy is rejected without freeing the registered target. */
  {
    struct cross_remove cross = {0};
    struct capture target = {0};
    hyle_obs_sink* remover =
        hyle_obs_sink_create(cross_remove_write, &cross);
    CHECK(remover != NULL);
    cross.target = hyle_obs_sink_create(capture_write, &target);
    CHECK(cross.target != NULL);
    CHECK(hyle_obs_log_info("cross-remove"));
    CHECK(!cross.result);
    CHECK(strstr(hyle_obs_last_error(), "cross-removal") != NULL);
    CHECK(target.calls == 1);
    CHECK(hyle_obs_sink_destroy(remover));
    CHECK(hyle_obs_sink_destroy(cross.target));
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

  CHECK(hyle_obs_set_default_sink_enabled(1));
  CHECK(hyle_obs_default_sink_enabled());
  CHECK(hyle_obs_set_default_sink_enabled(0));

  if (g_fails != 0) {
    fprintf(stderr, "obs_c_test: %d failure(s)\n", g_fails);
    return 1;
  }
  puts("obs_c_test: all checks passed");
  return 0;
}
