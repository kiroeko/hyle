/* C-syntax compiler/schema/link fixture for include/hyle/c/obs.h. */
#include "hyle/c/error_helpers.h"
#include "hyle/c/obs.h"

#include <stdio.h>
#include <string.h>

_Static_assert(sizeof(hyle_obs_log_level) == sizeof(uint32_t),
               "log level must have a fixed C ABI width");
_Static_assert(offsetof(hyle_obs_log_record, level) == 4,
               "log record level ABI offset");
_Static_assert(offsetof(hyle_obs_log_record, message) == 8,
               "log record message ABI offset");
_Static_assert(offsetof(hyle_obs_log_record, message_size) == 16,
               "log record message size ABI offset");
_Static_assert(HYLE_OBS_LOG_RECORD_REQUIRED_SIZE == 24,
               "log record semantic required prefix");
_Static_assert(offsetof(hyle_obs_log_record, fields) == 24,
               "log record fields ABI offset");
_Static_assert(offsetof(hyle_obs_log_record, field_count) == 32,
               "log record field count ABI offset");
_Static_assert(offsetof(hyle_obs_log_record, file) == 40,
               "log record file ABI offset");
_Static_assert(offsetof(hyle_obs_log_record, function) == 48,
               "log record function ABI offset");
_Static_assert(offsetof(hyle_obs_log_record, line) == 56,
               "log record line ABI offset");
_Static_assert(offsetof(hyle_obs_log_record, timestamp_ns) == 64,
               "log record timestamp ABI offset");
_Static_assert(sizeof(hyle_obs_log_record) == 72,
               "log record 64-bit ABI size");

typedef struct obs_record_required_prefix {
  uint32_t struct_size;
  hyle_obs_log_level level;
  const char* message;
  size_t message_size;
} obs_record_required_prefix;

typedef struct obs_record_partial_fields_pair {
  obs_record_required_prefix required;
  const hyle_obs_log_field* fields;
  unsigned char field_count_fragment;
} obs_record_partial_fields_pair;

_Static_assert(sizeof(obs_record_required_prefix) ==
                   HYLE_OBS_LOG_RECORD_REQUIRED_SIZE,
               "physical log record required prefix size");
_Static_assert(offsetof(obs_record_partial_fields_pair, fields) ==
                   offsetof(hyle_obs_log_record, fields),
               "physical log record fields offset");
_Static_assert(offsetof(obs_record_partial_fields_pair,
                        field_count_fragment) ==
                   offsetof(hyle_obs_log_record, field_count),
               "physical log record field count fragment offset");

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
    char message[256];
    hyle_status_message(status, message, sizeof(message));
    fprintf(stderr, "unexpected status: %s\n", message);
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
  int fields_present;
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
  capture->fields_present = record->fields != NULL;
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
  size_t flushes;
};

static void self_remove_write(void* state,
                              const hyle_obs_log_record* record) {
  struct self_remove* self = (struct self_remove*)state;
  (void)record;
  ++self->calls;
  CHECK_OK(hyle_obs_sink_destroy(self->sink));
  self->sink = NULL;
}

static void self_remove_flush(void* state) {
  struct self_remove* self = (struct self_remove*)state;
  ++self->flushes;
  CHECK_OK(hyle_obs_sink_destroy(self->sink));
  self->sink = NULL;
}

struct cross_remove {
  hyle_obs_sink* target;
  hyle_status_code result;
};

enum { TEST_SINK_CONTROL_BYTES = 256 };
struct sink_control_storage {
  unsigned char bytes[TEST_SINK_CONTROL_BYTES];
};

static void* sink_control(struct sink_control_storage* storage,
                          size_t* out_bytes) {
  const size_t alignment = hyle_obs_sink_control_alignment();
  const uintptr_t begin = (uintptr_t)storage->bytes;
  const uintptr_t aligned = (begin + alignment - 1) & ~(uintptr_t)(alignment - 1);
  const size_t skipped = (size_t)(aligned - begin);
  *out_bytes = sizeof(storage->bytes) - skipped;
  return (void*)aligned;
}

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

    invalid = (hyle_obs_sink*)(uintptr_t)1;
    CHECK(take_status_code(hyle_obs_sink_init(
              &invalid, NULL, 0, capture_write, NULL, NULL)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(invalid == NULL);
  }

  {
    struct capture capture = {0};
    struct sink_control_storage control_storage;
    size_t control_bytes = 0;
    void* control = sink_control(&control_storage, &control_bytes);
    hyle_obs_sink* sink = NULL;
    CHECK(hyle_obs_sink_control_size() <= control_bytes);
    CHECK(hyle_obs_sink_control_alignment() != 0);
    CHECK((hyle_obs_sink_control_alignment() &
           (hyle_obs_sink_control_alignment() - 1)) == 0);
    CHECK_OK(hyle_obs_sink_init(&sink, control, control_bytes, capture_write,
                                capture_flush, &capture));
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
      CHECK(capture.fields_present);
      CHECK(capture.line == 42);
      CHECK(capture.timestamp_ns == UINT64_C(123));
    }

    /* The semantic minimum is a real, physically shorter record object. All
     * missing optional fields come from the current local defaults. */
    {
      obs_record_required_prefix prefix = {
          (uint32_t)sizeof(prefix), HYLE_OBS_LOG_INFO, "short", 5};
      int emitted = 0;
      CHECK_OK(hyle_obs_emit_record(&emitted, &prefix));
      CHECK(emitted);
      CHECK(strcmp(capture.message, "short") == 0);
      CHECK(capture.fields == 0);
      CHECK(!capture.fields_present);
      CHECK(capture.line == 0);
      CHECK(capture.timestamp_ns == 0);
    }

    /* fields+field_count is one extension unit. A declared prefix containing
     * all of fields but only one byte of count ignores both members. */
    {
      hyle_obs_log_field ignored = {"ignored", "ignored"};
      obs_record_partial_fields_pair partial;
      memset(&partial, 0, sizeof(partial));
      partial.required.struct_size =
          (uint32_t)(offsetof(obs_record_partial_fields_pair,
                             field_count_fragment) +
                     1);
      partial.required.level = HYLE_OBS_LOG_INFO;
      partial.required.message = "pair";
      partial.required.message_size = 4;
      partial.fields = &ignored;
      partial.field_count_fragment = 1;
      CHECK_OK(hyle_obs_emit_record(NULL, &partial));
      CHECK(capture.fields == 0);
      CHECK(!capture.fields_present);
    }

    /* A size ending inside an individual optional member ignores that member
     * instead of manufacturing a value from partial bytes. */
    {
      hyle_obs_log_record partial = HYLE_OBS_LOG_RECORD_INIT;
      partial.message = "partial-time";
      partial.message_size = 12;
      partial.struct_size =
          (uint32_t)(offsetof(hyle_obs_log_record, timestamp_ns) + 4);
      partial.timestamp_ns = UINT64_C(0xA5A5A5A5A5A5A5A5);
      CHECK_OK(hyle_obs_emit_record(NULL, &partial));
      CHECK(capture.timestamp_ns == 0);
    }

    /* Current readers ignore bytes appended by a future record version. */
    {
      struct future_obs_record {
        hyle_obs_log_record known;
        unsigned char tail[16];
      } future;
      memset(&future, 0xA5, sizeof(future));
      future.known = (hyle_obs_log_record)HYLE_OBS_LOG_RECORD_INIT;
      future.known.struct_size = (uint32_t)sizeof(future);
      future.known.message = "future";
      future.known.message_size = 6;
      future.known.line = 77;
      CHECK_OK(hyle_obs_emit_record(NULL, &future.known));
      CHECK(strcmp(capture.message, "future") == 0);
      CHECK(capture.line == 77);
    }

    /* Every input member, including struct_size, is byte-read from an
     * unaligned FFI buffer into the aligned local record. */
    {
      hyle_obs_log_record known = HYLE_OBS_LOG_RECORD_INIT;
      union {
        hyle_obs_log_record alignment;
        unsigned char bytes[sizeof(hyle_obs_log_record) + 1];
      } raw;
      int emitted = 0;
      known.message = "unaligned";
      known.message_size = 9;
      known.timestamp_ns = UINT64_C(456);
      memcpy(raw.bytes + 1, &known, sizeof(known));
      CHECK_OK(hyle_obs_emit_record(&emitted, raw.bytes + 1));
      CHECK(emitted);
      CHECK(strcmp(capture.message, "unaligned") == 0);
      CHECK(capture.timestamp_ns == UINT64_C(456));
    }

    {
      uint32_t physical_too_short = (uint32_t)sizeof(physical_too_short);
      hyle_obs_log_record bad = HYLE_OBS_LOG_RECORD_INIT;
      CHECK(take_status_code(hyle_obs_emit_record(
                NULL, &physical_too_short)) ==
            HYLE_STATUS_INVALID_ARGUMENT);
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
    CHECK(capture.calls == 14);
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

  /* Caller-provided control storage has the same self-removal guarantee. Once
   * the dispatch call returns and retires the old slot, the ended control
   * object's storage can be initialized again, including after flush removal. */
  {
    struct sink_control_storage control_storage;
    struct self_remove self = {0};
    size_t control_bytes = 0;
    void* control = sink_control(&control_storage, &control_bytes);

    CHECK_OK(hyle_obs_sink_init(&self.sink, control, control_bytes,
                                self_remove_write, NULL, &self));
    CHECK(self.sink != NULL);
    CHECK_OK(hyle_obs_log_info("caller-storage-write-remove"));
    CHECK(self.calls == 1);
    CHECK(self.sink == NULL);
    CHECK(hyle_obs_sink_count() == 0);

    CHECK_OK(hyle_obs_sink_init(&self.sink, control, control_bytes,
                                self_remove_write, self_remove_flush, &self));
    CHECK(self.sink != NULL);
    CHECK(hyle_obs_flush_sinks() == 1);
    CHECK(self.flushes == 1);
    CHECK(self.sink == NULL);
    CHECK(hyle_obs_sink_count() == 0);

    CHECK_OK(hyle_obs_sink_init(&self.sink, control, control_bytes,
                                self_remove_write, NULL, &self));
    CHECK(self.sink != NULL);
    CHECK_OK(hyle_obs_sink_destroy(self.sink));
    self.sink = NULL;
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
    static char oversized_trace_text[HYLE_OBS_TRACE_TEXT_MAX + 257u];
    int value = 7;
    hyle_obs_trace_zone zone = HYLE_OBS_TRACE_ZONE_BEGIN("c-zone");
    hyle_obs_trace_zone direct_zone =
        hyle_obs_trace_zone_begin("direct", __FILE__, __func__, __LINE__);
    hyle_obs_trace_zone_text(zone, "work");
    hyle_obs_trace_plot("value", 1.5);
    hyle_obs_trace_counter("count", 2);
    hyle_obs_trace_message("message");
    hyle_obs_trace_fiber_enter("c-fiber-stable");
    hyle_obs_trace_fiber_leave();
    hyle_obs_trace_fiber_enter(NULL);
    hyle_obs_trace_frame(NULL);
    hyle_obs_trace_alloc(&value, sizeof(value), "test");
    hyle_obs_trace_free(&value, "test");
    hyle_obs_trace_thread_name("hyle-obs-test");
    hyle_obs_trace_zone_end(zone);
    hyle_obs_trace_zone_end(direct_zone);
    memset(oversized_trace_text, 'x', sizeof(oversized_trace_text) - 1);
    oversized_trace_text[sizeof(oversized_trace_text) - 1] = '\0';
    zone = hyle_obs_trace_zone_begin(
        oversized_trace_text, oversized_trace_text, oversized_trace_text, 1);
    hyle_obs_trace_zone_text(zone, oversized_trace_text);
    hyle_obs_trace_message(oversized_trace_text);
    hyle_obs_trace_zone_end(zone);
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
