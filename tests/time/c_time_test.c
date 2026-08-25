/* C-syntax compiler/schema/link fixture for include/hyle/c/time.h. */
#include "hyle/c/time.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fails = 0;
#define CHECK(c)                                                             \
  do {                                                                       \
    if (!(c)) {                                                              \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #c, __FILE__, __LINE__); \
      ++g_fails;                                                             \
    }                                                                        \
  } while (0)

static int consume_ok(hyle_status status, const char* expression) {
  int ok = hyle_status_is_ok(status);
  if (!ok) {
    fprintf(stderr, "%s failed: %s\n", expression,
            hyle_status_message_cstr(status));
    ++g_fails;
  }
  hyle_status_free(&status);
  return ok;
}

#define CHECK_OK(expr) consume_ok((expr), #expr)

struct visit_state {
  size_t count;
  uintptr_t sum;
};

struct rearm_state {
  hyle_time_queue* queue;
  size_t visits;
};

enum pending_action {
  PENDING_CANCEL,
  PENDING_RESCHEDULE,
  PENDING_DESTROY,
  PENDING_REJECT_RESCHEDULE,
  PENDING_REJECT_RESCHEDULE_AFTER
};

struct pending_state {
  hyle_time_queue* queue;
  hyle_time_common_queue* common;
  hyle_time_timer* target;
  uint64_t reschedule_deadline;
  enum pending_action action;
  size_t visits;
  int cancelled;
};

struct destroy_source_state {
  hyle_time_queue* queue;
  hyle_time_common_queue* common;
  int rejected;
  int destroyed;
};

static void collect_timer(hyle_time_timer* timer, void* ctx) {
  struct visit_state* state = (struct visit_state*)ctx;
  ++state->count;
  state->sum += (uintptr_t)hyle_time_timer_user_data(timer);
  CHECK(!hyle_time_timer_is_scheduled(timer));
}

static void rearm_once(hyle_time_timer* timer, void* ctx) {
  struct rearm_state* state = (struct rearm_state*)ctx;
  CHECK(!hyle_time_timer_is_scheduled(timer));
  ++state->visits;
  if (state->visits == 1) {
    CHECK_OK(hyle_time_queue_schedule(state->queue, timer, UINT64_C(700001)));
  }
}

static void mutate_pending_timer(hyle_time_timer* timer, void* ctx) {
  struct pending_state* state = (struct pending_state*)ctx;
  ++state->visits;
  if (state->visits != 1) {
    CHECK(state->action == PENDING_REJECT_RESCHEDULE ||
          state->action == PENDING_REJECT_RESCHEDULE_AFTER);
    CHECK(timer == state->target);
    return;
  }
  CHECK(timer != state->target);
  CHECK(!hyle_time_timer_is_scheduled(state->target));

  if (state->action == PENDING_CANCEL) {
    if (state->common != NULL) {
      CHECK_OK(hyle_time_common_queue_cancel(&state->cancelled, state->common,
                                             state->target));
    } else {
      CHECK_OK(hyle_time_queue_cancel(&state->cancelled, state->queue,
                                      state->target));
    }
    return;
  }

  if (state->action == PENDING_RESCHEDULE) {
    if (state->common != NULL) {
      CHECK_OK(hyle_time_common_queue_schedule(state->common, state->target,
                                               state->reschedule_deadline));
    } else {
      CHECK_OK(hyle_time_queue_schedule(state->queue, state->target,
                                        state->reschedule_deadline));
    }
    return;
  }

  if (state->action == PENDING_REJECT_RESCHEDULE ||
      state->action == PENDING_REJECT_RESCHEDULE_AFTER) {
    hyle_status status =
        state->action == PENDING_REJECT_RESCHEDULE
            ? hyle_time_common_queue_schedule(state->common, state->target,
                                              state->reschedule_deadline)
            : hyle_time_common_queue_schedule_after(
                  state->common, state->target, state->reschedule_deadline);
    CHECK(hyle_status_is_invalid_argument(status));
    hyle_status_free(&status);
    return;
  }

  CHECK_OK(hyle_time_timer_destroy(state->target));
  state->target = NULL;
}

static void try_destroy_source(hyle_time_timer* timer, void* ctx) {
  struct destroy_source_state* state = (struct destroy_source_state*)ctx;
  hyle_status status;
  (void)timer;
  if (state->queue != NULL) {
    status = hyle_time_queue_destroy(state->queue);
  } else {
    status = hyle_time_common_queue_destroy(state->common);
  }
  state->rejected = hyle_status_is_failed_precondition(status);
  state->destroyed = hyle_status_is_ok(status);
  hyle_status_free(&status);
  if (state->destroyed) {
    state->queue = NULL;
    state->common = NULL;
  }
}

static void check_source_destroy_contract(void) {
  hyle_time_queue* queue = NULL;
  hyle_time_common_queue* common = NULL;
  hyle_time_timer* timer = NULL;
  size_t moved = 0;
  struct destroy_source_state state;

  CHECK_OK(hyle_time_queue_create_at(&queue, 0));
  CHECK_OK(hyle_time_timer_create(&timer, NULL));
  CHECK_OK(hyle_time_queue_schedule(queue, timer, 1));
  state = (struct destroy_source_state){queue, NULL, 0, 0};
  CHECK_OK(
      hyle_time_queue_advance(&moved, queue, 1, try_destroy_source, &state));
  CHECK(moved == 1 && state.rejected && !state.destroyed);
  if (state.queue != NULL) {
    CHECK(hyle_time_queue_empty(state.queue));
    CHECK_OK(hyle_time_queue_destroy(state.queue));
  }
  CHECK_OK(hyle_time_timer_destroy(timer));

  timer = NULL;
  CHECK_OK(hyle_time_common_queue_create(&common, 1));
  CHECK_OK(hyle_time_timer_create(&timer, NULL));
  CHECK_OK(hyle_time_common_queue_schedule(common, timer, 1));
  state = (struct destroy_source_state){NULL, common, 0, 0};
  CHECK_OK(hyle_time_common_queue_cancel_all(&moved, common, try_destroy_source,
                                             &state));
  CHECK(moved == 1 && state.rejected && !state.destroyed);
  if (state.common != NULL) {
    CHECK(hyle_time_common_queue_empty(state.common));
    CHECK_OK(hyle_time_common_queue_destroy(state.common));
  }
  CHECK_OK(hyle_time_timer_destroy(timer));
}

int main(void) {
  (void)"HYLE_C_TIME_COVERAGE_BODY_MARKER";
  uint64_t m0 = hyle_time_monotonic_ns();
  uint64_t r0 = hyle_time_realtime_ns();
  hyle_unix_timestamp timestamp = HYLE_UNIX_TIMESTAMP_INIT;
  hyle_time_realtime_timestamp(&timestamp);
  CHECK(m0 != 0);
  CHECK(r0 > UINT64_C(1577836800000000000)); /* 2020-01-01 */
  CHECK(hyle_unix_timestamp_is_valid(timestamp));
  CHECK(timestamp.seconds > INT64_C(1577836800)); /* 2020-01-01 */
  hyle_time_realtime_timestamp(NULL);
  CHECK(hyle_time_profiling_clock_ns() != 0);
  {
    const uint64_t start = hyle_time_profiling_measurement_start_ns();
    const uint64_t stop = hyle_time_profiling_measurement_stop_ns();
    CHECK(start != 0);
    CHECK(stop != 0);
  }
  CHECK(hyle_time_coarse_realtime_ns() != 0);
  CHECK(hyle_time_profiling_clock_available() == 0 ||
        hyle_time_profiling_clock_available() == 1);
  CHECK(hyle_time_coarse_realtime_available() == 0 ||
        hyle_time_coarse_realtime_available() == 1);
  CHECK(hyle_time_deadline_after_ns(UINT64_MAX) == UINT64_MAX);
  CHECK(hyle_time_saturating_add_ns(UINT64_MAX - 1, 2) == UINT64_MAX);
  CHECK(hyle_time_remaining_ns(0) == 0);
  hyle_time_sleep_for_ns(0);
  hyle_time_sleep_for_us(0);
  hyle_time_sleep_for_ms(0);
  hyle_time_sleep_until_ns(hyle_time_monotonic_ns());
  check_source_destroy_contract();
  hyle_time_queue* queue = NULL;
  hyle_time_common_queue* common = NULL;
  hyle_time_timer* one = NULL;
  hyle_time_timer* two = NULL;
  hyle_time_timer* three = NULL;
  CHECK_OK(hyle_time_queue_create_at(&queue, 0));
  {
    hyle_time_queue* live_clock_queue = NULL;
    CHECK_OK(hyle_time_queue_create(&live_clock_queue));
    CHECK_OK(hyle_time_queue_destroy(live_clock_queue));
  }
  CHECK_OK(hyle_time_common_queue_create(&common, UINT64_C(30)));
  CHECK_OK(hyle_time_timer_create(&one, (void*)(uintptr_t)1));
  CHECK_OK(hyle_time_timer_create(&two, (void*)(uintptr_t)2));
  CHECK_OK(hyle_time_timer_create(&three, (void*)(uintptr_t)3));
  CHECK(queue != NULL && common != NULL);
  CHECK(one != NULL && two != NULL && three != NULL);

  /* Libevent-shaped caller storage: one arena can hold many opaque timers
   * without one allocation per control and without exposing the C++ layout. */
  {
    const size_t timer_size = hyle_time_timer_storage_size();
    const size_t timer_alignment = hyle_time_timer_storage_alignment();
    const size_t timer_count = 3;
    unsigned char* raw = NULL;
    unsigned char* storage = NULL;
    hyle_time_timer* timers[3] = {NULL, NULL, NULL};
    CHECK(timer_size != 0);
    CHECK(timer_alignment != 0 &&
          (timer_alignment & (timer_alignment - 1)) == 0);
    CHECK(timer_size % timer_alignment == 0);
    raw =
        (unsigned char*)malloc(timer_size * timer_count + timer_alignment - 1);
    CHECK(raw != NULL);
    if (raw != NULL) {
      uintptr_t address = (uintptr_t)raw;
      address =
          (address + timer_alignment - 1) & ~(uintptr_t)(timer_alignment - 1);
      storage = (unsigned char*)address;
      for (size_t i = 0; i < timer_count; ++i) {
        CHECK_OK(hyle_time_timer_init(&timers[i], storage + i * timer_size,
                                      timer_size, (void*)(uintptr_t)(10 + i)));
        CHECK(timers[i] != NULL);
        CHECK_OK(hyle_time_queue_schedule(queue, timers[i], 100 + i));
      }
      {
        hyle_status status = hyle_time_timer_deinit(timers[0]);
        CHECK(hyle_status_is_failed_precondition(status));
        hyle_status_free(&status);
      }
      {
        struct visit_state bulk_state = {0, 0};
        size_t cancelled = 0;
        CHECK_OK(hyle_time_queue_cancel_all(&cancelled, queue, collect_timer,
                                            &bulk_state));
        CHECK(cancelled == timer_count);
        CHECK(bulk_state.count == timer_count);
        CHECK(bulk_state.sum == 33);
      }
      {
        hyle_status status = hyle_time_timer_destroy(timers[0]);
        CHECK(hyle_status_is_failed_precondition(status));
        hyle_status_free(&status);
      }
      for (size_t i = 0; i < timer_count; ++i) {
        CHECK_OK(hyle_time_timer_deinit(timers[i]));
      }
      CHECK_OK(hyle_time_timer_init(&timers[0], storage, timer_size,
                                    (void*)(uintptr_t)99));
      CHECK((uintptr_t)hyle_time_timer_user_data(timers[0]) == 99);
      CHECK_OK(hyle_time_timer_deinit(timers[0]));
    }
    {
      hyle_time_timer* invalid = (hyle_time_timer*)(uintptr_t)1;
      hyle_status status =
          hyle_time_timer_init(NULL, storage, timer_size, NULL);
      CHECK(hyle_status_is_invalid_argument(status));
      hyle_status_free(&status);
      status = hyle_time_timer_init(&invalid, NULL, timer_size, NULL);
      CHECK(hyle_status_is_invalid_argument(status));
      CHECK(invalid == NULL);
      hyle_status_free(&status);
      status = hyle_time_timer_init(&invalid, storage, timer_size - 1, NULL);
      CHECK(hyle_status_is_invalid_argument(status));
      CHECK(invalid == NULL);
      hyle_status_free(&status);
      if (storage != NULL && timer_alignment > 1) {
        status = hyle_time_timer_init(&invalid, storage + 1, timer_size, NULL);
        CHECK(hyle_status_is_invalid_argument(status));
        CHECK(invalid == NULL);
        hyle_status_free(&status);
      }
    }
    {
      hyle_status status = hyle_time_timer_deinit(one);
      CHECK(hyle_status_is_failed_precondition(status));
      hyle_status_free(&status);
    }
    free(raw);
  }

  CHECK(hyle_time_queue_empty(queue));
  CHECK_OK(hyle_time_queue_schedule(queue, one, UINT64_C(500000)));
  CHECK(hyle_time_timer_is_scheduled(one));
  CHECK(hyle_time_timer_deadline_ns(one) == UINT64_C(500000));
  CHECK(hyle_time_queue_size(queue) == 1);

  /* Cross-source ownership is an ordinary Status failure in C. */
  {
    int cancelled = 99;
    hyle_status status =
        hyle_time_common_queue_schedule(common, one, UINT64_C(10));
    CHECK(hyle_status_is_failed_precondition(status));
    CHECK(strstr(hyle_status_message_cstr(status), "another source") != NULL);
    hyle_status_free(&status);
    status = hyle_time_common_queue_cancel(&cancelled, common, one);
    CHECK(hyle_status_is_failed_precondition(status));
    CHECK(cancelled == 0);
    hyle_status_free(&status);
  }

  struct visit_state state = {0, 0};
  size_t moved = 99;
  CHECK_OK(hyle_time_queue_advance(&moved, queue, UINT64_C(400000),
                                   collect_timer, &state));
  CHECK(moved == 0);
  CHECK_OK(hyle_time_queue_advance(&moved, queue, UINT64_C(500000),
                                   collect_timer, &state));
  CHECK(moved == 1);
  CHECK(state.count == 1 && state.sum == 1);
  CHECK(hyle_time_queue_empty(queue));

  hyle_time_timer_set_user_data(one, (void*)(uintptr_t)4);
  CHECK((uintptr_t)hyle_time_timer_user_data(one) == 4);
  CHECK_OK(hyle_time_queue_schedule(queue, one, UINT64_C(900000)));
  {
    int cancelled = -1;
    CHECK_OK(hyle_time_queue_cancel(&cancelled, queue, one));
    CHECK(cancelled == 1);
    CHECK_OK(hyle_time_queue_cancel(&cancelled, queue, one));
    CHECK(cancelled == 0);
  }
  CHECK(!hyle_time_timer_is_scheduled(one));
  CHECK_OK(hyle_time_queue_schedule_after_at(queue, one, 5, 100));
  CHECK(hyle_time_timer_deadline_ns(one) == 105);
  CHECK_OK(hyle_time_queue_cancel(NULL, queue, one));
  CHECK_OK(hyle_time_queue_schedule_after(queue, one, 1));
  CHECK_OK(hyle_time_queue_cancel(NULL, queue, one));

  /* advance releases ownership before visiting, so callback rearm is legal. */
  {
    struct rearm_state rearm = {queue, 0};
    CHECK_OK(hyle_time_queue_schedule(queue, three, UINT64_C(700000)));
    CHECK_OK(hyle_time_queue_advance(&moved, queue, UINT64_C(700000),
                                     rearm_once, &rearm));
    CHECK(moved == 1 && rearm.visits == 1);
    CHECK(hyle_time_timer_is_scheduled(three));
    CHECK_OK(hyle_time_queue_advance(&moved, queue, UINT64_C(800000),
                                     rearm_once, &rearm));
    CHECK(moved == 1 && rearm.visits == 2);
    CHECK(!hyle_time_timer_is_scheduled(three));
    {
      moved = 99;
      hyle_status status =
          hyle_time_queue_advance(&moved, queue, UINT64_C(799999), NULL, NULL);
      CHECK(hyle_status_is_invalid_argument(status));
      CHECK(moved == 0);
      hyle_status_free(&status);
    }
  }

  /* A visitor may suppress a not-yet-delivered peer from the same batch.
   * The implementation must not retain a traversal pointer into that peer. */
  {
    struct pending_state pending = {queue, NULL, two, 0, PENDING_CANCEL, 0, 0};
    CHECK_OK(hyle_time_queue_schedule(queue, one, UINT64_C(810000)));
    CHECK_OK(hyle_time_queue_schedule(queue, two, UINT64_C(810000)));
    CHECK_OK(hyle_time_queue_advance(&moved, queue, UINT64_C(810000),
                                     mutate_pending_timer, &pending));
    CHECK(moved == 1 && pending.visits == 1 && pending.cancelled == 1);
    CHECK(hyle_time_queue_empty(queue));
  }
  {
    struct pending_state pending = {
        queue, NULL, two, UINT64_C(830000), PENDING_RESCHEDULE, 0, 0};
    CHECK_OK(hyle_time_queue_schedule(queue, one, UINT64_C(820000)));
    CHECK_OK(hyle_time_queue_schedule(queue, two, UINT64_C(820000)));
    CHECK_OK(hyle_time_queue_advance(&moved, queue, UINT64_C(820000),
                                     mutate_pending_timer, &pending));
    CHECK(moved == 1 && pending.visits == 1);
    CHECK(hyle_time_timer_is_scheduled(two));
    state.count = 0;
    state.sum = 0;
    CHECK_OK(hyle_time_queue_advance(&moved, queue, UINT64_C(830000),
                                     collect_timer, &state));
    CHECK(moved == 1 && state.count == 1 && state.sum == 2);
  }
  {
    hyle_time_timer* doomed = NULL;
    struct pending_state pending;
    CHECK_OK(hyle_time_timer_create(&doomed, (void*)(uintptr_t)9));
    pending =
        (struct pending_state){queue, NULL, doomed, 0, PENDING_DESTROY, 0, 0};
    CHECK_OK(hyle_time_queue_schedule(queue, one, UINT64_C(840000)));
    CHECK_OK(hyle_time_queue_schedule(queue, doomed, UINT64_C(840000)));
    CHECK_OK(hyle_time_queue_advance(&moved, queue, UINT64_C(840000),
                                     mutate_pending_timer, &pending));
    CHECK(moved == 1 && pending.visits == 1 && pending.target == NULL);
    CHECK(hyle_time_queue_empty(queue));
  }

  CHECK(hyle_time_common_queue_duration_ns(common) == 30);
  CHECK_OK(hyle_time_common_queue_schedule_after(common, one, 100));
  CHECK(hyle_time_timer_deadline_ns(one) == 130);
  CHECK_OK(hyle_time_common_queue_schedule_after(common, two, 200));
  CHECK(hyle_time_common_queue_size(common) == 2);
  CHECK(hyle_time_timer_deadline_ns(two) == 230);
  {
    hyle_status status = hyle_time_common_queue_schedule(common, two, 50);
    CHECK(hyle_status_is_invalid_argument(status));
    hyle_status_free(&status);
  }
  CHECK(hyle_time_timer_is_scheduled(two));
  CHECK(hyle_time_timer_deadline_ns(two) == 230);

  state.count = 0;
  state.sum = 0;
  CHECK_OK(hyle_time_common_queue_advance(&moved, common, 130, collect_timer,
                                          &state));
  CHECK(moved == 1 && state.count == 1 && state.sum == 4);
  CHECK(hyle_time_common_queue_next_deadline_ns(common) == 230);
  {
    int cancelled = 0;
    CHECK_OK(hyle_time_common_queue_cancel(&cancelled, common, two));
    CHECK(cancelled == 1);
  }
  CHECK(hyle_time_common_queue_empty(common));

  /* Common-timeout delivery uses the same mutation-safe visitor contract. */
  {
    struct pending_state pending = {
        NULL, common, two, UINT64_C(450), PENDING_REJECT_RESCHEDULE, 0, 0};
    CHECK_OK(hyle_time_common_queue_schedule(common, one, UINT64_C(400)));
    CHECK_OK(hyle_time_common_queue_schedule(common, two, UINT64_C(400)));
    CHECK_OK(hyle_time_common_queue_schedule(common, three, UINT64_C(500)));
    CHECK_OK(hyle_time_common_queue_advance(&moved, common, UINT64_C(400),
                                            mutate_pending_timer, &pending));
    CHECK(moved == 2 && pending.visits == 2);
    CHECK(!hyle_time_timer_is_scheduled(two));
    CHECK(hyle_time_timer_is_scheduled(three));
    CHECK_OK(hyle_time_common_queue_cancel(NULL, common, three));
  }
  {
    struct pending_state pending = {
        NULL, common, two, UINT64_C(550), PENDING_REJECT_RESCHEDULE_AFTER,
        0,    0};
    CHECK_OK(hyle_time_common_queue_schedule(common, one, UINT64_C(510)));
    CHECK_OK(hyle_time_common_queue_schedule(common, two, UINT64_C(510)));
    CHECK_OK(hyle_time_common_queue_schedule(common, three, UINT64_C(600)));
    CHECK_OK(hyle_time_common_queue_advance(&moved, common, UINT64_C(510),
                                            mutate_pending_timer, &pending));
    CHECK(moved == 2 && pending.visits == 2);
    CHECK(!hyle_time_timer_is_scheduled(two));
    CHECK(hyle_time_timer_is_scheduled(three));
    CHECK_OK(hyle_time_common_queue_cancel(NULL, common, three));
  }
  {
    struct pending_state pending = {NULL, common, two, 0, PENDING_CANCEL, 0, 0};
    CHECK_OK(hyle_time_common_queue_schedule(common, one, UINT64_C(300)));
    CHECK_OK(hyle_time_common_queue_schedule(common, two, UINT64_C(300)));
    CHECK_OK(hyle_time_common_queue_advance(&moved, common, UINT64_C(300),
                                            mutate_pending_timer, &pending));
    CHECK(moved == 1 && pending.visits == 1 && pending.cancelled == 1);
    CHECK(hyle_time_common_queue_empty(common));
  }
  {
    struct pending_state pending = {
        NULL, common, two, UINT64_C(320), PENDING_RESCHEDULE, 0, 0};
    CHECK_OK(hyle_time_common_queue_schedule(common, one, UINT64_C(310)));
    CHECK_OK(hyle_time_common_queue_schedule(common, two, UINT64_C(310)));
    CHECK_OK(hyle_time_common_queue_advance(&moved, common, UINT64_C(310),
                                            mutate_pending_timer, &pending));
    CHECK(moved == 1 && pending.visits == 1);
    CHECK(hyle_time_timer_is_scheduled(two));
    state.count = 0;
    state.sum = 0;
    CHECK_OK(hyle_time_common_queue_advance(&moved, common, UINT64_C(320),
                                            collect_timer, &state));
    CHECK(moved == 1 && state.count == 1 && state.sum == 2);
  }
  {
    hyle_time_timer* doomed = NULL;
    struct pending_state pending;
    CHECK_OK(hyle_time_timer_create(&doomed, (void*)(uintptr_t)9));
    pending =
        (struct pending_state){NULL, common, doomed, 0, PENDING_DESTROY, 0, 0};
    CHECK_OK(hyle_time_common_queue_schedule(common, one, UINT64_C(330)));
    CHECK_OK(hyle_time_common_queue_schedule(common, doomed, UINT64_C(330)));
    CHECK_OK(hyle_time_common_queue_advance(&moved, common, UINT64_C(330),
                                            mutate_pending_timer, &pending));
    CHECK(moved == 1 && pending.visits == 1 && pending.target == NULL);
    CHECK(hyle_time_common_queue_empty(common));
  }

  CHECK_OK(hyle_time_common_queue_schedule_after(common, one, 300));
  CHECK_OK(hyle_time_common_queue_schedule_after(common, two, 400));
  state.count = 0;
  state.sum = 0;
  CHECK_OK(
      hyle_time_common_queue_cancel_all(&moved, common, collect_timer, &state));
  CHECK(moved == 2 && state.count == 2 && state.sum == 6);
  CHECK(hyle_time_common_queue_empty(common));

  /* Destruction misuse is recoverable; explicit cancel_all detaches nodes. */
  CHECK_OK(hyle_time_queue_schedule(queue, one, UINT64_C(900000)));
  {
    hyle_status status = hyle_time_timer_destroy(one);
    CHECK(hyle_status_is_failed_precondition(status));
    hyle_status_free(&status);
    status = hyle_time_queue_destroy(queue);
    CHECK(hyle_status_is_failed_precondition(status));
    hyle_status_free(&status);
  }
  state.count = 0;
  state.sum = 0;
  CHECK_OK(hyle_time_queue_cancel_all(&moved, queue, collect_timer, &state));
  CHECK(moved == 1 && state.count == 1);
  CHECK(!hyle_time_timer_is_scheduled(one));

  CHECK(hyle_time_queue_size(NULL) == 0);
  CHECK(hyle_time_queue_empty(NULL));
  CHECK(hyle_time_queue_next_deadline_ns(NULL) == UINT64_MAX);
  {
    int cancelled = 99;
    hyle_status status = hyle_time_queue_schedule(NULL, three, 1);
    CHECK(hyle_status_is_invalid_argument(status));
    hyle_status_free(&status);
    moved = 99;
    status = hyle_time_queue_advance(&moved, NULL, 1, NULL, NULL);
    CHECK(hyle_status_is_invalid_argument(status));
    CHECK(moved == 0);
    hyle_status_free(&status);
    moved = 99;
    status = hyle_time_queue_cancel_all(&moved, NULL, NULL, NULL);
    CHECK(hyle_status_is_invalid_argument(status));
    CHECK(moved == 0);
    hyle_status_free(&status);
    status = hyle_time_queue_cancel(&cancelled, queue, NULL);
    CHECK(hyle_status_is_invalid_argument(status));
    CHECK(cancelled == 0);
    hyle_status_free(&status);
    moved = 99;
    status = hyle_time_common_queue_advance(&moved, NULL, 1, NULL, NULL);
    CHECK(hyle_status_is_invalid_argument(status));
    CHECK(moved == 0);
    hyle_status_free(&status);
    moved = 99;
    status = hyle_time_common_queue_cancel_all(&moved, NULL, NULL, NULL);
    CHECK(hyle_status_is_invalid_argument(status));
    CHECK(moved == 0);
    hyle_status_free(&status);
  }
  CHECK(hyle_time_timer_user_data(NULL) == NULL);
  CHECK(!hyle_time_timer_is_scheduled(NULL));
  CHECK(hyle_time_common_queue_size(NULL) == 0);
  CHECK(hyle_time_common_queue_empty(NULL));
  CHECK(hyle_time_common_queue_next_deadline_ns(NULL) == UINT64_MAX);
  CHECK(hyle_time_common_queue_duration_ns(NULL) == 0);

  CHECK_OK(hyle_time_timer_destroy(three));
  CHECK_OK(hyle_time_timer_destroy(two));
  CHECK_OK(hyle_time_timer_destroy(one));
  CHECK_OK(hyle_time_common_queue_destroy(common));
  CHECK_OK(hyle_time_queue_destroy(queue));
  CHECK_OK(hyle_time_common_queue_destroy(NULL));
  CHECK_OK(hyle_time_queue_destroy(NULL));
  CHECK_OK(hyle_time_timer_destroy(NULL));
  CHECK_OK(hyle_time_timer_deinit(NULL));

  if (g_fails) {
    fprintf(stderr, "c_time_test: %d failed\n", g_fails);
    return 1;
  }
  printf("c_time_test: all checks passed\n");
  return 0;
}
