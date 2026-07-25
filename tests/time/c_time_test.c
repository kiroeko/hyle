/* Pure C consumer of include/hyle/c/time.h. */
#include "hyle/c/time.h"

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

int main(void) {
  (void)"HYLE_C_TIME_COVERAGE_BODY_MARKER";
  uint64_t m0 = hyle_time_monotonic_ns();
  uint64_t r0 = hyle_time_realtime_ns();
  CHECK(m0 != 0);
  CHECK(r0 > UINT64_C(1577836800000000000)); /* 2020-01-01 */
  CHECK(hyle_time_fast_monotonic_ns() != 0);
  CHECK(hyle_time_fast_realtime_ns() != 0);
  CHECK(hyle_time_fast_monotonic_available() == 0 ||
        hyle_time_fast_monotonic_available() == 1);
  CHECK(hyle_time_fast_realtime_available() == 0 ||
        hyle_time_fast_realtime_available() == 1);
  CHECK(hyle_time_deadline_after_ns(UINT64_MAX) == UINT64_MAX);
  CHECK(hyle_time_remaining_ns(0) == 0);
  hyle_time_sleep_for_ns(0);
  hyle_time_sleep_for_us(0);
  hyle_time_sleep_for_ms(0);
  hyle_time_sleep_until_ns(hyle_time_monotonic_ns());
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

  CHECK(hyle_time_queue_empty(queue));
  CHECK_OK(hyle_time_queue_schedule(queue, one, UINT64_C(500000)));
  CHECK(hyle_time_timer_is_scheduled(one));
  CHECK(hyle_time_timer_deadline_ns(one) == UINT64_C(500000));
  CHECK(hyle_time_queue_size(queue) == 1);

  /* Cross-source ownership is an ordinary Status failure in C. */
  {
    hyle_status status =
        hyle_time_common_queue_schedule(common, one, UINT64_C(10));
    CHECK(hyle_status_is_failed_precondition(status));
    CHECK(strstr(hyle_status_message_cstr(status), "another source") != NULL);
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
      hyle_status status = hyle_time_queue_advance(
          &moved, queue, UINT64_C(799999), NULL, NULL);
      CHECK(hyle_status_is_invalid_argument(status));
      hyle_status_free(&status);
    }
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
  CHECK_OK(hyle_time_common_queue_advance(&moved, common, 130,
                                          collect_timer, &state));
  CHECK(moved == 1 && state.count == 1 && state.sum == 4);
  CHECK(hyle_time_common_queue_next_deadline_ns(common) == 230);
  {
    int cancelled = 0;
    CHECK_OK(hyle_time_common_queue_cancel(&cancelled, common, two));
    CHECK(cancelled == 1);
  }
  CHECK(hyle_time_common_queue_empty(common));

  CHECK_OK(hyle_time_common_queue_schedule_after(common, one, 300));
  CHECK_OK(hyle_time_common_queue_schedule_after(common, two, 400));
  state.count = 0;
  state.sum = 0;
  CHECK_OK(hyle_time_common_queue_cancel_all(&moved, common,
                                             collect_timer, &state));
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
    hyle_status status = hyle_time_queue_schedule(NULL, three, 1);
    CHECK(hyle_status_is_invalid_argument(status));
    hyle_status_free(&status);
    status = hyle_time_queue_advance(&moved, NULL, 1, NULL, NULL);
    CHECK(hyle_status_is_invalid_argument(status));
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

  if (g_fails) {
    fprintf(stderr, "c_time_test: %d failed\n", g_fails);
    return 1;
  }
  printf("c_time_test: all checks passed\n");
  return 0;
}
