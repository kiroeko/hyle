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

struct visit_state {
  size_t count;
  uintptr_t sum;
};

struct rearm_state {
  hyle_time_wheel* wheel;
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
    CHECK(hyle_time_wheel_schedule(state->wheel, timer, UINT64_C(7000001)));
  }
}

int main(void) {
  uint64_t m0 = hyle_time_monotonic_ns();
  uint64_t r0 = hyle_time_realtime_ns();
  CHECK(m0 != 0);
  CHECK(r0 > UINT64_C(1577836800000000000)); /* 2020-01-01 */
  CHECK(hyle_time_fast_monotonic_ns() != 0);
  CHECK(hyle_time_fast_realtime_ns() != 0);
  CHECK(hyle_time_fast_monotonic_available() == 0 ||
        hyle_time_fast_monotonic_available() == 1);
  CHECK(hyle_time_deadline_after_ns(UINT64_MAX) == UINT64_MAX);
  CHECK(hyle_time_remaining_ns(0) == 0);
  hyle_time_sleep_for_ns(0);
  hyle_time_sleep_for_us(0);
  hyle_time_sleep_for_ms(0);
  hyle_time_sleep_until_ns(hyle_time_monotonic_ns());

  hyle_time_wheel* wheel = hyle_time_wheel_create(0);
  hyle_time_common_queue* common =
      hyle_time_common_queue_create(0, UINT64_C(30));
  hyle_time_timer* one = hyle_time_timer_create((void*)(uintptr_t)1);
  hyle_time_timer* two = hyle_time_timer_create((void*)(uintptr_t)2);
  hyle_time_timer* three = hyle_time_timer_create((void*)(uintptr_t)3);
  CHECK(wheel != NULL && common != NULL);
  CHECK(one != NULL && two != NULL && three != NULL);
  CHECK(strcmp(hyle_time_last_error(), "") == 0);

  CHECK(hyle_time_wheel_empty(wheel));
  CHECK(hyle_time_wheel_schedule(wheel, one, UINT64_C(5000000)));
  CHECK(hyle_time_timer_is_scheduled(one));
  CHECK(hyle_time_timer_deadline_ns(one) == UINT64_C(5000000));
  CHECK(hyle_time_wheel_size(wheel) == 1);
  /* Cross-source ownership is a recoverable C-boundary error. */
  CHECK(!hyle_time_common_queue_schedule(common, one, UINT64_C(10)));
  CHECK(strstr(hyle_time_last_error(), "another source") != NULL);

  struct visit_state state = {0, 0};
  CHECK(hyle_time_wheel_advance(wheel, UINT64_C(4000000), collect_timer,
                                &state) == 0);
  CHECK(hyle_time_wheel_advance(wheel, UINT64_C(5000000), collect_timer,
                                &state) == 1);
  CHECK(strcmp(hyle_time_last_error(), "") == 0);
  CHECK(state.count == 1 && state.sum == 1);
  CHECK(hyle_time_wheel_empty(wheel));

  hyle_time_timer_set_user_data(one, (void*)(uintptr_t)4);
  CHECK((uintptr_t)hyle_time_timer_user_data(one) == 4);
  CHECK(hyle_time_wheel_schedule(wheel, one, UINT64_C(9000000)));
  CHECK(hyle_time_wheel_cancel(wheel, one));
  CHECK(!hyle_time_timer_is_scheduled(one));
  CHECK(hyle_time_wheel_schedule_after(wheel, one, 1));
  CHECK(hyle_time_wheel_cancel(wheel, one));

  /* advance clears source ownership before visiting, so callback rearm is legal. */
  {
    struct rearm_state rearm = {wheel, 0};
    CHECK(hyle_time_wheel_schedule(wheel, three, UINT64_C(7000000)));
    CHECK(hyle_time_wheel_advance(wheel, UINT64_C(7000000), rearm_once,
                                  &rearm) == 1);
    CHECK(rearm.visits == 1);
    CHECK(hyle_time_timer_is_scheduled(three));
    CHECK(hyle_time_wheel_advance(wheel, UINT64_C(8000000), rearm_once,
                                  &rearm) == 1);
    CHECK(rearm.visits == 2);
    CHECK(!hyle_time_timer_is_scheduled(three));
  }

  CHECK(hyle_time_common_queue_duration_ns(common) == 30);
  CHECK(hyle_time_common_queue_schedule_after(common, one, 100));
  CHECK(hyle_time_timer_deadline_ns(one) == 130);
  CHECK(hyle_time_common_queue_schedule_after(common, two, 200));
  CHECK(hyle_time_common_queue_size(common) == 2);
  CHECK(hyle_time_timer_deadline_ns(two) == 230);
  CHECK(!hyle_time_common_queue_schedule(common, two, 50));
  CHECK(hyle_time_timer_is_scheduled(two));
  CHECK(hyle_time_timer_deadline_ns(two) == 230);
  CHECK(!hyle_time_common_queue_schedule(common, three, 50));
  CHECK(strstr(hyle_time_last_error(), "non-decreasing") != NULL);

  state.count = 0;
  state.sum = 0;
  CHECK(hyle_time_common_queue_advance(common, 130, collect_timer, &state) ==
        1);
  CHECK(state.count == 1 && state.sum == 4);
  CHECK(hyle_time_common_queue_next_deadline_ns(common) == 230);
  CHECK(hyle_time_common_queue_cancel(common, two));
  CHECK(hyle_time_common_queue_empty(common));
  CHECK(hyle_time_common_queue_schedule_now(common, one));
  CHECK(hyle_time_common_queue_cancel(common, one));

  CHECK(hyle_time_wheel_size(NULL) == 0);
  CHECK(hyle_time_wheel_empty(NULL));
  CHECK(hyle_time_wheel_next_deadline_ns(NULL) == UINT64_MAX);
  CHECK(!hyle_time_wheel_schedule(NULL, three, 1));
  CHECK(hyle_time_last_error()[0] != '\0');
  CHECK(hyle_time_wheel_advance(NULL, 1, NULL, NULL) == 0);
  CHECK(hyle_time_timer_user_data(NULL) == NULL);
  CHECK(!hyle_time_timer_is_scheduled(NULL));
  CHECK(hyle_time_common_queue_size(NULL) == 0);
  CHECK(hyle_time_common_queue_empty(NULL));
  CHECK(hyle_time_common_queue_next_deadline_ns(NULL) == UINT64_MAX);
  CHECK(hyle_time_common_queue_duration_ns(NULL) == 0);
  CHECK(!hyle_time_common_queue_schedule(NULL, three, 1));
  CHECK(hyle_time_common_queue_advance(NULL, 1, NULL, NULL) == 0);

  hyle_time_timer_destroy(three);
  hyle_time_timer_destroy(two);
  hyle_time_timer_destroy(one);
  hyle_time_common_queue_destroy(common);
  hyle_time_wheel_destroy(wheel);
  hyle_time_common_queue_destroy(NULL);
  hyle_time_wheel_destroy(NULL);
  hyle_time_timer_destroy(NULL);

  if (g_fails) {
    fprintf(stderr, "c_time_test: %d failed\n", g_fails);
    return 1;
  }
  printf("c_time_test: all checks passed\n");
  return 0;
}
