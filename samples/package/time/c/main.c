#include "hyle/c/time.h"

#include <stdint.h>

static void count_timer(hyle_time_timer* timer, void* context) {
  size_t* count = (size_t*)context;
  if (!hyle_time_timer_is_scheduled(timer)) ++*count;
}

static int consume(hyle_status status) {
  const int ok = hyle_status_is_ok(status);
  hyle_status_free(&status);
  return ok;
}

int main(void) {
  hyle_unix_timestamp stamp = HYLE_UNIX_TIMESTAMP_INIT;
  hyle_time_queue* queue = NULL;
  hyle_time_timer* timer = NULL;
  size_t fired = 0;
  size_t visited = 0;
  const uint64_t profile_start =
      hyle_time_profiling_measurement_start_ns();
  const uint64_t profile_stop =
      hyle_time_profiling_measurement_stop_ns();

  if (profile_start == 0 || profile_stop == 0) return 10;

  hyle_time_realtime_timestamp(&stamp);
  if (!hyle_unix_timestamp_is_valid(stamp)) return 1;
  if (!consume(hyle_time_queue_create_at(&queue, 0))) return 2;
  if (!consume(hyle_time_timer_create(&timer, (void*)(uintptr_t)7))) return 3;
  if (!consume(hyle_time_queue_schedule_after_at(queue, timer, 1, 0))) {
    return 4;
  }
  if (!consume(hyle_time_queue_advance(&fired, queue, 1, count_timer,
                                       &visited))) {
    return 5;
  }
  if (fired != 1 || visited != 1) return 6;
  if (!consume(hyle_time_timer_destroy(timer))) return 7;
  if (!consume(hyle_time_queue_destroy(queue))) return 8;
  return hyle_time_monotonic_ns() == 0 ? 9 : 0;
}
