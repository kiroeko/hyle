#include "hyle/time/clock.h"
#include "hyle/time/timer_queue.h"

#include <chrono>

int main() {
  using namespace std::chrono_literals;
  const auto profile_start =
      hyle::time::profiling_measurement_start_ns();
  const auto profile_stop = hyle::time::profiling_measurement_stop_ns();
  if (profile_start == 0 || profile_stop == 0) return 4;

  hyle::time::TimerQueue queue(0);
  hyle::time::TimerEntry timer;
  queue.schedule_after(&timer, 1ns, hyle::time::MonotonicTimePoint{});
  hyle::time::TimerList fired;
  if (queue.advance(1, fired) != 1 || fired.pop_front() != &timer) return 1;

  const auto stamp = hyle::time::realtime_timestamp();
  if (!hyle::time::unix_timestamp_valid(stamp)) return 2;
  const auto deadline = hyle::time::deadline_after(1ms);
  return deadline < hyle::time::monotonic_now() ? 3 : 0;
}
