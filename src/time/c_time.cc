#include "hyle/c/time.h"

#include <new>

#include "hyle/macros.h"
#include "hyle/time/clock.h"
#include "hyle/time/common_timeout_queue.h"
#include "hyle/time/timer_wheel.h"

struct hyle_time_timer_o {
  hyle::time::TimerEntry entry;
  void* user_data = nullptr;
  const void* c_owner = nullptr;

  explicit hyle_time_timer_o(void* data) noexcept
      : entry(this), user_data(data) {}
};

struct hyle_time_wheel_o {
  hyle::time::TimerWheel wheel;
  explicit hyle_time_wheel_o(uint64_t start_ns) noexcept : wheel(start_ns) {}
};

struct hyle_time_common_queue_o {
  hyle::time::CommonTimeoutQueue queue;
  hyle_time_common_queue_o(uint64_t start_ns, uint64_t duration_ns) noexcept
      : queue(start_ns, duration_ns) {}
};

namespace {

thread_local const char* g_last_error = "";

void clear_error() noexcept { g_last_error = ""; }
void set_error(const char* message) noexcept { g_last_error = message; }

bool valid_pair(const void* source, const hyle_time_timer* timer) noexcept {
  if (source == nullptr || timer == nullptr) {
    set_error("time source and timer must be non-null");
    return false;
  }
  if (timer->c_owner != nullptr && timer->c_owner != source) {
    set_error("timer is already scheduled on another source");
    return false;
  }
  return true;
}

template <class Source>
size_t advance_and_visit(Source& source, uint64_t now_ns,
                         hyle_time_timer_visit_fn visit, void* ctx) noexcept {
  hyle::time::TimerList fired;
  source.advance(now_ns, fired);
  size_t count = 0;
  fired.consume_all([&](hyle::time::TimerEntry* entry) noexcept {
    auto* timer = static_cast<hyle_time_timer*>(entry->data);
    HYLE_FATAL_ASSERT(timer != nullptr);
    timer->c_owner = nullptr;
    ++count;
    if (visit != nullptr) visit(timer, ctx);
  });
  return count;
}

}  // namespace

extern "C" {

const char* hyle_time_last_error(void) { return g_last_error; }

uint64_t hyle_time_monotonic_ns(void) {
  return hyle::time::monotonic_ns();
}
uint64_t hyle_time_fast_monotonic_ns(void) {
  return hyle::time::fast_monotonic_ns();
}
int hyle_time_fast_monotonic_available(void) {
  return hyle::time::fast_monotonic_available() ? 1 : 0;
}
uint64_t hyle_time_realtime_ns(void) {
  return hyle::time::realtime_ns();
}
uint64_t hyle_time_fast_realtime_ns(void) {
  return hyle::time::fast_realtime_ns();
}
void hyle_time_sleep_for_ns(uint64_t ns) {
  hyle::time::sleep_for_ns(ns);
}
void hyle_time_sleep_until_ns(uint64_t deadline_ns) {
  hyle::time::sleep_until_ns(deadline_ns);
}
uint64_t hyle_time_deadline_after_ns(uint64_t timeout_ns) {
  return hyle::time::deadline_after_ns(timeout_ns);
}
uint64_t hyle_time_remaining_ns(uint64_t deadline_ns) {
  return hyle::time::remaining_ns(deadline_ns);
}

hyle_time_timer* hyle_time_timer_create(void* user_data) {
  auto* timer = new (std::nothrow) hyle_time_timer(user_data);
  if (timer == nullptr) {
    set_error("timer allocation failed");
    return nullptr;
  }
  clear_error();
  return timer;
}

void hyle_time_timer_destroy(hyle_time_timer* timer) {
  if (timer != nullptr) HYLE_FATAL_ASSERT(timer->c_owner == nullptr);
  delete timer;
}

void* hyle_time_timer_user_data(const hyle_time_timer* timer) {
  return timer != nullptr ? timer->user_data : nullptr;
}

void hyle_time_timer_set_user_data(hyle_time_timer* timer, void* user_data) {
  if (timer != nullptr) timer->user_data = user_data;
}

uint64_t hyle_time_timer_deadline_ns(const hyle_time_timer* timer) {
  return timer != nullptr ? timer->entry.deadline_ns : 0;
}

int hyle_time_timer_is_scheduled(const hyle_time_timer* timer) {
  return timer != nullptr && timer->entry.scheduled() ? 1 : 0;
}

hyle_time_wheel* hyle_time_wheel_create(uint64_t start_ns) {
  auto* wheel = new (std::nothrow) hyle_time_wheel(start_ns);
  if (wheel == nullptr) {
    set_error("timer wheel allocation failed");
    return nullptr;
  }
  clear_error();
  return wheel;
}

void hyle_time_wheel_destroy(hyle_time_wheel* wheel) { delete wheel; }

int hyle_time_wheel_schedule(hyle_time_wheel* wheel, hyle_time_timer* timer,
                             uint64_t deadline_ns) {
  if (!valid_pair(wheel, timer)) return 0;
  wheel->wheel.schedule(&timer->entry, deadline_ns);
  timer->c_owner = wheel;
  clear_error();
  return 1;
}

int hyle_time_wheel_cancel(hyle_time_wheel* wheel, hyle_time_timer* timer) {
  if (!valid_pair(wheel, timer)) return 0;
  wheel->wheel.cancel(&timer->entry);
  timer->c_owner = nullptr;
  clear_error();
  return 1;
}

size_t hyle_time_wheel_advance(hyle_time_wheel* wheel, uint64_t now_ns,
                               hyle_time_timer_visit_fn visit, void* ctx) {
  if (wheel == nullptr) {
    set_error("timer wheel must be non-null");
    return 0;
  }
  const size_t count = advance_and_visit(wheel->wheel, now_ns, visit, ctx);
  clear_error();
  return count;
}

uint64_t hyle_time_wheel_next_deadline_ns(const hyle_time_wheel* wheel) {
  return wheel != nullptr ? wheel->wheel.next_deadline_ns() : UINT64_MAX;
}
size_t hyle_time_wheel_size(const hyle_time_wheel* wheel) {
  return wheel != nullptr ? wheel->wheel.size() : 0;
}
int hyle_time_wheel_empty(const hyle_time_wheel* wheel) {
  return wheel == nullptr || wheel->wheel.empty() ? 1 : 0;
}

hyle_time_common_queue* hyle_time_common_queue_create(uint64_t start_ns,
                                                       uint64_t duration_ns) {
  auto* queue =
      new (std::nothrow) hyle_time_common_queue(start_ns, duration_ns);
  if (queue == nullptr) {
    set_error("common timeout queue allocation failed");
    return nullptr;
  }
  clear_error();
  return queue;
}

void hyle_time_common_queue_destroy(hyle_time_common_queue* queue) {
  delete queue;
}

int hyle_time_common_queue_schedule(hyle_time_common_queue* queue,
                                    hyle_time_timer* timer,
                                    uint64_t deadline_ns) {
  if (!valid_pair(queue, timer)) return 0;
  const bool ok = queue->queue.try_schedule(&timer->entry, deadline_ns);
  timer->c_owner =
      timer->entry.scheduled() ? static_cast<const void*>(queue) : nullptr;
  if (!ok) {
    set_error("common timeout deadlines must be non-decreasing");
    return 0;
  }
  clear_error();
  return 1;
}

int hyle_time_common_queue_schedule_after(hyle_time_common_queue* queue,
                                          hyle_time_timer* timer,
                                          uint64_t now_ns) {
  if (!valid_pair(queue, timer)) return 0;
  const bool ok = queue->queue.try_schedule_after(&timer->entry, now_ns);
  timer->c_owner =
      timer->entry.scheduled() ? static_cast<const void*>(queue) : nullptr;
  if (!ok) {
    set_error("common timeout deadlines must be non-decreasing");
    return 0;
  }
  clear_error();
  return 1;
}

int hyle_time_common_queue_cancel(hyle_time_common_queue* queue,
                                  hyle_time_timer* timer) {
  if (!valid_pair(queue, timer)) return 0;
  queue->queue.cancel(&timer->entry);
  timer->c_owner = nullptr;
  clear_error();
  return 1;
}

size_t hyle_time_common_queue_advance(hyle_time_common_queue* queue,
                                      uint64_t now_ns,
                                      hyle_time_timer_visit_fn visit,
                                      void* ctx) {
  if (queue == nullptr) {
    set_error("common timeout queue must be non-null");
    return 0;
  }
  const size_t count = advance_and_visit(queue->queue, now_ns, visit, ctx);
  clear_error();
  return count;
}

uint64_t hyle_time_common_queue_next_deadline_ns(
    const hyle_time_common_queue* queue) {
  return queue != nullptr ? queue->queue.next_deadline_ns() : UINT64_MAX;
}
size_t hyle_time_common_queue_size(const hyle_time_common_queue* queue) {
  return queue != nullptr ? queue->queue.size() : 0;
}
int hyle_time_common_queue_empty(const hyle_time_common_queue* queue) {
  return queue == nullptr || queue->queue.empty() ? 1 : 0;
}
uint64_t hyle_time_common_queue_duration_ns(
    const hyle_time_common_queue* queue) {
  return queue != nullptr ? queue->queue.duration_ns() : 0;
}

}  // extern "C"
