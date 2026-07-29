#include "hyle/c/time.h"

#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/error/status.h"
#include "hyle/time/clock.h"
#include "hyle/time/common_timeout_queue.h"
#include "hyle/time/timer_queue.h"
#include "../error/status_access.h"

struct hyle_time_timer_o {
  hyle::time::TimerEntry entry;
  void* user_data = nullptr;
  bool caller_storage = false;

  explicit hyle_time_timer_o(void* data, bool in_place = false) noexcept
      : entry(this), user_data(data), caller_storage(in_place) {}
};

static_assert(std::is_standard_layout_v<hyle_time_timer_o>);
static_assert(sizeof(hyle_time_timer_o) % alignof(hyle_time_timer_o) == 0);
// Replacing the duplicate C owner pointer with a storage-kind byte must not
// make every timer larger. The native TimerEntry remains unchanged.
static_assert(sizeof(hyle_time_timer_o) <=
              sizeof(hyle::time::TimerEntry) + 2 * sizeof(void*));

struct hyle_time_queue_o {
  hyle::time::TimerQueue queue;
  explicit hyle_time_queue_o(uint64_t now_ns) noexcept : queue(now_ns) {}
};

struct hyle_time_common_queue_o {
  hyle::time::CommonTimeoutQueue queue;
  explicit hyle_time_common_queue_o(uint64_t duration_ns) noexcept
      : queue(duration_ns) {}
};

namespace {

hyle_status steal_status(hyle::Status&& status) noexcept {
  return hyle_status{hyle::detail::StatusAccess::release(status)};
}

hyle::Status validate_pair(const void* source,
                           const hyle_time_timer* timer) noexcept {
  if (source == nullptr || timer == nullptr) {
    return hyle::Status::InvalidArgument(
        "time source and timer must be non-null");
  }
  if (timer->entry.scheduled() &&
      !hyle::time::detail::TimerEntryAccess::owned_by(timer->entry, source)) {
    return hyle::Status::FailedPrecondition(
        "timer is already scheduled on another source");
  }
  return hyle::Status{};
}

size_t visit_list(hyle::time::TimerList& entries,
                  hyle_time_timer_visit_fn visit, void* ctx) noexcept {
  size_t count = 0;
  entries.consume_all([&](hyle::time::TimerEntry* entry) noexcept {
    auto* timer = static_cast<hyle_time_timer*>(entry->data());
    HYLE_FATAL_ASSERT(timer != nullptr);
    ++count;
    if (visit != nullptr) visit(timer, ctx);
  });
  return count;
}

template <class Source>
size_t advance_and_visit(Source& source, uint64_t now_ns,
                         hyle_time_timer_visit_fn visit, void* ctx) noexcept {
  hyle::time::TimerList fired;
  source.advance(now_ns, fired);
  return visit_list(fired, visit, ctx);
}

template <class Source>
size_t cancel_all_and_visit(Source& source, hyle_time_timer_visit_fn visit,
                            void* ctx) noexcept {
  hyle::time::TimerList cancelled;
  source.cancel_all(cancelled);
  return visit_list(cancelled, visit, ctx);
}

}  // namespace

extern "C" {

uint64_t hyle_time_monotonic_ns(void) {
  return hyle::time::monotonic_ns();
}
uint64_t hyle_time_profiling_clock_ns(void) {
  return hyle::time::profiling_clock_ns();
}
uint64_t hyle_time_profiling_measurement_start_ns(void) {
  return hyle::time::profiling_measurement_start_ns();
}
uint64_t hyle_time_profiling_measurement_stop_ns(void) {
  return hyle::time::profiling_measurement_stop_ns();
}
int hyle_time_profiling_clock_available(void) {
  return hyle::time::profiling_clock_available() ? 1 : 0;
}
uint64_t hyle_time_realtime_ns(void) {
  return hyle::time::realtime_ns();
}
void hyle_time_realtime_timestamp(hyle_unix_timestamp* out) {
  if (out != nullptr) *out = hyle::time::realtime_timestamp();
}
uint64_t hyle_time_coarse_realtime_ns(void) {
  return hyle::time::coarse_realtime_ns();
}
int hyle_time_coarse_realtime_available(void) {
  return hyle::time::coarse_realtime_available() ? 1 : 0;
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

hyle_status hyle_time_timer_create(hyle_time_timer** out, void* user_data) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("timer output must be non-null"));
  }
  *out = nullptr;
  auto* timer = new (std::nothrow) hyle_time_timer(user_data);
  if (timer == nullptr) {
    return steal_status(
        hyle::Status::ResourceExhausted("timer allocation failed"));
  }
  *out = timer;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_timer_destroy(hyle_time_timer* timer) {
  if (timer == nullptr) return steal_status(hyle::Status{});
  if (timer->entry.scheduled()) {
    return steal_status(hyle::Status::FailedPrecondition(
        "scheduled timer must be cancelled before destroy"));
  }
  if (timer->caller_storage) {
    return steal_status(hyle::Status::FailedPrecondition(
        "caller-storage timer must be released with timer_deinit"));
  }
  delete timer;
  return steal_status(hyle::Status{});
}

size_t hyle_time_timer_storage_size(void) { return sizeof(hyle_time_timer); }

size_t hyle_time_timer_storage_alignment(void) {
  return alignof(hyle_time_timer);
}

hyle_status hyle_time_timer_init(hyle_time_timer** out, void* storage,
                                 size_t storage_size, void* user_data) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "timer output must be non-null"));
  }
  *out = nullptr;
  if (storage == nullptr || storage_size < sizeof(hyle_time_timer)) {
    return steal_status(hyle::Status::InvalidArgument(
        "timer storage is null or smaller than timer_storage_size"));
  }
  if (reinterpret_cast<std::uintptr_t>(storage) % alignof(hyle_time_timer) !=
      0) {
    return steal_status(hyle::Status::InvalidArgument(
        "timer storage does not satisfy timer_storage_alignment"));
  }
  *out = ::new (storage) hyle_time_timer(user_data, true);
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_timer_deinit(hyle_time_timer* timer) {
  if (timer == nullptr) return steal_status(hyle::Status{});
  if (timer->entry.scheduled()) {
    return steal_status(hyle::Status::FailedPrecondition(
        "scheduled timer must be cancelled before deinit"));
  }
  if (!timer->caller_storage) {
    return steal_status(hyle::Status::FailedPrecondition(
        "created timer must be released with timer_destroy"));
  }
  std::destroy_at(timer);
  return steal_status(hyle::Status{});
}

void* hyle_time_timer_user_data(const hyle_time_timer* timer) {
  return timer != nullptr ? timer->user_data : nullptr;
}

void hyle_time_timer_set_user_data(hyle_time_timer* timer, void* user_data) {
  if (timer != nullptr) timer->user_data = user_data;
}

uint64_t hyle_time_timer_deadline_ns(const hyle_time_timer* timer) {
  return timer != nullptr ? timer->entry.deadline_ns() : 0;
}

int hyle_time_timer_is_scheduled(const hyle_time_timer* timer) {
  return timer != nullptr && timer->entry.scheduled() ? 1 : 0;
}

hyle_status hyle_time_queue_create_at(hyle_time_queue** out,
                                      uint64_t now_ns) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("timer queue output must be non-null"));
  }
  *out = nullptr;
  auto* queue = new (std::nothrow) hyle_time_queue(now_ns);
  if (queue == nullptr) {
    return steal_status(
        hyle::Status::ResourceExhausted("timer queue allocation failed"));
  }
  *out = queue;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_queue_create(hyle_time_queue** out) {
  return hyle_time_queue_create_at(out, hyle::time::monotonic_ns());
}

hyle_status hyle_time_queue_destroy(hyle_time_queue* queue) {
  if (queue == nullptr) return steal_status(hyle::Status{});
  if (!queue->queue.empty()) {
    return steal_status(hyle::Status::FailedPrecondition(
        "timer queue must be empty before destroy"));
  }
  delete queue;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_queue_schedule(hyle_time_queue* queue,
                                     hyle_time_timer* timer,
                                     uint64_t deadline_ns) {
  hyle::Status status = validate_pair(queue, timer);
  if (!status.ok()) return steal_status(std::move(status));
  queue->queue.schedule(&timer->entry, deadline_ns);
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_queue_cancel(int* out_cancelled,
                                   hyle_time_queue* queue,
                                   hyle_time_timer* timer) {
  hyle::Status status = validate_pair(queue, timer);
  if (!status.ok()) return steal_status(std::move(status));
  const bool cancelled = queue->queue.cancel(&timer->entry);
  if (out_cancelled != nullptr) *out_cancelled = cancelled ? 1 : 0;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_queue_advance(size_t* out_fired,
                                    hyle_time_queue* queue, uint64_t now_ns,
                                    hyle_time_timer_visit_fn visit, void* ctx) {
  if (out_fired == nullptr || queue == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "timer queue and fired output must be non-null"));
  }
  if (now_ns < queue->queue.now_ns()) {
    return steal_status(hyle::Status::InvalidArgument(
        "timer queue advance timestamp must be non-decreasing"));
  }
  *out_fired = advance_and_visit(queue->queue, now_ns, visit, ctx);
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_queue_cancel_all(size_t* out_cancelled,
                                       hyle_time_queue* queue,
                                       hyle_time_timer_visit_fn visit,
                                       void* ctx) {
  if (out_cancelled == nullptr || queue == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "timer queue and cancelled output must be non-null"));
  }
  *out_cancelled = cancel_all_and_visit(queue->queue, visit, ctx);
  return steal_status(hyle::Status{});
}

uint64_t hyle_time_queue_next_deadline_ns(const hyle_time_queue* queue) {
  return queue != nullptr ? queue->queue.next_deadline_ns() : UINT64_MAX;
}
size_t hyle_time_queue_size(const hyle_time_queue* queue) {
  return queue != nullptr ? queue->queue.size() : 0;
}
int hyle_time_queue_empty(const hyle_time_queue* queue) {
  return queue == nullptr || queue->queue.empty() ? 1 : 0;
}

hyle_status hyle_time_common_queue_create(hyle_time_common_queue** out,
                                          uint64_t duration_ns) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "common timeout queue output must be non-null"));
  }
  *out = nullptr;
  auto* queue = new (std::nothrow) hyle_time_common_queue(duration_ns);
  if (queue == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "common timeout queue allocation failed"));
  }
  *out = queue;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_common_queue_destroy(hyle_time_common_queue* queue) {
  if (queue == nullptr) return steal_status(hyle::Status{});
  if (!queue->queue.empty()) {
    return steal_status(hyle::Status::FailedPrecondition(
        "common timeout queue must be empty before destroy"));
  }
  delete queue;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_common_queue_schedule(hyle_time_common_queue* queue,
                                            hyle_time_timer* timer,
                                            uint64_t deadline_ns) {
  hyle::Status status = validate_pair(queue, timer);
  if (!status.ok()) return steal_status(std::move(status));
  const bool ok = queue->queue.try_schedule(&timer->entry, deadline_ns);
  if (!ok) {
    return steal_status(hyle::Status::InvalidArgument(
        "common timeout deadlines must be non-decreasing"));
  }
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_common_queue_schedule_after(
    hyle_time_common_queue* queue, hyle_time_timer* timer, uint64_t now_ns) {
  hyle::Status status = validate_pair(queue, timer);
  if (!status.ok()) return steal_status(std::move(status));
  const bool ok = queue->queue.try_schedule_after(&timer->entry, now_ns);
  if (!ok) {
    return steal_status(hyle::Status::InvalidArgument(
        "common timeout deadlines must be non-decreasing"));
  }
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_common_queue_cancel(int* out_cancelled,
                                          hyle_time_common_queue* queue,
                                          hyle_time_timer* timer) {
  hyle::Status status = validate_pair(queue, timer);
  if (!status.ok()) return steal_status(std::move(status));
  const bool cancelled = queue->queue.cancel(&timer->entry);
  if (out_cancelled != nullptr) *out_cancelled = cancelled ? 1 : 0;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_common_queue_advance(
    size_t* out_fired, hyle_time_common_queue* queue, uint64_t now_ns,
    hyle_time_timer_visit_fn visit, void* ctx) {
  if (out_fired == nullptr || queue == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "common timeout queue and fired output must be non-null"));
  }
  *out_fired = advance_and_visit(queue->queue, now_ns, visit, ctx);
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_common_queue_cancel_all(
    size_t* out_cancelled, hyle_time_common_queue* queue,
    hyle_time_timer_visit_fn visit, void* ctx) {
  if (out_cancelled == nullptr || queue == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "common timeout queue and cancelled output must be non-null"));
  }
  *out_cancelled = cancel_all_and_visit(queue->queue, visit, ctx);
  return steal_status(hyle::Status{});
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
